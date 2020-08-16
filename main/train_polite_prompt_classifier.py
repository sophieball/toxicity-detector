# Lint as: python3
"""Use prompt types and politeness to predict toxicity"""

import logging
logging.basicConfig(filename="bad_conver.log", filemode="w", level=logging.INFO)

from src import download_data
download_data.download_data()

import numpy as np
np.seterr(divide="ignore", invalid="ignore")

import warnings
with warnings.catch_warnings():
  warnings.simplefilter("ignore")

from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from scipy import stats
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import sys

from convokit import Corpus
from convokit import PolitenessStrategies
from convokit import download
from convokit.convokitPipeline import ConvokitPipeline
from convokit.phrasing_motifs import CensorNouns, QuestionSentences
from convokit.phrasing_motifs import PhrasingMotifs
from convokit.prompt_types import PromptTypeWrapper, PromptTypes
from convokit.text_processing import TextParser
from convokit.text_processing import TextToArcs

from sklearn import model_selection
from sklearn.feature_selection import f_classif, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, fbeta_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import conversation_struct
from src import predict_bad_conver_helpers as hp

VERBOSITY = 10000

if len(sys.argv) > 1:
  google = False
else:
  google = True

# read in data
comments = pd.read_csv("src/data/pr_body_comments.csv")
# data from MongoDB contains duplicates
comments = comments.drop_duplicates()
# construct corpus and preprocess text
speakers = conversation_struct.create_speakers(comments)
corpus = conversation_struct.prepare_corpus(comments, speakers, google)

# parse the text with spacy
parser = TextParser(verbosity=0)
corpus = parser.transform(corpus)

# prompt type
N_TYPES = 6
pt = PromptTypeWrapper(
    n_types=N_TYPES,
    use_prompt_motifs=False,
    root_only=False,
    questions_only=False,
    enforce_caps=False,
    min_support=2,
    min_df=2,
    svd__n_components=50,
    max_dist=2.,
    random_state=1000)

pt.load_models("main/pt_model_10K.files")
corpus = pt.transform(corpus)

prompt_dist_df = pd.DataFrame(
    index=corpus.vector_reprs["prompt_types__prompt_dists.6"]["keys"],
    data=corpus.vector_reprs["prompt_types__prompt_dists.6"]["vects"])
logging.info("len dist df:%d", len(prompt_dist_df))
type_ids = np.argmin(prompt_dist_df.values, axis=1)
mask = np.min(prompt_dist_df.values, axis=1) > 1.
type_ids[mask] = 6
prompt_dist_df.columns = ["km_%d_dist" % c for c in prompt_dist_df.columns]
logging.info("num prompts with ids:%d", len(prompt_dist_df))

#TYPE_NAMES = [
#    "Prompt: Casual",
#    "Prompt: Moderation",
#    "Prompt: Coordination",
#    "Prompt: Contention",
#    "Prompt: Editing",
#    "Prompt: Procedures",
#    #"Prompt: Something", "Prompt: Anotherthing"
#]
prompt_type_assignments = np.zeros(
    (len(prompt_dist_df), prompt_dist_df.shape[1] + 1))
prompt_type_assignments[np.arange(len(type_ids)), type_ids] = 1
prompt_type_assignment_df = pd.DataFrame(
    columns=np.arange(prompt_dist_df.shape[1] + 1),
    index=prompt_dist_df.index,
    data=prompt_type_assignments)
prompt_type_assignment_df = prompt_type_assignment_df[
    prompt_type_assignment_df.columns[:-1]]

prompt_type_assignment_df.columns = prompt_dist_df.columns
prompt_type_assignment_df = prompt_type_assignment_df
logging.info(prompt_type_assignment_df.shape)
logging.info(prompt_type_assignment_df.head())

# politeness strategies
# 1 for the type of politeness strategy manifested in the comment
ps = PolitenessStrategies(verbose=VERBOSITY)
corpus = ps.transform(corpus)
utterance_ids = corpus.get_utterance_ids()
rows = []
conv_info = []
has_label = False
for uid in utterance_ids:
  utt = corpus.get_utterance(uid)
  rows.append(utt.meta["politeness_strategies"])
  if "label" in utt.meta:
    has_label = True
    conv_info.append([utt.root, utt.meta["label"]])
  else:
    conv_info.append([utt.root])

if has_label:
  conv_info_df = pd.DataFrame(
      conv_info, columns=["conversation_id", "label"], index=utterance_ids)
else:
  conv_info_df = pd.DataFrame(
      conv_info, columns=["conversation_id"], index=utterance_ids)
politeness_strategies = pd.DataFrame(rows, index=utterance_ids)
# remove neg and pos lexicon counts
politeness_strategies = politeness_strategies.drop(
    columns=[
        "feature_politeness_==HASNEGATIVE==",
        "feature_politeness_==HASPOSITIVE=="
    ],
    axis=1)
logging.info(politeness_strategies.shape)
logging.info(politeness_strategies.head(10))

# Step4: (no pairing) aggregate comments measures
politeness_strategies.columns = [
    hp.clean_feature_name(col) for col in politeness_strategies.columns
]

# join prompt assignment and politeness counts
# index is the utterance id
all_features = politeness_strategies.join(prompt_type_assignment_df)
all_features = all_features.join(conv_info_df)
all_features = all_features.fillna(0)
logging.info("all_features:%d", len(all_features))
logging.info(all_features.columns)
logging.info([(k, sum(all_features[k])) for k in all_features.columns if k != "conversation_id" and k != "label"])

# aggregate features by prs
feature_table = all_features.groupby("conversation_id").sum().reset_index()
feature_table["label"] = feature_table["label"].map(lambda x: x > 0)
logging.info(feature_table.describe())
pd.DataFrame(feature_table.describe()).to_csv("feature_desc.csv")

# Prediction
logging.info(feature_table.head(5))
logging.info("feature shape:%s", str(feature_table.shape))
logging.info(set(feature_table["label"]))
logging.info(feature_table.columns)
feature_table.to_csv("feature_table.csv")

# CONVOKIT PREDICTION
feature_combos = [["politeness_strategies"], ["prompt_types"],
                  ["politeness_strategies", "prompt_types"]]
combo_names = []
accs = []
for combo in feature_combos:
  combo_names.append("+".join(combo).replace("_", " "))
  accuracy = hp.run_pipeline(feature_table, combo, True)
  accs.append(accuracy)
results_df = pd.DataFrame({"Accuracy": accs}, index=combo_names)
results_df.index.name = "Feature set"
print(results_df)
