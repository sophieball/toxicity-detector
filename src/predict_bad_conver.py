# Lint as: python3
"""pr_comments has 142 toxic"""

import logging
logging.basicConfig(filename="bad_conver.log", filemode="w", level=logging.INFO)
from src import download_data
download_data.download_data()
import numpy as np
np.seterr(divide="ignore", invalid="ignore")
import pandas as pd
from scipy import stats
import os

from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.metrics import classification_report, fbeta_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.feature_selection import f_classif, SelectPercentile

from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from convokit import download
from convokit.prompt_types import PromptTypeWrapper, PromptTypes
from convokit import PolitenessStrategies
from convokit import Corpus
from convokit.text_processing import TextParser
from convokit.phrasing_motifs import PhrasingMotifs
from convokit.phrasing_motifs import CensorNouns, QuestionSentences
from convokit.convokitPipeline import ConvokitPipeline
from convokit.text_processing import TextToArcs

import warnings
with warnings.catch_warnings():
  warnings.simplefilter("ignore")

import pickle
import matplotlib.pyplot as plt

from src import conversation_struct
from src import predict_bad_conver_helpers as hp

VERBOSITY = 10000
N_TYPES = 6

# read in data
comments = pd.read_csv("src/data/pr_comments.csv")
body = pd.read_csv("src/data/pr_body.csv")
comments = pd.concat([comments, body])
comments = comments.sort_values(by=["_id", "created_at"])
# data from MongoDB contains duplicates
comments = comments.drop_duplicates()

temp = comments.loc[comments["_id"] == "openssl_____openssl_____12089"]
logging.info(temp[["id", "reply_to", "created_at"]])
# construct corpus and preprocess text
speakers = conversation_struct.create_speakers(comments)
corpus = conversation_struct.prepare_corpus(comments, speakers)

# get full corpus (10K PRs)
full_comments = pd.read_csv(
    "src/data/comments_from_random_sample_10000_prs.csv")
print("get full corpus")
full_speakers = conversation_struct.create_speakers(full_comments)
full_corpus = conversation_struct.prepare_corpus(full_comments, full_speakers)
print(len(full_corpus.get_conversation_ids()))
print("get full corpus")

# parse the text with spacy
parser = TextParser(verbosity=VERBOSITY)
corpus = parser.transform(corpus)
full_corpus = parser.transform(full_corpus)
full_corpus.dump("10K_PRs.corpus", base_path="./")

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
pt.fit(full_corpus)
pt.dump_models("pt_model_10K.files")
for i in range(N_TYPES):
  print(i)
  print(pt.display_type(i, corpus=full_corpus, k=15))
  print("\n\n")

pt.transform(corpus)
logging.info(pt.summarize(corpus, k=25))
