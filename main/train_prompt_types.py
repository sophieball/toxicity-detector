# Lint as: python3
"""use politeness and prompt types to classify toxicity"""

from src import receive_data

import logging
logging.basicConfig(
    filename="bad_conver.log",
    filemode="w",
    format="%(message)s",
    level=logging.DEBUG)
from src import download_data
download_data.download_data()
import numpy as np
np.seterr(divide="ignore", invalid="ignore")
import pandas as pd
from scipy import stats
import os
import sys


class StreamToLogger(object):
  """Fake file-like stream object that redirects writes to a logger instance.
  """

  def __init__(self, logger, log_level=logging.INFO):
    self.logger = logger
    self.log_level = log_level
    self.linebuf = ""

  def write(self, buf):
    for line in buf.rstrip().splitlines():
      self.logger.log(self.log_level, line.rstrip())

  def flush(self):
    pass


stdout_logger = logging.getLogger("STDOUT")
sl = StreamToLogger(stdout_logger, logging.INFO)
sys.stdout = sl

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
from convokit.prompt_types import PromptTypeWrapper
from convokit import PolitenessStrategies
from convokit import Corpus
from convokit.text_processing import TextParser
from convokit.convokitPipeline import ConvokitPipeline
from convokit.text_processing import TextToArcs

import warnings
with warnings.catch_warnings():
  warnings.simplefilter("ignore")

import pickle

from src import conversation_struct
from src import predict_bad_conver_helpers as hp

VERBOSITY = 10000
N_TYPES = 6


def prepare_corpus(comments, f_name, google):
  # construct corpus and preprocess text
  speakers = conversation_struct.create_speakers(comments)
  corpus = conversation_struct.prepare_corpus(comments, speakers, google)
  # parse the text with spacy
  parser = TextParser(verbosity=VERBOSITY)
  corpus = parser.transform(corpus)
  corpus.dump(f_name, base_path="./")
  return corpus


def train_prompt(corpus, f_name):
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
  pt.fit(corpus)
  pt.dump_models(f_name)
  for i in range(N_TYPES):
    logging.info(i)
    logging.info(pt.display_type(i, corpus=corpus, k=15))
    logging.info("\n\n")


if __name__ == "__main__":
  if len(sys.argv) > 1:
    if sys.argv[1] == "google":
      google = True
  else:
    google = False

  comments = receive_data.receive_single_data()
  corpus = prepare_corpus(comments, "10K_PRs.corpus", True)
  train_prompt(corpus, "pt_model_10K.files")
