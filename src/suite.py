from collections import Counter
from collections import defaultdict
from copy import copy, deepcopy
from gensim.models.keyedvectors import KeyedVectors
from multiprocessing import Pool
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from sklearn import model_selection
from sklearn.metrics import classification_report, fbeta_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from src import classifiers
from src import convo_politeness
from src import create_features
from src import text_modifier
from src import util
from wordfreq import word_frequency
import itertools
import logging
import numpy as np
import operator
import pandas as pd
import pickle
import re
import textblob
import time
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, "politeness3")
#import politeness3.model

num_subproc = 20
model = 0


def score(lexicon_dataframe, text):
  """Need to do stemming later"""

  all_specific = lexicon_dataframe["specific"].unique()

  text = word_tokenize(text)
  text = [i.lower() for i in text]

  score_dict = {}
  for category in all_specific:
    score_dict[category] = 0
    category_words = set(lexicon_dataframe[lexicon_dataframe["specific"] ==\
                                           category]["word"].tolist())
    score_dict[category] = len(category_words.intersection(text))

  return score_dict


def rescore(row, features, tf_idf_counter):
  new_sentence = row["text"]
  new_features_dict = {}
  for f in features:
    new_features_dict[f] = row[f]

  if "length" in features:
    new_features_dict["length"] = len(new_sentence)

  if "perspective_score" in features:
    persp_score = create_features.get_perspective_score(new_sentence, "en")
    new_features_dict["perspective_score"] = persp_score

  if "politeness" in features:
    polite_df = convo_politeness.get_politeness_score(
        pd.DataFrame([{
            "_id": "1",
            "text": new_sentence
        }]))
    new_features_dict["politeness"] = polite_df.iloc[0]["politeness"]

  if "word2vec_0" in features:
    # Calcualte word2vec
    df = pd.DataFrame([{"text": new_sentence}])
    df = text_modifier.add_word2vec(df).iloc[0]
    word2vec_values = [df["word2vec_{}".format(i)] for i in range(300)]

    for i in range(300):
      new_features_dict["word2vec_{}".format(i)] = word2vec_values[i]

  if "LIWC_anger" in features:
    lexicon_df = pd.read_csv("src/data/lexicons.txt")
    s = score(lexicon_df, new_sentence)
    new_features_dict["LIWC_anger"] = s["LIWC_anger"]

  if "negative_lexicon" in features:
    lexicon_df = pd.read_csv("src/data/lexicons.txt")
    s = score(lexicon_df, new_sentence)
    new_features_dict["negative_lexicon"] = s["negative_lexicon"]

  if "nltk_score" in features:
    sid = SentimentIntensityAnalyzer()
    nltk_score = sid.polarity_scores(new_sentence)["compound"]
    new_features_dict["nltk_score"] = nltk_score

  if "polarity" in features or "subjectivity" in features:
    textblob_scores = textblob.TextBlob(new_sentence)
    new_features_dict["polarity"] = textblob_scores.polarity
    new_features_dict["subjectivity"] = textblob_scores.subjectivity

  if "tf_idf_0" in features:
    df = pd.DataFrame([{"text": new_sentence}])
    df = add_counts(tf_idf_counter, df, name="tf_idf_").iloc[0]

    for f in features:
      if "tf_idf_" in f:
        new_features_dict[f] = df[f]

  new_features = []
  for f in features:
    new_features.append(new_features_dict[f])

  return new_features


# postprocessing (usually only done for toxic comments)
# returns list of clean text variants
def clean_text(text):
  result = []
  words = text.split(" ")
  words = [a.strip(",.!?:; ") for a in words]

  words = list(set(words))
  words = [
      word for word in words
      if not word.isalpha() or word.lower() in different_words
  ]

  for word in set(words):
    # Maybe unkify?
    result += [
        re.sub(r"[^a-zA-Z0-9]" + re.escape(word.lower()) + r"[^a-zA-Z0-9]",
               " potato ", " " + text.lower() + " ").strip()
    ]

  tokenizer = RegexpTokenizer(r"\w+")
  all_words = tokenizer.tokenize(text)
  # Try removing all unknown words
  for word in set(all_words):
    if word.lower() not in counter and word_frequency(
        word.lower(), "en") == 0 and len(word) > 2:
      text = text.replace(word, "")

  result += [text]
  return result


# input: comment, trained model, features used, ?
# output: 0 if the comment was labeled to be toxic NOT due to SE words (it IS toxic)
#         1 if the comment was labeled to be toxic due to SE words (it shouldn't
#         be toxic)
def remove_SE_comment(row, model, features, tf_idf_counter):
  text = row["original_text"]
  t = time.time()
  words = text.split(" ")
  words = [a.strip(",.!?:; ") for a in words]

  words = list(set(words))
  # different_words: words with a different distribution in SE context than in
  # normal EN context
  words = [
      word for word in words
      if not word.isalpha() or word.lower() in different_words
  ]

  # the comment was labeld to be toxic not because it contains SE words
  if len(words) == 0:
    return 0

  for word in set(words):
    # NV: Maybe unkify?
    # replace those SE words with "potato"
    new_sentence = re.sub(
        r"[^a-zA-Z0-9]" + re.escape(word.lower()) + r"[^a-zA-Z0-9]", " potato ",
        text.lower())
    # re-compute features
    row["text"] = new_sentence
    new_features = rescore(row, features, tf_idf_counter)

    # after removing SE words, the model labels it as non-toxic
    if model.predict([new_features])[0] == 0:
      # it was labeled to be toxic because of SE words
      return 1

  tokenizer = RegexpTokenizer(r"\w+")
  all_words = tokenizer.tokenize(text)
  # Try removing all unknown words
  for word in set(all_words):
    if word.lower() not in counter and word_frequency(
        word.lower(), "en") == 0 and len(word) > 2:
      text = text.replace(word, "")

  if model.predict([new_features])[0] == 0:
    return 1

  # after removing SE words and unknown words, still the classifier labels it
  # toxic
  return 0


class Suite:

  def __init__(self):
    global different_words
    global counter

    self.features = []
    self.nice_features = []
    self.parameter_names = []
    self.hyper_parameters_lists = []
    self.param_grid = {}
    self.last_time = time.time()
    self.tf_idf_counter = 0
    self.use_filters = True
    self.counter = pickle.load(open("src/pickles/github_words.p", "rb"))
    counter = self.counter
    self.our_words = dict([
        (i, word_frequency(i, "en") * 10**9) for i in self.counter
    ])
    self.different_words = util.log_odds(
        defaultdict(int, self.counter), defaultdict(int, self.our_words))
    different_words = self.different_words
    self.anger_classifier = pickle.load(open("src/pickles/anger.p", "rb"))
    self.all_words = pickle.load(open("src/pickles/all_words.p", "rb"))
    self.m = sum(self.counter.values())
    self.all_false = {word: False for word in self.all_words}

    start_time = time.time()
    self.alpha = 0.1

    self.all_train_data = None
    self.test_data = None
    self.train_data = None
    self.model_function = None
    self.model = None

  def set_model_function(self, model_function):
    self.model_function = model_function

  def set_trained_model(self, trained_model):
    self.model = trained_model

  def add_parameter(self, name, l):
    self.parameter_names.append(name)
    self.hyper_parameters_lists.append(l)

  def set_ratios(self, ratios):
    self.ratio = ratios

  def set_train_set(self, train_collection):
    self.train_collection = train_collection
    self.all_train_data = create_features.create_features(
        train_collection, "training")
    logging.info(
        "Prepared training dataset, it took {} seconds".format(time.time() - \
                                                               self.last_time))
    self.last_time = time.time()

  def set_unlabeled_set(self, test_collection):
    self.test_collection = test_collection
    self.test_data = create_features.create_features(test_collection,
                                                     "unlabeled")
    logging.info(
        "Prepared unlabeled dataset, it took {} seconds".format(time.time() - \
                                                              self.last_time))

    self.last_time = time.time()

  def convert(self, test_sentence):
    ret = copy(self.all_false)

    for word in word_tokenize(test_sentence.lower()):
      ret[word] = True

    return ret

  def remove_I(self, test_issues):
    test_issues["self_angry"] = 0

    test_issues.loc[test_issues.prediction == 1, "self_angry"] = test_issues[
        test_issues["prediction"] == 1]["original_text"].map(
            lambda x: self.anger_classifier.classify(self.convert(x)))

    test_issues.loc[test_issues.self_angry == "self", "prediction"] = 0

    return test_issues

  def remove_SE(self, data):

    features = self.features
    tf_idf_counter = self.tf_idf_counter
    model = self.model

    p = Pool(num_subproc)
    data["is_SE"] = 0
    new_pred = p.starmap(remove_SE_comment, [
        (x, self.model, features, tf_idf_counter)
        for x in data.loc[data["prediction"] == 1].T.to_dict().values()
    ])  #original_text])
    data.loc[data.prediction == 1, "is_SE"] = new_pred
    data.loc[data.is_SE == 1, "prediction"] = 0

    return data

  def classify_test(self):
    return classifiers.classify(self.model, self.train_data, self.test_data,
                                self.features)

  def classify_test_statistics(self):
    return classify_statistics(self.model, self.train_data, self.test_data,
                               self.features)

  def set_parameters(self, grid):
    self.param_grid = grid

  # training the model on comments
  def self_issue_classification_all(self, estimator_name):
    # n-fold nested cross validation
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
    num_trials = 5
    n_splits = 10
    best_model = None
    best_score = 0
    if estimator_name == "svm":
      estimator = SVC()
    for i in range(num_trials):
      # find the best paramter combination
      train_data = self.all_train_data
      y_train = train_data["label"]
      X_train = train_data[self.features]
      model = GridSearchCV(
          estimator=estimator,
          param_grid=self.param_grid,
          scoring="f1_weighted",
          n_jobs=14,  # parallel
          cv=StratifiedKFold(n_splits=n_splits, shuffle=True),
          verbose=0)
      model.fit(X_train, y_train)

      # nested cross validation with paramter optimization
      nested_score = model_selection.cross_val_score(
          model,
          X_train,
          y_train,
          cv=StratifiedKFold(n_splits=n_splits, shuffle=True))
      nested_scores = nested_score.mean()
      if nested_scores > best_score:
        best_score = nested_scores
        best_model = model

    # Find the optimal parameters
    logging.info("Trying all combinations of hyper parameters.")
    logging.info("Scores with {}-fold cross validation".format(n_splits))
    logging.info("Best parameter: {}.".format(best_model.best_estimator_))
    logging.info("Best score: {}.".format(best_model.best_score_))
    self.model = best_model

    logging.info("Removing angry words towards oneself and SE words.")
    y_train = self.all_train_data["label"]
    X_train = self.all_train_data[self.features]
    self.all_train_data["raw_prediction"] = model.predict(X_train)
    self.all_train_data["prediction"] = self.all_train_data["raw_prediction"]

    self.all_train_data = self.remove_I(self.all_train_data)
    self.all_train_data = self.remove_SE(self.all_train_data)
    logging.info("Crossvalidation score after adjustment is\n{}".format(
        classification_report(self.all_train_data["label"].tolist(),
                              self.all_train_data["prediction"].tolist())))
    return model

  # applying the model to the test data
  def test_issue_classifications_from_comments_all(self, matched_pairs=False):
    X_test = self.test_data[self.features]

    self.test_data["raw_prediction"] = self.model.predict(X_test)
    self.test_data["prediction"] = self.test_data["raw_prediction"]
    self.test_data = self.remove_I(self.test_data)
    self.test_data = self.remove_SE(self.test_data)
    return self.test_data
