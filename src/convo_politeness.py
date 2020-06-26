# Lint as: python3
"""Use ConvoKit to get politeness score"""

from convokit import Corpus, Speaker, Utterance
from collections import defaultdict
from convokit.text_processing import TextParser
from convokit import PolitenessStrategies
import logging
import numpy as np
import pandas as pd
import pickle
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
import sklearn

test_size = 0.2


# Creating corpus from the list of utterances
def prepare_corpus(comments):
  speaker_meta = {}
  for i, row in comments.iterrows():
    speaker_meta[row["_id"]] = {"id": row["_id"]}
  corpus_speakers = {k: Speaker(id=k, meta=v) for k, v in speaker_meta.items()}

  utterance_corpus = {}
  for idx, row in comments.iterrows():
    # training data
    if "label" in comments.columns:
      utterance_corpus[row["_id"]] = Utterance(
          id=row["_id"],
          speaker=corpus_speakers[row["_id"]],
          text=row["text"],
          meta={
              "id": row["_id"],
              "label": row["label"]
          })
    else:
      utterance_corpus[row["_id"]] = Utterance(
          id=row["_id"],
          speaker=corpus_speakers[row["_id"]],
          text=row["text"],
          meta={"id": row["_id"]})

  utterance_list = utterance_corpus.values()
  corpus = Corpus(utterances=utterance_list)
  return corpus


def transform_politeness(corpus):
  parser = TextParser(verbosity=0)
  corpus = parser.transform(corpus)
  ps = PolitenessStrategies()
  corpus = ps.transform(corpus, markers=True)
  return corpus


# input: convo corpus (from training comments)
# output: a list of dictionaries, each of which corresponds to politeness
# strategies of an utterance
def polite_score(corpus):
  num_features = len(
      corpus.get_utterance(
          corpus.get_utterance_ids()[0]).meta["politeness_strategies"])
  scores = []
  for x in corpus.get_utterance_ids():
    ret = {"_id": x}
    utt = corpus.get_utterance(x)
    total = 0
    # get the number of words in each politeness strategy
    for ((k, v), (k1, v2)) in zip(utt.meta["politeness_strategies"].items(),
                                  utt.meta["politeness_markers"].items()):
      cur_name = k.split("==")[1]
      if v != 0:
        ret[cur_name] = len(v2)
      else:
        ret[cur_name] = 0
    # training data
    if "label" in utt.meta:
      ret["label"] = utt.meta["label"]
    scores.append(ret)
  return pd.DataFrame(scores)


# These are based on manual inspection on the variance
def transform_features(X):
  X["HASHEDGE"] = np.log(1 + X["HASHEDGE"])
  X["Please"] = X["Please"] > 0
  X["Please_start"] = X["Please_start"] > 0
  X["Indirect_(btw)"] = X["Indirect_(btw)"] > 0
  X["Factuality"] = X["Factuality"] > 0
  X["Deference"] = X["Deference"] > 0
  X["Gratitude"] = X["Gratitude"] > 0
  X["Apologizing"] = X["Apologizing"] > 0
  X["1st_person_pl."] = np.log(1 + X["1st_person_pl."])
  X["1st_person"] = np.log(1 + X["1st_person"])
  X["1st_person_start"] = np.log(1 + X["1st_person_start"])
  X["2nd_person"] = np.log(1 + X["2nd_person"])
  X["2nd_person_start"] = X["2nd_person_start"] > 0
  X["Indirect_(greeting)"] = X["Indirect_(greeting)"] > 0
  X["Direct_question"] = X["Direct_question"] > 0
  X["Direct_start"] = np.log(1 + X["Direct_start"])
  X["HASPOSITIVE"] = np.log(1 + X["HASPOSITIVE"])
  X["HASNEGATIVE"] = np.log(1 + X["HASNEGATIVE"])
  X["SUBJUNCTIVE"] = X["SUBJUNCTIVE"] > 0
  X["INDICATIVE"] = X["INDICATIVE"] > 0
  return X


# These features are picked based on a logistic regression
def pick_features(X):
  if "label" in X.columns:
    X = X.drop(columns=["_id", "label"], axis=1)
  else:
    X = X.drop(columns=["_id"], axis=1)
  X = transform_features(X)
  X = X[[
      "HASHEDGE", "2nd_person", "HASNEGATIVE", "1st_person", "2nd_person_start"
  ]]
  return X


# split data to do cross validation
def cross_validate(comments):
  corpus = prepare_corpus(comments)
  corpus = transform_politeness(corpus)
  scores = polite_score(corpus, need_stanford=False)
  y = scores["label"]
  X = pick_features(scores)
  X_train, X_test, y_train, y_test = model_selection.train_test_split(
      X, y, test_size=test_size, random_state=0)
  #clf = linear_model.LogisticRegression(random_state=0).fit(X_train, y_train)
  clf = ensemble.RandomForestClassifier(
      max_depth=2, random_state=0).fit(X_train, y_train)
  #clf = sklearn.tree.DecisionTreeClassifier().fit(X_train, y_train)
  pred = clf.predict(X_test)
  logging.info("\n" + metrics.classification_report(y_test, pred))
  # save the model
  out = open("src/pickles/politeness.p", "wb")
  pickle.dump(clf, out)
  out.close()


# train a logistic regression model using all training data
def train_polite(comments):
  corpus = prepare_corpus(comments)
  corpus = transform_politeness(corpus)
  # get only politeness strategy markers counts
  scores = polite_score(corpus, need_stanford=False)
  y = scores["label"]
  X = pick_features(scores)
  clf = linear_model.LogisticRegression(random_state=0).fit(X, y)
  # save the model
  out = open("src/pickles/politeness.p", "wb")
  pickle.dump(clf, out)
  out.close()


# input: a pandas dataframe: _id, text
# output: a pandas dataframe: _id, text, politeness
def get_politeness_score(comments):
  corpus = prepare_corpus(comments)
  # Processing utterance texts
  corpus = transform_politeness(corpus)
  # Calculate politeness score
  scores = polite_score(corpus)
  X = pick_features(scores)
  # Load the model
  clf = pickle.load(open("src/pickles/politeness.p", "rb"))
  y = clf.predict_proba(X)
  comments["politeness"] = y[:, 1]
  return comments
