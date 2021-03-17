# Lint as: python3
"""Use ConvoKit to get politeness score"""

from src import download_data
download_data.download_data()

from sklearn.metrics import classification_report, fbeta_score, roc_auc_score, roc_curve
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC
from collections import defaultdict
from convokit import Corpus, Speaker, Utterance
from convokit import PolitenessStrategies
from convokit.text_processing import TextParser
from nltk.tokenize import sent_tokenize
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from src import receive_data
import logging
import numpy as np
import pandas as pd
import pickle
import sklearn

test_size = 0.2


# load bot list
f = open("src/data/speakers_bots_full.list")
bots = [l.strip() for l in f.readlines()]
f.close()


# Creating corpus from the list of utterances
def prepare_corpus(comments):
  speaker_meta = {}
  for i, row in comments.iterrows():
    if "author" in comments.columns and row["author"] in bots:
      continue
    speaker_meta[row["_id"]] = {"id": row["_id"]}
  corpus_speakers = {k: Speaker(id=k, meta=v) for k, v in speaker_meta.items()}

  utterance_corpus = {}
  for idx, row in comments.iterrows():
    if "author" in comments.columns and row["author"] in bots:
      continue
    num_sentences = len(sent_tokenize(row["text"]))
    alpha_text = " ".join([x for x in row["text"].split(" ") if x.isalpha()])

    # training data
    if "label" in comments.columns:
      utterance_corpus[row["_id"]] = Utterance(
          id=row["_id"],
          speaker=corpus_speakers[row["_id"]],
          text=alpha_text,
          meta={
              "id": row["_id"],
              "num_sents": num_sentences,
              "label": row["label"],
              "thread_label": row["thread_label"],
              "thread_id": row["thread_id"],
          })
    elif "rounds" in comments.columns:
      utterance_corpus[row["_id"]] = Utterance(
          id=row["_id"],
          speaker=corpus_speakers[row["_id"]],
          text=alpha_text,
          meta={
              "id": row["_id"],
              "num_sents": num_sentences,
              "rounds": row["rounds"],
              "shepherd_time": row["shepherd_time"],
              "label": row["label"],
              "thread_label": row["thread_label"],
              "thread_id": row["thread_id"],
          })
    else:
      utterance_corpus[row["_id"]] = Utterance(
          id=row["_id"],
          speaker=corpus_speakers[row["_id"]],
          text=alpha_text,
          meta={
              "id": row["_id"],
              "num_sents": num_sentences
          })

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
# output: a pd frame; each row corresponds to politeness
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
    ret["num_words"] = len(utt.text.split(" "))
    ret["length"] = len(utt.text)
    # training data
    if "label" in utt.meta:
      ret["label"] = utt.meta["label"]
      ret["thread_label"] = utt.meta["thread_label"]
      ret["thread_id"] = utt.meta["thread_id"]
    if "is_pr" in utt.meta:
      ret["is_pr"] = utt.meta["is_pr"]
    if "rounds" in utt.meta:
      ret["rounds"] = utt.meta["rounds"]
      ret["shepherd_time"] = utt.meta["shepherd_time"]
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
  X = transform_features(X)
  return X.drop(columns=["Indirect_(btw)", "Indirect_(greeting)",
  "Direct_start", "Gratitude", "Apologizing", "Direct_start",
  "SUBJUNCTIVE", "INDICATIVE"])


# split data to do cross validation
def cross_validate(comments):
  corpus = transform_politeness(prepare_corpus(comments))
  scores = polite_score(corpus)
  y = scores["label"]
  X = pick_features(scores)
  X_train, X_test, y_train, y_test = model_selection.train_test_split(
      X, y, test_size=test_size, random_state=0)
  clf = ensemble.RandomForestClassifier(
      max_depth=2, random_state=0).fit(X_train, y_train)
  pred = clf.predict(X_test)
  logging.info("\n" + metrics.classification_report(y_test, pred))
  # save the model
  out = open("src/pickles/politeness.p", "wb")
  pickle.dump(clf, out)
  out.close()


# train a logistic regression model using all training data
def train_polite(comments):
  corpus = transform_politeness(prepare_corpus(comments))
  # get only politeness strategy markers counts
  scores = polite_score(corpus)
  y = scores["label"]
  X = pick_features(scores)
  X = X.drop(columns=["_id", "label", "thread_label"], axis=1)
  clf = LinearSVC(random_state=0, max_iter=10000)
  clf = clf.fit(X, y)
  pred = clf.predict(X)
  print(classification_report(y,
                              pred))
  # save the model
  out = open("src/pickles/politeness.p", "wb")
  print(clf.coef_)
  pickle.dump(clf, out)
  out.close()


# input: a pandas dataframe: _id, text
# output: a pandas dataframe: _id, text, politeness
def get_politeness_score(comments):
  corpus = transform_politeness(prepare_corpus(comments))
  scores = polite_score(corpus)
  if "thread_label" in scores:
    scores = scores.drop(["_id", "label", "thread_label"], axis=1)
  elif "label" in scores:
    scores = scores.drop(["_id", "label"], axis=1)
  else:
    scores = scores.drop(["_id"], axis=1)
  comments = pd.concat([comments, scores], axis=1)
  return comments


if __name__ == "__main__":
  [comments, _] = receive_data.receive_data()
  comments["text"] = comments["text"].replace(np.nan, "-")
  corpus = transform_politeness(prepare_corpus(comments))
  scores = polite_score(corpus)
  scores_thread = scores.groupby("thread_id").sum().reset_index()
  scores_thread.to_csv("politeness_features.csv", index=False)
