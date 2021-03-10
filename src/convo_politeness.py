# Lint as: python3
"""Use ConvoKit to get politeness score"""

from src import download_data
download_data.download_data()

from cleantext import clean
from collections import defaultdict
from convokit import Corpus, Speaker, Utterance
from convokit import PolitenessStrategies
from convokit.text_processing import TextParser
from nltk.tokenize import sent_tokenize, word_tokenize
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
from src import text_parser

test_size = 0.2


se_file = open("src/data/SE_words_G.list")
SE_words = [se_word.strip() for se_word in se_file.readlines()]

remove_SE_words = lambda x:" ".join([w for w in word_tokenize(x) if not w in SE_words])
clean_str = lambda s: clean(remove_SE_words(text_parser.remove_inline_code(s)),
    fix_unicode=True,               # fix various unicode errors
    to_ascii=True,                  # transliterate to closest ASCII representation
    lower=True,                     # lowercase text
    no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                  # replace all URLs with a special token
    no_emails=True,                # replace all email addresses with a special token
    no_phone_numbers=True,         # replace all phone numbers with a special token
    no_numbers=False,               # replace all numbers with a special token
    no_digits=False,                # replace all digits with a special token
    no_currency_symbols=True,      # replace all currency symbols with a special token
    no_punct=False,                 # fully remove punctuation
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number="<NUMBER>",
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    lang="en"
    )

# Creating corpus from the list of utterances
def prepare_corpus(comments):
  speaker_meta = {}
  for i, row in comments.iterrows():
    speaker_meta[row["_id"]] = {"id": row["_id"]}
  corpus_speakers = {k: Speaker(id=k, meta=v) for k, v in speaker_meta.items()}

  utterance_corpus = {}
  for idx, row in comments.iterrows():
    num_sentences = len(sent_tokenize(row["text"]))
    alpha_text = clean_str(row["text"])
    #alpha_text = " ".join([x for x in row["text"].split(" ") if x.isalpha()])

    # training data
    if "thread_label" in comments.columns:
      utterance_corpus[row["_id"]] = Utterance(
          id=row["_id"],
          speaker=corpus_speakers[row["_id"]],
          text=alpha_text,
          meta={
              "id": row["_id"],
              "num_sents": num_sentences,
              "thread_id": row["thread_id"],
              "thread_label": row["thread_label"],
              "label": row["label"],
          }
      )
    elif "label" in comments.columns:
      utterance_corpus[row["_id"]] = Utterance(
          id=row["_id"],
          speaker=corpus_speakers[row["_id"]],
          text=alpha_text,
          meta={
              "id": row["_id"],
              "num_sents": num_sentences,
              "label": row["label"],
          }
      )
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
      if "thread_label" in utt.meta:
        ret["thread_label"] = utt.meta["thread_label"]
        ret["thread_id"] = utt.meta["thread_id"]
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
  return X[[
      "HASHEDGE", "2nd_person", "HASNEGATIVE", "1st_person", "2nd_person_start"
  ]]


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
  X = X.drop(columns=["_id", "label"], axis=1)
  clf = linear_model.LogisticRegression(random_state=0).fit(X, y)
  # save the model
  out = open("src/pickles/politeness.p", "wb")
  pickle.dump(clf, out)
  out.close()


# input: a pandas dataframe: _id, text
# output: a pandas dataframe: _id, text, politeness
def get_politeness_score(comments):
  corpus = transform_politeness(prepare_corpus(comments))
  # Calculate politeness score
  scores = polite_score(corpus)
  X = pick_features(scores)
  # Load the model
  clf = pickle.load(open("src/pickles/politeness.p", "rb"))
  y = clf.predict_proba(X)
  comments["politeness"] = y[:, 1]
  if "label" in scores:
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
  scores = scores.loc[(scores["thread_label"] == True) | (scores["thread_label"] == False)]
  try:
    print("here")
    scores["rounds"] = comments["rounds"]
    max_round = max(scores["rounds"].tolist())
    scores["rounds"] = scores["rounds"].map(lambda x: x / max_round)

    scores["shepherd_time"] = comments["shepherd_time"]
    min_sheph = min(scores["shepherd_time"].tolist())
    scores["shepherd_time"] = scores["shepherd_time"].map(lambda x: x / max_sheph)
  except:
    pass
  # Google
  try:
    scores["review_time"] = comments["review_time"]
    min_review = min(scores["review_time"].tolist())
    scores["review_time"] = scores["review_time"].map(lambda x: x / max_review)
  except:
    pass

  scores_thread = scores.groupby("thread_id").mean().reset_index()
  scores_thread.to_csv("politeness_features.csv", index=False)
