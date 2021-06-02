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
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, fbeta_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from src import classifiers
from src import convo_politeness
from src import create_features
from src import text_modifier
from src import text_parser
from wordfreq import word_frequency
import itertools
import logging
import numpy as np
import operator
import pandas as pd
import pickle
import random
import re
import textblob
import time
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, "politeness3")
#import politeness3.model

# stop words: https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
stop_words = ["ourselves", "hers", "between", "yourself", "but", "again", "there", "about",
"once", "during", "out", "very", "having", "with", "they", "own", "an", "be",
"some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself",
"other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each",
"the", "themselves", "until", "below", "are", "we", "these", "your", "his",
"through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down",
"should", "our", "their", "while", "above", "both", "up", "to", "ours", "had",
"she", "all", "no", "when", "at", "any", "before", "them", "same", "and",
"been", "have", "in", "will", "on", "does", "yourselves", "then", "that",
"because", "what", "over", "why", "so", "can", "did", "not", "now", "under",
"he", "you", "herself", "has", "just", "where", "too", "only", "myself",
"which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs",
"my", "against", "a", "by", "doing", "it", "how", "further", "was", "here",
"than"]
# flip due to the removal of SE words
FLIP = 1
DONT_FLIP = 0


def isascii(s):
  return all(ord(c) < 128 for c in s)


num_subproc = 20
model = 0

se_file = open("src/data/SE_words_G.list")
SE_words = [se_word.strip() for se_word in se_file.readlines()]

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

  if "perspective_score" in features:
    persp_score = create_features.get_perspective_score(new_sentence, "en")
    new_features_dict["perspective_score"] = persp_score[0]
    new_features_dict["identity_attack"] = persp_score[1]

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

  return new_features_dict

# postprocessing (usually only done for toxic comments)
# returns list of clean text variants
def clean_text(text):
  result = []
  words = text.split(" ")
  words = [a.strip(",.!?:; ") for a in words]

  words = list(set(words))
  words = [
      word for word in words if not isascii(word) or word.lower() in SE_words
  ]

  for word in set(words):
    # Maybe unkify?
    result += [
        re.sub(r"[^a-zA-Z0-9]" + re.escape(word.lower()) + r"[^a-zA-Z0-9]",
               " potato ", " " + text.lower() + " ").strip()
    ]

  tokenizer = RegexpTokenizer(r"\w+")
  all_words = tokenizer.tokenize(text)

  result += [text]
  return result


# input: comment, trained model, features used, ?
# output: 0 if the comment was labeled to be toxic NOT due to SE words (it IS toxic)
#         1 if the comment was labeled to be toxic due to SE words (it shouldn't
#         be toxic)
def remove_SE_comment(features_df, Google, row, model, features, max_values, tf_idf_counter):
  text = row["text"]
  t = time.time()
  words = text.split(" ")
  words = [a.strip(",.!?:; ") for a in words]

  words = list(set(words))
  # SE_words: words with a different distribution in SE context than in
  # normal EN context
  words = [
      word for word in words if not word.isalpha() or word.lower() in SE_words
  ]

  # the comment was labeld to be toxic not because it contains SE words
  if len(words) == 0:
    return 0

  for word in set(words):
    # if word is a stop word
    if word in stop_words or (not word.isalpha()):
      continue

    new_sentence = re.sub(
        r"[^a-zA-Z0-9]" + re.escape(word.lower()) + r"[^a-zA-Z0-9]", " ",
        text.lower())
    # re-compute features
    row["text"] = new_sentence
    new_features_dict = rescore(row, features, tf_idf_counter)

    new_features = {}
    for f in features:
      max_f = max_values[f]
      if max_f != 0:
        new_features[f] = new_features_dict[f]/max_f
      else:
        new_features[f] = new_features_dict[f]

    # after removing SE words, the model labels it as non-toxic
    new_features = pd.DataFrame([new_features])
    if model.predict(new_features)[0] == 0:
      # it was labeled to be toxic because of SE words
      if not Google and row["label"]:
        # print out some quotes for debugging
        logging.info("going to be flipped:{}, {}: {}".format(row["thread_id"],
                row["label"], text))
        logging.info("old values: {}".format(row))
        logging.info("new values: {}".format(new_features.iloc[0]))
        logging.info("after being flipped: word:{},  new sentence: {}".format(word, 
                new_sentence))
      return FLIP

  # after removing SE words and unknown words, still the classifier labels it
  # toxic
  return DONT_FLIP 

class Suite:

  def __init__(self):
    global counter

    self.features = []
    self.max_feature_values = {}
    self.nice_features = []
    self.parameter_names = []
    self.hyper_parameters_lists = []
    self.param_grid = {}
    self.last_time = time.time()
    self.tf_idf_counter = 0
    self.use_filters = True

    self.anger_classifier = pickle.load(open("src/pickles/anger.p", "rb"))
    self.all_words = pickle.load(open("src/pickles/all_words.p", "rb"))
    self.all_false = {word: False for word in self.all_words}

    start_time = time.time()
    self.alpha = 0.1

    self.all_train_data = None
    self.test_data = None
    self.train_data = None
    self.model_function = None
    self.model = None
    self.Google = False

  def set_G(self, G):
    self.Google = G

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
        train_collection, "training", self.Google)
    for f in self.all_train_data.columns:
      if f in ["text", "author", "author_association", "url", "html_url"]: 
        continue
      try:
        self.max_feature_values[f] = max(self.all_train_data[f].tolist())
      except:
        pass
    logging.info(
        "Prepared training dataset, it took {} seconds".format(time.time() - \
                                                               self.last_time))
    self.last_time = time.time()

  def set_unlabeled_set(self, test_collection):
    self.test_collection = test_collection
    self.test_data = create_features.create_features(test_collection,
                                                     "unlabeled", self.Google)
    logging.info(
        "Prepared unlabeled dataset, it took {} seconds".format(time.time() - \
                                                              self.last_time))

    self.last_time = time.time()

  def convert(self, test_sentence):
    ret = copy(self.all_false)

    for word in word_tokenize(str(test_sentence).lower()):
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
        (data, self.Google, x, self.model, features, self.max_feature_values, tf_idf_counter)
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
  def self_issue_classification_all(self, model_name, fid):
    # n-fold nested cross validation
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
    num_trials = 5
    n_splits = 10
    best_model = None
    best_score = 0
    if model_name == "svm":
      estimator = SVC()
    elif model_name == "rf":
      estimator = RandomForestClassifier()
    elif model_name == "lg":
      estimator = LogisticRegression()

    # split training and test
    # split thread_labels
    thread_id_label = self.all_train_data[["thread_id", "thread_label"]]
    thread_id_label = thread_id_label.drop_duplicates()
    X_train_id, X_test_id, _, _ = train_test_split(
        thread_id_label, thread_id_label["thread_label"], test_size=0.33,
        random_state=42)
    # split data into train and test
    X_train_id = X_train_id["thread_id"]
    X_test_id = X_test_id["thread_id"]
    train_data = self.all_train_data.loc[self.all_train_data["thread_id"].isin(X_train_id)]
    test_data = self.all_train_data.loc[self.all_train_data["thread_id"].isin(X_test_id)]

    X_train = train_data[self.features]
    y_train = train_data["label"]
    y_test = test_data["label"]
    X_test = test_data[self.features]

    # feature importance
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X_train, y_train)
    logging.info("Feature importance: {}\n".format(
            [(self.features[i], round(x, 3)) 
              for (i, x) in enumerate(clf.feature_importances_)]))

    for i in range(num_trials):
      # find the best paramter combination
      model = GridSearchCV(
          estimator=estimator,
          param_grid=self.param_grid,
          scoring="f1_weighted",
          n_jobs=num_subproc,  # parallel
          cv=StratifiedKFold(n_splits=n_splits, shuffle=True),
          verbose=0)
      model.fit(X_train, y_train)

      # nested cross validation with paramter optimization
      nested_score = cross_val_score(
          model,
          X_train,
          y_train,
          cv=StratifiedKFold(n_splits=n_splits, shuffle=True))
      nested_scores = nested_score.mean()
      if nested_scores > best_score:
        best_score = nested_scores
        best_model = model

    # permutation importance
    logging.info("importance on train set\n")
    r = permutation_importance(best_model, X_train, y_train,
                                n_repeats=30,
                                random_state=0)
    logging.info("For ploting feature importance")
    logging.info(X_test.columns)
    logging.info(",".join([str(round(x, 3)) for x in r.importances_mean]))
    logging.info(",".join([str(round(x, 3)) for x in r.importances_std]))

    # Find the optimal parameters
    logging.info("Trying all combinations of hyper parameters.")
    logging.info("Scores with {}-fold cross validation".format(n_splits))
    logging.info("Best parameter: {}.".format(best_model.best_estimator_))
    logging.info("Best score: {}.".format(best_model.best_score_))
    self.model = best_model

    # test
    test_data["raw_prediction"] = model.predict(X_test)
    # without removing SE words and anger words, prediction == raw_pred
    test_data["prediction"] = test_data["raw_prediction"]

    logging.info("Features: {}".format(self.features))
    logging.info("Crossvalidation score for comments before adjustment is\n{}".format(
        classification_report(test_data["label"].tolist(),
                              test_data["prediction"].tolist())))

    # adjust SE words and anger words
    logging.info("Removing angry words towards oneself and SE words.")
    if "perspective_score" in self.features:
      test_data = self.remove_I(test_data)
      test_data = self.remove_SE(test_data)
    logging.info("Crossvalidation score for comments after adjustment is\n{}".format(
        classification_report(test_data["label"].tolist(),
                              test_data["prediction"].tolist())))

    logging.info("Number of 1's in raw prediction: {}.".format(
        sum(test_data["raw_prediction"])))
    try:
      logging.info("Number of data flipped due to SE: {}.".format(
          len(test_data.loc[test_data["is_SE"] == 1])))
      logging.info("Number of data flipped due to self angry: {}.".format(
          len(test_data.loc[test_data["self_angry"] == "self"])))
    except:
      pass

    # THREAD level accuracy
    #if not self.Google:
    # if any comment is predicted as 1(toxic) then the whole thread is consider
    # toxic
    label_data = test_data[["thread_id", "thread_label"]]
    true_thread_label = label_data.groupby("thread_id").first()
    true_thread_label = true_thread_label.reset_index()
    true_thread_label = true_thread_label["thread_label"]
    test_data.to_csv("reported_pb_oss_pred.csv", index=False)

    label_data = test_data[["thread_id", "prediction"]]
    predicted_threads = label_data.groupby("thread_id")["prediction"].sum()
    predicted_threads = predicted_threads.reset_index()
    predicted_threads["thread_prediction"] = \
            predicted_threads["prediction"].map(lambda x: int(x>0))
    predicted_thread_label = predicted_threads["thread_prediction"]

    logging.info("Crossvalidation score for thread after adjustment is\n{}".format(
        classification_report(true_thread_label.tolist(),
                              predicted_thread_label.tolist())))

    """
    if not self.Google:
      logging.info("\n")
      incidentally_toxic = test_data.loc[(test_data["prediction"]==1) &
                                         (test_data["label"]==0) &
                                         (test_data["thread_label"]==1)]
      logging.info("comments in a toxic thread predicted as 1 but with label 0:{}".format(len(incidentally_toxic)))
      if len(incidentally_toxic) > 0:
        for i in range(10):
          if i > len(incidentally_toxic):
            break
          logging.info(incidentally_toxic.iloc[i])
    """
    
    model_out = open(
        "src/pickles/{}_model_{}.p".format(model_name.upper(), str(fid)),
        "wb")
    pickle.dump(self.model, model_out)

    return model

  # applying the model to the test data
  def test_issue_classifications_from_comments_all(self, matched_pairs=False):
    X_test = self.test_data[self.features]

    self.test_data["raw_prediction"] = self.model.predict(X_test)
    self.test_data["prediction"] = self.test_data["raw_prediction"]
    if "perspective_score" in self.features:
      self.test_data = self.remove_I(self.test_data)
      self.test_data = self.remove_SE(self.test_data)
    return self.test_data
