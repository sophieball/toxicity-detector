from src import download_data
download_data.download_data()
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
import numpy as np
import pandas as pd
import re
import spacy
import time

nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
ps = PorterStemmer()
words = set(nltk.corpus.words.words())


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


def count_vector(text):
  """ Create count vector for text """

  count_vect = CountVectorizer(
      analyzer="word", token_pattern=r"\w{1,}", ngram_range=(1, 1))
  count_vect.fit(text)

  return count_vect


def tf_idf(text):
  """ Create TF IDF for text """

  tfidf_vect = TfidfVectorizer(
      analyzer="word",
      token_pattern=r"\w{1,}",
      ngram_range=(1, 1),
      max_features=5000)
  tfidf_vect.fit(text)

  return tfidf_vect


def add_counts(counter, text_data, name="count_"):
  """ Add some counter (TFIDF/Count Vector) as a feature to dataframe """

  counts = counter.transform(text_data)

  # Create the column names
  num_words = counts.shape[1]
  column_names = [name + str(i) for i in range(num_words)]

  # Add in the count matrices
  counts = pd.DataFrame(counts.toarray(), columns=column_names)
  return counts


def get_counts(counter, df):
  """ Counter to list """

  counts = counter.transform(df["total_text"].tolist())
  return counts


def add_word2vec(df):
  """ Add word2vec to a dataframe """

  word_list = [nlp(i) for i in df["text"]]

  vector_representation = []
  for i in range(len(word_list)):
    vector_representation.append(list(word_list[i].vector))

  column_names = [
      "word2vec_" + str(i) for i in range(len(vector_representation[0]))
  ]
  return pd.concat([
      df.reset_index(drop=True),
      pd.DataFrame(vector_representation,
                   columns=column_names).reset_index(drop=True)
  ],
                   axis=1)


def add_linguistic_scores_suite(s):
  """Add differnet linguistics scores, such as whether it's doing name calling"""

  datasets = [s.all_train_data]

  if type(s.test_data) != type(None):
    datasets.append(s.test_data)

  for data in datasets:
    data["insult_product"] = data["text"].map(
        lambda x: score_comment(x, insult_product))
    data["reaction_to_toxicity"] = data["text"].map(
        lambda x: score_comment(x, reaction_to_toxicity))
    data["name_calling"] = data["text"].map(
        lambda x: score_comment(x, score_name_calling))
    data["toxicity"] = data["text"].map(lambda x: score_comment(x, score_toxic))
    data["frustrated"] = data["text"].map(
        lambda x: score_comment(x, user_frustrated))

  s.features += [
      "insult_product", "reaction_to_toxicity", "name_calling", "toxicity",
      "frustrated"
  ]
  s.nice_features += [
      "insult_product", "reaction_to_toxicity", "name_calling", "toxicity",
      "frustrated"
  ]

  return s


def add_count_vector_suite(s):
  """ Add count vector to a suite """

  if type(s) != type(None):
    s.counter = count_vector(train_data["text"].tolist() +
                             test_data["text"].tolist())
  else:
    s.counter = count_vector(train_data["text"].tolist())
  num_words = len(s.counter.get_feature_names())

  s.all_train_data = add_counts(s.counter, s.all_train_data)

  if type(s) != type(None):
    s.test_data = add_counts(s.counter, s.test_data)

  s.features += append_to_str("count_", num_words)
  s.nice_features += ["count"]

  return s


# TODO: fix return and join the return df to the main df
def add_tf_idf_suite(text_data):
  """ Add TF IDF to a suite """

  start = time.time()
  logging.info("Adding TF_IDF Suite")

  if type(data) != type(None):
    tf_idf_counter = tf_idf(text_data)
  tf_idf_words = len(tf_idf_counter.get_feature_names())
  logging.info("Finished adding TF IDF, it took {} seconds".format(time.time() -
                                                                   start))
  return add_counts(tf_idf_counter, text_data, name="tf_idf_")


def add_word2vec_suite(s):
  """ Add word2vec to a suite """

  if type(s.test_data) != type(None):
    s.test_data = add_word2vec(s.test_data)
  s.all_train_data = add_word2vec(s.all_train_data)
  s.features += append_to_str("word2vec_", 300)
  s.nice_features += ["word2vec"]

  return s


def add_context_suite(s, window, aggregate=True, past=False):
  for i, row in s.all_train_data.iterrows():
    if aggregate:
      for f in s.features:
        s.all_train_data.ix[i, "{}_{}".format(f, "before")] = 0
        s.all_train_data.ix[i, "{}_{}".format(f, "after")] = 0

    surrounding_comments = util.find_surrounding_comments(
        s.all_train_data, row["_id"], window=window)
    before = surrounding_comments["before"]
    after = surrounding_comments["after"]

    for j in range(window):
      # Get the before and the after windows
      if j < len(before):
        before_df = before.iloc[[j]][s.features]
      else:
        before_df = pd.DataFrame(
            np.zeros((1, len(s.features))), columns=s.features)

      if j < len(after):
        after_df = after.iloc[[j]][s.features]
      else:
        after_df = pd.DataFrame(
            np.zeros((1, len(s.features))), columns=s.features)

      for f in s.features:

        if aggregate:
          s.all_train_data.ix[i, "{}_{}".format(f, "before")] += list(
              before_df[f])[0] / window
          s.all_train_data.ix[i, "{}_{}".format(f, "after")] += list(
              after_df[f])[0] / window
        else:
          s.all_train_data.ix[i, "{}_{}_{}".format(f, "before", j)] = list(
              before_df[f])[0]
          s.all_train_data.ix[i, "{}_{}_{}".format(f, "after", j)] = list(
              after_df[f])[0]
  new_features = []
  if True:
    if aggregate:
      for f in s.features:
        if "{}_{}".format(f, "before") not in new_features:
          new_features += ["{}_{}".format(f, "before")]
        if past:
          for f in s.features:
            if "{}_{}".format(f, "after") not in new_features:
              new_features += ["{}_{}".format(f, "after")]
    else:
      for i in range(window):
        for f in s.features:
          if "{}_{}_{}".format(f, "before", i) not in new_features:
            new_features += ["{}_{}_{}".format(f, "before", i)]
        if past:
          for f in s.features:
            if "{}_{}_{}".format(f, "after", i) not in new_features:
              new_features += ["{}_{}_{}".format(f, "after", i)]

    s.features += new_features

    if aggregate:
      if past:
        if "aggregate_context_window_past_{}".format(
            window) not in s.nice_features:
          s.nice_features += ["aggregate_context_window_past_{}".format(window)]
      else:
        if "aggregate_context_window_{}".format(window) not in s.nice_features:
          s.nice_features += ["aggregate_context_window_{}".format(window)]
    else:
      if past:
        if "context_window_past_{}".format(window) not in s.nice_features:
          s.nice_features += ["context_window_past_{}".format(window)]
      else:
        if "context_window_{}".format(window) not in s.nice_features:
          s.nice_features += ["context_window_{}".format(window)]

  return s
