# input: a pandas dataframe
# output: a pandas dataframe

from nltk.stem import WordNetLemmatizer
from src import convo_politeness
from src import text_cleaning
from src import text_modifier
from src import text_parser
from src import util
from src import config
import json
import logging
import multiprocessing as mp
import numpy as np
import pandas as pd
import re
import requests
import spacy
import sys
import time
wordnet_lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])


def isascii(s):
  return all(ord(c) < 128 for c in s)


# number of multiprocess
num_proc = 14

url = ("https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze" +    \
      "?key=" + config.perspective_api_key)


def get_perspective_score(text, det_lang):
  data_dict = {
      "comment": {
          "text": text
      },
      "languages": det_lang,
      "requestedAttributes": {
          "TOXICITY": {}
      }
  }

  try:
    response = requests.post(url=url, data=json.dumps(data_dict))
    response_dict = json.loads(response.content.decode("utf-8"))
    return response_dict["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
  except:
    time.sleep(10)
    response = requests.post(url=url, data=json.dumps(data_dict))
    response_dict = json.loads(response.content.decode("utf-8"))
    try:
      return response_dict["attributeScores"]["TOXICITY"]["summaryScore"][
          "value"]
    except:
      return -1


def cleanup_text(text):
  if len(text) == 0:
    return ""

  text = nlp(text.lower().strip())
  # Stem Non-Urls/non-Stop Words/Non Punctuation/symbol/numbers
  text = [token.lemma_ for token in text]
  # Remove ampersands
  text = [re.sub(r"&[^\w]+", "", i) for i in text]
  # Remove symbols
  text = [
      w.replace("#", "").replace("&", "").replace("  ", " ")
      for w in text
      if isascii(w) and not w.isdigit()
  ]
  return text


# input: pd.DataFrame (comment)
# output: dict (features)
def extract_features(total_comment_info):
  text = total_comment_info["text"]
  total_comment_info["original_text"] = text
  if not isinstance(text, (str, list, dict)) or text is None:
    text = ""

  # count ?
  num_q = text.count("?")
  # count !
  num_exclamation = text.count("!")

  uppercase = text_parser.percent_uppercase(text)
  c_length = len(text)

  num_reference = text_parser.count_reference_line(text)
  text = text_parser.remove_reference(text)

  num_url = text_parser.count_url(text)
  text = text_parser.remove_url(text)

  num_emoji = text_parser.count_emoji(text)
  text = text_parser.remove_emoji_marker(
      text)  # remove the two semi-colons on two sides of emoji
  text = text_parser.remove_newline(text)

  num_mention = text_parser.count_mention(text)
  text = text_parser.replace_mention(text)
  num_plus_one = text_parser.count_plus_one(text)
  text = text_parser.sub_PlusOne(text)

  if text == "":
    perspective_score = -1
  else:
    perspective_score = get_perspective_score(text, "en")

  # remove stop words and lemmatization
  emoticon_r = " (:\)|;\)|;\-\)|:P|:D|:p)"
  text, emoticon_n = re.subn(emoticon_r, "", text)
  total_comment_info["num_emoticons"] = emoticon_n

  text = cleanup_text(text)
  total_comment_info["num_words"] = len(text)
  text = " ".join(text)

  total_comment_info["num_q_marks"] = num_q
  total_comment_info["num_e_marks"] = num_exclamation
  total_comment_info["percent_uppercase"] = uppercase
  total_comment_info["num_reference"] = num_reference
  total_comment_info["num_url"] = num_url
  total_comment_info["num_emoji"] = num_emoji
  total_comment_info["num_mention"] = num_mention
  total_comment_info["num_plus_one"] = num_plus_one
  total_comment_info["perspective_score"] = perspective_score
  total_comment_info["text"] = text

  return total_comment_info


# input: pd.DataFrame
# output: pd.DataFrame
def create_features(comments_df, training):
  # remove invalide toxicity scores or empty comments
  comments_df = comments_df.dropna()

  # get politeness scores for all comments
  comments_df = convo_politeness.get_politeness_score(
      comments_df)

  # remove comments longer than 300 characters (perspective limit)
  comments_df = util.remove_large_comments(comments_df)

  # convert it to a list of dictionaries
  comments = comments_df.T.to_dict().values()

  # iterate through my collection to preprocess
  pool = mp.Pool(processes=num_proc)
  features = pool.map(extract_features, comments)
  pool.close()
  features_df = pd.DataFrame(features)
  features_df = features_df.loc[features_df["perspective_score"] >= 0]
  logging.info("Total number of {} data: {}.".format(training,
                                                     len(features_df)))
  if training == "training":
    logging.info(
        "Some descriptive statistics of {} data label == 1 perspective scores:\n{}"
        .format(
            training, features_df.loc[features_df["label"] == 1,
                                      "perspective_score"].describe()))
    logging.info(
        "Some descriptive statistics of {} data label == 0 perspective scores:\n{}"
        .format(
            training, features_df.loc[features_df["label"] == 0,
                                      "perspective_score"].describe()))
    logging.info(
        "Some descriptive statistics of {} data label == 1 politeness scores:\n{}"
        .format(
            training, features_df.loc[features_df["label"] == 1,
                                      "politeness"].describe()))
    logging.info(
        "Some descriptive statistics of {} data label == 0 politeness scores:\n{}"
        .format(
            training, features_df.loc[features_df["label"] == 0,
                                      "politeness"].describe()))

  else:
    logging.info(
        "Some descriptive statistics of {} data's perspective scores:\n{}"
        .format(training, features_df["perspective_score"].describe()))
    logging.info(
        "Some descriptive statistics of {} data's politeness scores:\n{}"
        .format(training, features_df["politeness"].describe()))

  features_df.to_csv(training + "_data_label_cleaned.csv", index=False)
  return features_df
