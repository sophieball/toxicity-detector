# input: a pandas dataframe
# output: a pandas dataframe

from cleantext import clean
from convokit import Corpus
from convokit import PolitenessStrategies
from convokit import download
from convokit.convokitPipeline import ConvokitPipeline
from convokit.phrasing_motifs import CensorNouns, QuestionSentences
from convokit.phrasing_motifs import PhrasingMotifs
from convokit.prompt_types import PromptTypeWrapper, PromptTypes
from convokit.text_processing import TextParser
from convokit.text_processing import TextToArcs

from src import receive_data
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from src import convo_politeness
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
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

clean_str = lambda s: clean(s,
              fix_unicode=True,         # fix various unicode errors
              to_ascii=True,          # transliterate to closest ASCII representation
              lower=True,           # lowercase text
              no_line_breaks=True,       # fully strip line breaks as opposed to only normalizing them
              no_urls=True,          # replace all URLs with a special token
              no_emails=True,        # replace all email addresses with a special token
              no_phone_numbers=True,     # replace all phone numbers with a special token
              no_numbers=False,         # replace all numbers with a special token
              no_digits=False,        # replace all digits with a special token
              no_currency_symbols=True,    # replace all currency symbols with a special token
              no_punct=False,         # fully remove punctuation
              replace_with_url="<URL>",
              replace_with_email="<EMAIL>",
              replace_with_phone_number="<PHONE>",
              replace_with_number="<NUMBER>",
              replace_with_digit="0",
              replace_with_currency_symbol="<CUR>",
              lang="en"
              )

VERBOSITY = 10000

if len(sys.argv) > 1:
  google = False
else:
  google = True



def isascii(s):
  return all(ord(c) < 128 for c in s)


# number of multiprocess
num_proc = 24

if config.perspective_api_key == "TODO:FILL_THIS_IN":
  print("need perspective score API key.")
  exit()

url = ("https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze" +    \
      "?key=" + config.perspective_api_key)


def get_perspective_score(text, det_lang):
  data_dict = {
      "comment": {
          "text": text
      },
      "languages": det_lang,
      "requestedAttributes": {
          "TOXICITY": {},
          "IDENTITY_ATTACK": {}
      }
  }

  try:
    response = requests.post(url=url, data=json.dumps(data_dict))
    response_dict = json.loads(response.content.decode("utf-8"))
    toxicity = response_dict["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
    identity_attack = response_dict["attributeScores"]["IDENTITY_ATTACK"]["summaryScore"]["value"]
    return [toxicity, identity_attack]
  except:
    time.sleep(10)
    response = requests.post(url=url, data=json.dumps(data_dict))
    response_dict = json.loads(response.content.decode("utf-8"))
    try:
      toxicity = response_dict["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
      identity_attack = response_dict["attributeScores"]["IDENTITY_ATTACK"]["summaryScore"]["value"]
      return [toxicity, identity_attack]
    except:
      return [-1, -1]


# input: pd.DataFrame (comment)
# output: dict (features)
def extract_features(total_comment_info):
  text = total_comment_info["text"]
  total_comment_info["original_text"] = total_comment_info["text"]
  if not isinstance(text, (str, list, dict)) or text is None:
    text = "-"

  uppercase = text_parser.percent_uppercase(text)

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

  text = clean_str(total_comment_info["text"])
  c_length = len(word_tokenize(text))
  total_comment_info["length"] = c_length

  if text == "":
    perspective_score = [-1, -1]
  else:
    perspective_score = get_perspective_score(text, "en")

  total_comment_info["percent_uppercase"] = uppercase
  total_comment_info["num_reference"] = num_reference
  total_comment_info["num_url"] = num_url
  total_comment_info["num_emoji"] = num_emoji
  total_comment_info["num_mention"] = num_mention
  total_comment_info["num_plus_one"] = num_plus_one
  total_comment_info["perspective_score"] = perspective_score[0]
  total_comment_info["identity_attack"] = perspective_score[1]
  total_comment_info["text"] = text

  return total_comment_info


# input: pd.DataFrame
# output: pd.DataFrame
def create_features(comments_df, training, G):
  # remove invalide toxicity scores or empty comments
  comments_df["text"] = comments_df["text"].replace(np.nan, "-")
  comments_df["text"] = comments_df["text"].map(text_parser.remove_reference)
  comments_df["text"] = comments_df["text"].map(text_parser.remove_inline_code)
  comments_df["text"] = comments_df["text"].map( \
              lambda x: "-" if (len(x.strip()) == 0) else x)

  # get politeness scores for all comments
  comments_df = convo_politeness.get_politeness_score(
      comments_df)

  # convert it to a list of dictionaries
  comments = comments_df.T.to_dict().values()

  # iterate through my collection to preprocess
  pool = mp.Pool(processes=num_proc)
  features = pool.map(extract_features, comments)
  pool.close()
  features_df = pd.DataFrame(features)
  features_df = features_df.loc[features_df["perspective_score"] >= 0]
  logging.info("length {}".format(len(features_df)))
  features_df = features_df.replace(np.nan, 0)

  features = [
        "Please", "Please_start", "HASHEDGE", "Indirect_(btw)", "Hedges",
        "Factuality", "Deference", "Gratitude", "Apologizing", "1st_person_pl.",
        "1st_person", "1st_person_start", "2nd_person", "2nd_person_start",
        "Indirect_(greeting)", "Direct_question", "Direct_start", "HASPOSITIVE",
        "HASNEGATIVE", "SUBJUNCTIVE", "INDICATIVE", "num_words", "length",
        "rounds", "shepherd_time", "review_time",
        "percent_uppercase", "num_reference", "num_url", "num_emoji",
        "num_mention", "num_plus_one", "perspective_score", "identity_attack"]
  for feature in features:
    if feature not in features_df.columns:
      continue
    max_f = max(features_df[feature].tolist())
    if max_f != 0:
      features_df[feature] = features_df[feature].map(lambda x: x/max_f)


  logging.info("Total number of {} data: {}.".format(training,
                                                     len(features_df)))
  try:
    logging.info("Total number of {} positive data: {}.".format(training,
                                                     sum(features_df["label"])))
  except:
    pass

  comments_df["label"].to_csv("pos_labels.csv")
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
        "Some descriptive statistics of {} data label == 1 HASHEDGE scores:\n{}"
        .format(
            training, features_df.loc[features_df["label"] == 1,
                                      "HASHEDGE"].describe()))
    logging.info(
        "Some descriptive statistics of {} data label == 0 HASHEDGE scores:\n{}"
        .format(
            training, features_df.loc[features_df["label"] == 0,
                                      "HASHEDGE"].describe()))
    logging.info(
        "Some descriptive statistics of {} data label == 1 HASNEGATIVE scores:\n{}"
        .format(
            training, features_df.loc[features_df["label"] == 1,#
                                      "HASNEGATIVE"].describe()))
    logging.info(
        "Some descriptive statistics of {} data label == 0 HASNEGATIVE scores:\n{}"
        .format(
            training, features_df.loc[features_df["label"] == 0,
                                      "HASNEGATIVE"].describe()))
    logging.info(
        "Some descriptive statistics of {} data label == 1 2nd person scores:\n{}"
        .format(
            training, features_df.loc[features_df["label"] == 1,
                                      "2nd_person"].describe()))
    logging.info(
        "Some descriptive statistics of {} data label == 0 2nd person scores:\n{}"
        .format(
            training, features_df.loc[features_df["label"] == 0,
                                      "2nd_person"].describe()))

  else:
    logging.info(
        "Some descriptive statistics of {} data's perspective scores:\n{}"
        .format(training, features_df["perspective_score"].describe()))
    logging.info(
        "Some descriptive statistics of {} data's politeness scores:\n{}"
        .format(training, features_df["politeness"].describe()))

  features_df.to_csv(training + "_data_label_cleaned.csv", index=False)
  return features_df
