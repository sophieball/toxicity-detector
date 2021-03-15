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

from src import conversation_struct
from src import predict_bad_conver_helpers as hp

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
      text)  # emove the two semi-colons on two sides of emoji
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


def get_prompt_types(comments):
# read in data
  comments_10K = pd.read_csv("src/data/random_sample_10000_prs_body_comments.csv")
  print(len(comments_10K))
  # data from MongoDB contains duplicates
  comments = comments.drop_duplicates()
  # construct corpus and preprocess text
  speakers = conversation_struct.create_speakers(comments)
  corpus = conversation_struct.prepare_corpus(comments, speakers, google)

  speakers_10K = conversation_struct.create_speakers(comments_10K)
  corpus_10K = conversation_struct.prepare_corpus(comments_10K, speakers_10K, google)
  # get avg word count, sent len
  total_words = 0
  total_sents = 0
  total_sent_lens = 0
  total_utt = 0
  for utt_id in corpus_10K.get_utterance_ids():
    utt = corpus_10K.get_utterance(utt_id)
    total_utt += 1
    total_words += utt.meta["num_words"]
    total_sents += utt.meta["num_sents"]
    total_sent_lens += utt.meta["sent_len"]
  logging.info("Avg words per utt: {}".format(total_words/total_utt))
  logging.info("Avg sents per utt: {}".format(total_sents/total_utt))
  logging.info("Avg sent lens per utt: {}".format(total_sent_lens/total_utt))

  # parse the text with spacy
  parser = TextParser(verbosity=0)
  corpus = parser.transform(corpus)
  corpus_10K = parser.transform(corpus_10K)

  # prompt type
  N_TYPES = 6
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

  pt.fit(corpus_10K)
  corpus = pt.transform(corpus)

  prompt_dist_df = corpus.get_vectors(name='prompt_types__prompt_dists.6',
                                           as_dataframe=True)
  logging.info("len dist df:%d", len(prompt_dist_df))
  type_ids = np.argmin(prompt_dist_df.values, axis=1)
  mask = np.min(prompt_dist_df.values, axis=1) > 1.
  type_ids[mask] = 6
  prompt_dist_df.columns = ["km_%d_dist" % c for c in range(len(prompt_dist_df.columns))]
  logging.info("num prompts with ids:%d", len(prompt_dist_df))

  prompt_type_assignments = np.zeros(
      (len(prompt_dist_df), prompt_dist_df.shape[1] + 1))
  prompt_type_assignments[np.arange(len(type_ids)), type_ids] = 1
  prompt_type_assignment_df = pd.DataFrame(
      columns=np.arange(prompt_dist_df.shape[1] + 1),
      index=prompt_dist_df.index,
      data=prompt_type_assignments)
  prompt_type_assignment_df = prompt_type_assignment_df[
      prompt_type_assignment_df.columns[:-1]]
  
  prompt_type_assignment_df.columns = prompt_dist_df.columns
  return prompt_type_assignment_df.reset_index()
  

# input: pd.DataFrame
# output: pd.DataFrame
def create_features(comments_df, training):
  # remove invalide toxicity scores or empty comments
  comments_df["text"] = comments_df["text"].replace(np.nan, "-")
  comments_df["text"] = comments_df["text"].map(text_parser.remove_reference)
  comments_df["text"] = comments_df["text"].map(text_parser.remove_inline_code)
  comments_df["text"] = comments_df["text"].map( \
              lambda x: "-" if (len(x.strip()) == 0) else x)

  # get politeness scores for all comments
  comments_df = convo_politeness.get_politeness_score(
      comments_df)
  
  #prompt_types = get_prompt_types(comments_df)
  #comments_df = comments_df.join(prompt_types)


  # remove comments longer than 300 characters (perspective limit)
  #comments_df = comments_df.loc[comments_df["length"] <= 300]
  #comments_df = util.remove_large_comments(comments_df)

  ## get sentimoji
  # it's not working yet...
  #comments_df = sentimoji_classify.sentimoji(
  #                  comments_df, 
  #                  "src/model",
  #                  "SEntiMoji-G",
  #                  64
  #              )

  #logging.info(comments_df.columns())
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
        "percent_uppercase", "num_reference", "num_url", "num_emoji",
        "num_mention", "num_plus_one", "perspective_score", "identity_attack"]
  for feature in features:
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
    #logging.info(
    #    "Some descriptive statistics of {} data label == 1 politeness scores:\n{}"
    #    .format(
    #        training, features_df.loc[features_df["label"] == 1,
    #                                  "politeness"].describe()))
    #logging.info(
    #    "Some descriptive statistics of {} data label == 0 politeness scores:\n{}"
    #    .format(
    #        training, features_df.loc[features_df["label"] == 0,
    #                                  "politeness"].describe()))
    logging.info(
        "Some descriptive statistics of {} data label == 1 perspective scores:\n{}"
        .format(
            training, features_df.loc[features_df["label"] == 1,
                                      "HASHEDGE"].describe()))
    logging.info(
        "Some descriptive statistics of {} data label == 0 perspective scores:\n{}"
        .format(
            training, features_df.loc[features_df["label"] == 0,
                                      "HASHEDGE"].describe()))
    logging.info(
        "Some descriptive statistics of {} data label == 1 perspective scores:\n{}"
        .format(
            training, features_df.loc[features_df["label"] == 1,#
                                      "HASNEGATIVE"].describe()))
    logging.info(
        "Some descriptive statistics of {} data label == 0 perspective scores:\n{}"
        .format(
            training, features_df.loc[features_df["label"] == 0,
                                      "HASNEGATIVE"].describe()))
    logging.info(
        "Some descriptive statistics of {} data label == 1 perspective scores:\n{}"
        .format(
            training, features_df.loc[features_df["label"] == 1,
                                      "2nd_person"].describe()))
    logging.info(
        "Some descriptive statistics of {} data label == 0 perspective scores:\n{}"
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
