# input: a pandas dataframe
# output: a pandas dataframe

import re
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from convo_politeness import get_politeness_score
from functools import partial
import text_parser
import text_cleaning
from text_modifier import percent_uppercase, is_ascii
from util import remove_large_comments
import config
import json
import multiprocessing as mp
import pandas as pd
import pickle
import requests
import sys
import spacy
nlp = spacy.load("en_core_web_md",disable=["parser","ner"])

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
      return response_dict["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
    except:
      return -1

def cleanup_text(text):
  text = nlp(text.lower().strip())
  # Stem Non-Urls/non-Stop Words/Non Punctuation/symbol/numbers
  text = [token.lemma_ for token in text]  
  # Remove ampersands
  text = [re.sub(r"&[^\w]+", "", i) for i in text]
  # Lower case
  text = [w for w in text if w.lower() in text]
  # Remove symbols
  text = [
      w.replace("#", "").replace("&", "").replace("  ", " ")
      for w in text
      if is_ascii(w)
  ]
  return " ".join(text)

# input: pd.DataFrame (comment)
# output: dict (features)
def extract_features(total_comment_info):
  text = total_comment_info["text"]
  total_comment_info["original_text"] = text
  if not isinstance(text, (str, list, dict)) or text is None:
    text = ""

  uppercase = percent_uppercase(text)
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

  text = text_cleaning.remove_html(text, True)

  if text == "":
    perspective_score = -1
  else:
    perspective_score = get_perspective_score(text, "en")#det_lang

  # remove stop words and lemmatization
  text = cleanup_text(text)

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
def create_features(comments_df):
  # remove invalide toxicity scores or empty comments
  comments_df = comments_df.dropna()

  # get politeness scores for all comments
  all_stanford_polite = get_politeness_score(comments_df)
  comments_df = comments_df.join(all_stanford_polite.set_index("_id"), on="_id")

  # remove comments longer than 300 characters (perspective limit)
  comments_df = remove_large_comments(comments_df)

  # convert it to a list of dictionaries
  comments = comments_df.T.to_dict().values()

  # iterate through my collection to preprocess
  pool = mp.Pool(processes=10)
  features = pool.map(extract_features, comments)
  pool.close()
  features_df = pd.DataFrame(features)
  features_df = features_df.loc[features_df["perspective_score"] >= 0]
  print("Total number of training data: {}.".format(len(comments_df)))

  #features_df.to_csv("training_data_label_cleaned.csv", index=False)
  return features_df
