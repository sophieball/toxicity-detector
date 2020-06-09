# input: a pandas dataframe
# output: pickle

import time
start = time.time()
#from classifiers import *
from get_data import *
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
#from stanford_polite import *
from convo_politeness import *
from functools import partial
import TextParser
import text_cleaning
import text_modifier
from util import *
import argparse
import config
import json
import logging
import multiprocessing as mp
import pandas as pd
import pickle
import requests
import sys
nlp = spacy.load("en_core_web_md",disable=["parser","ner"])

# load some pretrained models
#anger_classifier = pickle.load(open("pickles/anger.p","rb"))

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
    print("retry")
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
  text = [token.lemma_ for token in text]  # if token.pos_ not in \
  #["PUNCT", "SYM", "NUM"] and token.text in nlp.vocab]
  #text = [wordnet_lemmatizer.lemmatize(re.sub(r'^https?:\/\/.*[\r\n]*', '',
  #        token.text, flags=re.MULTILINE), pos="v") \
  #        for token in text \
  #        if token.pos_ not in ["PUNCT","SYM","NUM"] and token.text in nlp.vocab]
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

  #print(text)
  num_reference = TextParser.count_reference_line(text)
  #print("num_reference: %d" % num_reference)
  text = TextParser.remove_reference(text)
  #print("text 0: %s" % text)

  #text = TextParser.transform_markdown(text)    # use mistune to transform markdown into html for removal later.
  #print("text 1: %s" % text)

  #text = TextParser.remove_inline_code(text)    # used place-holder: InlineCode
  #print("text 2: %s" % text)

  num_url = TextParser.count_url(text)
  #print("num_url: %d" % num_url)
  text = TextParser.remove_url(text)
  #print("text 4: %s" % text)

  num_emoji = TextParser.count_emoji(text)
  #print("num_emoji: %d" % num_emoji)
  text = TextParser.remove_emoji_marker(
      text)  # remove the two semi-colons on two sides of emoji
  #print("text 5: %s" % text)
  text = TextParser.remove_newline(text)
  #print("text 6: %s" % text)

  num_mention = TextParser.count_mention(text)
  #print("num_mention: %d" % num_mention)
  text = TextParser.replace_mention(text)
  #print("text 7: %s" % text)
  # sub all "+1" to "plus one"
  num_plus_one = TextParser.count_plus_one(text)
  #print("num_plus_one: %d" % num_plus_one)
  text = TextParser.sub_PlusOne(text)
  #print("text 8: %s" % text)

  text = text_cleaning.remove_html(text, True)

  # Not very reliable
  #text = text_cleaning.remove_non_english(text)

  # infer the language of the comment
  # det_lang = detect(text)
  #if TextParser.contain_non_english(text): # if the text contains non-english, we terminate early
  #    text = remove_non_english(text)
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

  #print(ret)
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
