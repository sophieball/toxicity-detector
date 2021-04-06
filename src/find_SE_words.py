# Lint as: python3
"""Use Convokit's log-odd-ratio to find words that are more likely to be SE words that may mess up our classifiers"""

import langdetect
import pandas as pd
import fighting_words_py3 as fighting
import re
import string
from src import receive_data

# https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate
translator = str.maketrans("", "", string.punctuation)

NGRAM = 1
reviews = receive_data.receive_single_data()


def isascii(s):
  return all(ord(c) < 128 for c in s)


# input: a string
# output: an array of word tokens
def preprocess_text(cur_text):
  if type(cur_text) != str:
    return ""
  # remove block quote
  cur_text, _ = re.subn(r">*.+\n", "", cur_text)

  # remove new lines
  cur_text = cur_text.replace("\n", " ")
  # remove code
  (alpha_text, _) = re.subn(r"```[\s|\S]+?```", "CODE", cur_text)
  # remove punctuation - this will mess up urls
  alpha_text = alpha_text.translate(translator)
  # keep words with only letters
  alpha_only = " ".join(
      [x for x in alpha_text.split() if isascii(x) and not x.isdigit()])
  try:
    comment_lang = langdetect.detect(alpha_only)
  except langdetect.lang_detect_exception.LangDetectException:
    comment_lang = ""
  if len(alpha_only.strip()) == 0 or comment_lang != "en":
    return ""
  return alpha_only


# load comments from other fields
other_comments = pd.read_csv("src/data/kaggle_toxicity_subset.csv")
num_review = len(reviews)
num_kaggle = len(other_comments)
# downsample one of the dataset to make the two sets same length
if num_kaggle > num_review:
  other_comments = other_comments.sample(num_review)
elif num_review > num_kaggle:
  reviews = reviews.sample(num_kaggle)
other_comments = [
    preprocess_text(x) for x in other_comments["comment_text"].to_list()
]
review_comments = [preprocess_text(x) for x in reviews["text"].to_list()]

# compare ngram in comments and English
results = fighting.bayes_compare_language(review_comments, other_comments, NGRAM)

results_df = pd.DataFrame(results, columns=["ngram", "z-score"])
results_df = results_df.loc[results_df["z-score"] >= 1.96]
results_df = results_df.sort_values("z-score", ascending=False)
results_df.to_csv("SE_words_G_zscores.csv", index=False)
words = list(set(results_df["ngram"].to_list()))
out = open("SE_words_G.list", "w")
out.write("\n".join(words))
