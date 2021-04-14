# Lint as: python3
"""check if necessary nltk data and spacy models are downloaded
"""

import nltk


def download_data():
  try:
    nltk.data.find("tokenizers/punkt")
  except LookupError:
    nltk.download("punkt")

  try:
    nltk.data.find("corpora/words")
  except LookupError:
    nltk.download("words")

  try:
    nltk.data.find("corpora/wordnet")
  except LookupError:
    nltk.download("wordnet")

  try:
    nltk.data.find("corpora/stopwords")
  except LookupError:
    nltk.download("stopwords")
