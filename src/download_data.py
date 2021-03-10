# Lint as: python3
"""check if necessary nltk data and spacy models are downloaded
"""

import nltk
import spacy


def download_data():
  try:
    nltk.data.find("tokenizers/punkt")
  except LookupError:
    nltk.download("punkt", quiet=True)

  try:
    nltk.data.find("corpora/words")
  except LookupError:
    nltk.download("words", quiet=True)

  try:
    nltk.data.find("corpora/wordnet")
  except LookupError:
    nltk.download("wordnet", quiet=True)

  try:
    nltk.data.find("corpora/stopwords")
  except LookupError:
    nltk.download("stopwords", quiet=True)

  try:
    spacy.load("en_core_web_sm")
  except LookupError:
    nltk.download("en_core_web_sm", quiet=True)
