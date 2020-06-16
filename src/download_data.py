# Lint as: python3
"""
check if necessary nltk data and spacy models are downloaded
"""

import nltk
import spacy

nltk.data.path = (["src/nltk_data"])
def download_data():
  try:
    from nltk.corpus import words
  except LookupError:
    nltk.download("words")

  try:
    from nltk.corpus import wordnet
  except LookupError:
    nltk.download("wordnet")

  try:
    from nltk.corpus import stopwords 
  except LookupError:
    nltk.download("stopwords")

  try:
    spacy.load("en")
  except IOError:
    spacy.cli.download("en")

  try:
    spacy.load("en_core_web_md")
  except IOError:
    spacy.cli.download("en_core_web_md")
