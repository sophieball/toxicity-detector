# Lint as: python3
"""
check if necessary nltk data and spacy models are downloaded
"""

import nltk
import spacy

def download_data():
  try:
    nltk.data.find("tokenizers/punkt")
  except LookupError:
    print("not found")

  try:
    nltk.data.find("corpora/words")
  except LookupError:
    nltk.download("words")

  try:
    spacy.load("en")
  except IOError:
    spacy.cli.download("en")

  try:
    spacy.load("en_core_web_md")
  except IOError:
    spacy.cli.download("en_core_web_md")
