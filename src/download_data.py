# Lint as: python3
"""
check if necessary nltk data are downloaded:
"""

import nltk

def download_data():
  try:
    nltk.data.find("tokenizers/punkt")
  except LookupError:
    print("not found")

  try:
    nltk.data.find("corpora/words")
  except LookupError:
    nltk.download("words")
