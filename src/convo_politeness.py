# Lint as: python3
"""
Use ConvoKit to get politeness score
"""

from spacy.lang.en import English
from convokit import Corpus, Speaker, Utterance
from collections import defaultdict
from convokit.text_processing import TextParser
from convokit import PolitenessStrategies
import pandas as pd

# predict politeness (don't need it for now?)
#import random
#from sklearn import svm
#from scipy.sparse import csr_matrix
#from convokit import Classifier
#from sklearn.metrics import classification_report

# Creating corpus from the list of utterances
def prepare_corpus(comments):
  speaker_meta = {}
  for i, row in comments.iterrows():
    speaker_meta[row["_id"]] = {"id": row["_id"]}
  corpus_speakers = {k: Speaker(id=k, meta=v) for k, v in speaker_meta.items()}

  utterance_corpus = {}
  for idx, row in comments.iterrows():
    utterance_corpus[row["_id"]] = Utterance(
        id=row["_id"],
        speaker=corpus_speakers[row["_id"]],
        text=row["text"],
        meta={"id": row["_id"]})

  utterance_list = utterance_corpus.values()
  corpus = Corpus(utterances=utterance_list)
  return corpus

# input: a pandas dataframe: _id, text
# output: a pandas dataframe: _id, text, stanford_politeness
def get_politeness_score(comments):
  corpus = prepare_corpus(comments)
  # Processing utterance texts
  parser = TextParser(verbosity=0)
  corpus = parser.transform(corpus)
  ps = PolitenessStrategies()
  corpus = ps.transform(corpus, markers=True)
  #query = lambda x : x.meta["politeness_strategies"]["feature_politeness_==HASPOSITIVE=="] == 1
  num_features = len(
      corpus.get_utterance(
          corpus.get_utterance_ids()[0]).meta["politeness_strategies"])
  scores = [{
      "_id":
          x,
      "stanford_polite":
          sum(corpus.get_utterance(x).meta["politeness_strategies"].values()) / num_features
  } for x in corpus.get_utterance_ids()]
  return pd.DataFrame(scores)

"""
# Bazel doesn't work with convokit yet
# So when running with Bazel, I comment out the code above to test the rest of
# the code
def get_politeness_score(comments):
  comments["stanford_polite"] = 1
  return comments#[["_id", "stanford_polite"]]
  #return pd.DataFrame({"_id": comments["_id"], "stanford_polite": 1}
"""
