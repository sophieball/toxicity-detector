# Lint as: python3
"""Use ConvoKit to get politeness score"""

from convokit import Corpus, Speaker, Utterance
from collections import defaultdict
from convokit.text_processing import TextParser
from convokit import PolitenessStrategies
import pandas as pd


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

  num_features = len(
      corpus.get_utterance(
          corpus.get_utterance_ids()[0]).meta["politeness_strategies"])
  scores = []
  for x in corpus.get_utterance_ids():
    ret = {"_id": x}
    utt = corpus.get_utterance(x)
    total = 0
    for ((k, v), (k1, v2)) in zip(utt.meta["politeness_strategies"].items(),
                                  utt.meta["politeness_markers"].items()):
      cur_name = k.split("==")[1]
      if v != 0:
        ret[cur_name] = len(v2)
      else:
        ret[cur_name] = 0
      # currently I'm taking average of strategies that are not associated with
      # higher likelihood of being toxic
      if cur_name != "HASNEGATIVE" and cur_name != "2nd_person" \
          and cur_name != "2nd_person_start":
        total += v
    ret["stanford_polite"] = total / num_features
    scores.append(ret)
  return pd.DataFrame(scores)
