# Lint as: python3
"""
try to find the words that appear more frequently in one group than the other
"""

import convokit
import pandas as pd


# Creating corpus from the list of utterances
def prepare_corpus(comments):
  speaker_meta = {}
  for i, row in comments.iterrows():
    speaker_meta[row["_id"]] = {"id": row["_id"]}
  corpus_speakers = {k: convokit.Speaker(id=k, meta=v) for k, v in speaker_meta.items()}

  utterance_corpus = {}
  for idx, row in comments.iterrows():
    utterance_corpus[row["_id"]] = convokit.Utterance(
        id=row["_id"],
        speaker=corpus_speakers[row["_id"]],
        text=row["text"],
        meta={"id": row["_id"],
              "toxic": row["toxic"]})

  utterance_list = utterance_corpus.values()
  corpus = convokit.Corpus(utterances=utterance_list)
  return corpus

# compare ngram in toxic and non-toxic comments
def word_freq(comments):
  corpus = prepare_corpus(comments)
  print(corpus.random_utterance())
  print(next(corpus.iter_utterances()))
  # load fighting words
  fw = convokit.FightingWords()

  toxic_comments = lambda utt: utt.meta["toxic"] == "y"
  non_toxic_comments = lambda utt: utt.meta["toxic"] == "n"
  fw.fit(corpus=corpus, class1_func = toxic_comments, class2_func = non_toxic_comments)
  fw.summarize(corpus).to_csv("ngram_zscore.csv")

comments = pd.read_csv("data/training_data.csv")

word_freq(comments)
