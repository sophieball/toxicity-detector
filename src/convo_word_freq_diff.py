# Lint as: python3
"""Collect fighting words and politeness stats"""

import logging
logging.basicConfig(
    filename="fighting_words.log", filemode="w", level=logging.INFO)

from collections import defaultdict, Counter
from convokit import Corpus, Speaker, Utterance
from convokit.fighting_words import fightingWords
from convokit import PolitenessStrategies
from convokit import TextParser
from convokit import download
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from pandas import DataFrame
from typing import List, Dict, Set
import convokit
import fighting_words_sq
import nltk
import numpy as np
import os
import pandas as pd
import receive_data
import sys
import text_parser
from sklearn.feature_extraction.text import CountVectorizer

from src import sep_ngram


NGRAM = 4


# load bot list
f = open("src/data/speakers_bots_full.list")
bots = [l.strip() for l in f.readlines()]
f.close()


# Creating corpus from the list of utterances
def prepare_corpus(comments):
  speaker_meta = {}
  for i, row in comments.iterrows():
    if "author" in comments.columns and row["author"] in bots:
      continue
    speaker_meta[row["_id"]] = {"id": row["_id"]}
  corpus_speakers = {k: Speaker(id=k, meta=v) for k, v in speaker_meta.items()}

  utterance_corpus = {}
  for idx, row in comments.iterrows():
    if "author" in comments.columns and row["author"] in bots:
      continue
    num_sentences = len(sent_tokenize(row["text"]))
    alpha_text = " ".join([x for x in row["text"].split(" ") if x.isalpha()])

    # training data
    if "label" in comments.columns:
      utterance_corpus[row["_id"]] = Utterance(
          id=row["_id"],
          speaker=corpus_speakers[row["_id"]],
          text=alpha_text,
          meta={
              "id": row["_id"],
              "num_sents": num_sentences,
              "label": row["label"],
              "thread_label": row["thread_label"],
              "thread_id": row["thread_id"],
          })
    elif "rounds" in comments.columns:
      utterance_corpus[row["_id"]] = Utterance(
          id=row["_id"],
          speaker=corpus_speakers[row["_id"]],
          text=alpha_text,
          meta={
              "id": row["_id"],
              "num_sents": num_sentences,
              "rounds": row["rounds"],
              "shepherd_time": row["shepherd_time"],
              "label": row["label"],
              "thread_label": row["thread_label"],
              "thread_id": row["thread_id"],
          })
    else:
      utterance_corpus[row["_id"]] = Utterance(
          id=row["_id"],
          speaker=corpus_speakers[row["_id"]],
          text=alpha_text,
          meta={
              "id": row["_id"],
              "num_sents": num_sentences
          })

  utterance_list = utterance_corpus.values()
  corpus = Corpus(utterances=utterance_list)
  return corpus

# compare ngram in toxic and non-toxic comments
def word_freq(corpus):
  # fighting words
  # extract text
  toxic_comments_fn = lambda utt: utt.meta["thread_label"] == 1.0
  non_toxic_comments_fn = lambda utt: utt.meta["thread_label"] == 0.0

  toxic_comments, non_toxic_comments = [], []
  for uid in corpus.get_utterance_ids():
    obj = corpus.get_utterance(uid)
    if toxic_comments_fn(obj):
      toxic_comments.append(obj)
    elif non_toxic_comments_fn(obj):
      non_toxic_comments.append(obj)
  if len(toxic_comments) == 0:
      raise ValueError("class1_func returned 0 valid corpus components.")
  if len(non_toxic_comments) == 0:
      raise ValueError("class2_func returned 0 valid corpus components.")

  # find words
  fw = fighting_words_sq.FightingWords(ngram_range=(1,NGRAM))
  fw.fit(corpus, class1_func=toxic_comments_fn,
               class2_func=non_toxic_comments_fn,)
  df = fw.summarize(corpus, plot=True, class1_name='pushback code review comments',
                class2_name='non-pushback code review comments')


  summary = fw.get_word_counts()
  summary = summary.sort_values(by="z-score", ascending=False)
  summary = summary.round(3)
  out = open("fighting_words_freq.csv", "w")
  summary.to_csv("fighting_words_freq.csv", index=False)
  sep_ngram.sep_ngram(summary.reset_index(), "fighting_words_sorted.csv", 20)
  print(
      "raw output are stored in the bazel binary's runfiles folder with the name `fighting_words_freq.csv`.\n"
  )
  print(
      "sorted by ngram version is stored in the bazel binary's runfiles folder with the name `fighting_words_sorted.csv`.\n"
  )


if __name__ == "__main__":
  what_data = "G"
  if len(sys.argv) > 1: # OSS data, Sophie passes an arg
    what_data = sys.argv[1]
  [comments, _] = receive_data.receive_data(what_data)
  corpus = prepare_corpus(comments)
  word_freq(corpus)
