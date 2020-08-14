# Lint as: python3
"""Collect fighting words and politeness stats"""

import logging
logging.basicConfig(
    filename="fighting_words.log", filemode="w", level=logging.INFO)

import download_data
download_data.download_data()

from collections import defaultdict, Counter
from convokit import Corpus, Speaker, Utterance
from convokit.fighting_words import fightingWords
from convokit import PolitenessStrategies
from convokit import TextParser
from convokit import download
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pandas import DataFrame
from typing import List, Dict, Set
import convokit
import convo_politeness
import nltk
import numpy as np
import os
import pandas as pd
import receive_data
import spacy
import text_cleaning

from src import sep_ngram
from src import plot_politeness

nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tokenizer = nltk.RegexpTokenizer(r"\w+")


# compare ngram in toxic and non-toxic comments
def word_freq(corpus):
  # fighting words
  fw = fightingWords.FightingWords(ngram_range=(1, 6))
  toxic_comments = lambda utt: utt.meta["label"] == 1.0
  non_toxic_comments = lambda utt: utt.meta["label"] == 0.0
  fw.fit(
      corpus=corpus, class1_func=toxic_comments, class2_func=non_toxic_comments)
  summary = fw.summarize(corpus)
  summary["abs_z-score"] = abs(summary["z-score"])
  summary = summary.sort_values(by="abs_z-score", ascending=False)
  out = open("fighting_words_freq.csv", "w")
  summary.to_csv("fighting_words_freq.csv")
  sep_ngram.sep_ngram(summary.reset_index(), "fighting_words_sorted.csv", 20)
  print(
      "raw output are stored in the bazel binary's runfiles folder with the name `fighting_words_freq.csv`.\n"
  )
  print(
      "sorted by ngram version is stored in the bazel binary's runfiles folder with the name `fighting_words_sorted.csv`.\n"
  )


def pl_summarize(corpus, selector):
  utts = list(corpus.iter_utterances(selector))
  if "politeness_markers" not in utts[0].meta:
    print(
        "Could not find politeness markers metadata. Running transform() on corpus first...",
        end="")
    self.transform(corpus, markers=True)
    print("Done.")

  counts = {
      k[21:len(k) - 2]: 0 for k in utts[0].meta["politeness_markers"].keys()
  }

  for utt in utts:
    if len(utt.text) == 0:
      continue
    for k, v in utt.meta["politeness_markers"].items():
      name = k[21:len(k) - 2]
      if name in [
          "Please_start", "Direct_start", "Direct_question", "1st_person_start",
          "2nd_person_start"
      ]:
        counts[name] += len(v) / utt.meta["num_sents"]
      else:
        counts[k[21:len(k) - 2]] += len(v) / len(utt.text)
  scores = {k: v / len(utts) for k, v in counts.items()}
  return scores


def politeness_hist(corpus):
  parser = TextParser(verbosity=0)
  corpus = parser.transform(corpus)
  ps = PolitenessStrategies()
  corpus_ps = ps.transform(corpus, markers=True)
  pos_query = lambda x: x.meta["label"] == 1
  neg_query = lambda x: x.meta["label"] == 0
  positive_count = pl_summarize(corpus, pos_query)
  negative_count = pl_summarize(corpus, neg_query)
  pos_df = pd.DataFrame(positive_count, positive_count.keys())
  neg_df = pd.DataFrame(negative_count, negative_count.keys())
  pos_df.to_csv("polite_strategies_label_1.csv")
  neg_df.to_csv("polite_strategies_label_0.csv")

  # plot the histogram
  plot_politeness.save_plot(positive_count, "label1_politeness.pdf", 0.2)
  plot_politeness.save_plot(negative_count, "label0_politeness.pdf", 0.2)

  # individual politeness strategies
  out = open("politeness_words_marked_sorted.txt", "w")

  pos_words = defaultdict(list)
  neg_words = defaultdict(list)

  for utt_id in corpus.get_utterance_ids():
    utt = corpus.get_utterance(utt_id)
    for ((k, v), (k1,
                  v_marked)) in zip(utt.meta["politeness_strategies"].items(),
                                    utt.meta["politeness_markers"].items()):
      if v != 0:
        for marked_words in v_marked:
          for marked_word in marked_words:
            if utt.meta["label"] == 1:
              # marked_word is a tuple, the first element is the word
              pos_words[k[21:len(k) - 2]].append((marked_word[0]))
            elif utt.meta["label"] == 0:
              neg_words[k[21:len(k) - 2]].append((marked_word[0]))
  out.write("LABEL == 1\n")
  for w in pos_words:
    out.write(str(w))
    c = Counter(pos_words[w])
    c = sorted(c.items(), key=lambda pair: pair[1], reverse=True)
    for each_word in c:
      out.write(str(each_word) + ",")
    out.write("\n")

  out.write("\nLABEL == 0\n")
  for w in pos_words:
    out.write(str(w))
    c = Counter(neg_words[w])
    c = sorted(c.items(), key=lambda pair: pair[1], reverse=True)
    for each_word in c:
      out.write(str(each_word) + ",")
    out.write("\n")
  out.close()
  print(
      "politeness words counts are stored in the bazel binary's runfiles folder with the name `polite_strategies_label_x.csv`, x = {{0, 1}}\n"
  )
  print(
      "politeness words lists are stored in the bazel binary's runfiles folder with the name `politeness_words_marked_sorted.txt`\n"
  )
  print(
      "politeness words plots are stored in the bazel binary's runfiles folder with the name `labelx_politeness.pdf`, x = {{0, 1}}\n"
  )


if __name__ == "__main__":
  [comments, _] = receive_data.receive_data()
  comments = comments.dropna()
  corpus = convo_politeness.prepare_corpus(comments)
  word_freq(corpus)
  politeness_hist(corpus)
