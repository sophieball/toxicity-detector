# Lint as: python3
"""Collect fighting words and politeness stats"""

import logging
logging.basicConfig(
    filename="fighting_words.log", filemode="w", level=logging.INFO)

import download_data
download_data.download_data()

from cleantext import clean
from collections import defaultdict, Counter
from convokit import Corpus, Speaker, Utterance
from convokit.fighting_words import fightingWords
from convokit import PolitenessStrategies
from convokit import TextParser
from convokit import download
from nltk import tokenize
from nltk.stem import WordNetLemmatizer
from pandas import DataFrame
from typing import List, Dict, Set
import convokit
import convo_politeness
import fighting_words_sq
import nltk
import numpy as np
import os
import pandas as pd
import receive_data
import spacy
import text_parser
from sklearn.feature_extraction.text import CountVectorizer

from src import sep_ngram
from src import plot_politeness


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
lemmatizer = WordNetLemmatizer()
tokenizer = nltk.RegexpTokenizer(r"\w+")
NGRAM = 4


# compare ngram in toxic and non-toxic comments
def word_freq(corpus):
  # fighting words
  # extract text
  toxic_comments_fn = lambda utt: utt.meta["label"] == 1.0
  non_toxic_comments_fn = lambda utt: utt.meta["label"] == 0.0

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
  total_sents = 0

  for utt in utts:
    if len(utt.text) == 0:
      continue
    for k, v in utt.meta["politeness_markers"].items():
      name = k[21:len(k) - 2]
      counts[name] += len(v)
      total_sents += utt.meta["num_sents"]
      """if name in [

          "Please_start", "Direct_start", "Direct_question", "1st_person_start",
          "2nd_person_start"
      ]:
        counts[name] += len(v) / utt.meta["num_sents"]
      else:
        counts[k[21:len(k) - 2]] += len(v) / len(utt.text)
      """
  scores = {k: v / total_sents for k, v in counts.items()}
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
  count_df = pd.DataFrame(
      {
          "label=1": list(positive_count.values()),
          "label=0": list(negative_count.values())
      },
      index=positive_count.keys())
  count_df.to_csv("polite_strategies.csv")

  # plot the histogram
  plot_politeness.save_plot(count_df, "politeness.pdf", 0.2)

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
  logging.info("Log-odds ratio plot is saved in the bazel binary's runfiles folder with the name `log-odds_ratio.PNG`\n")
  logging.info(
      "politeness words counts are stored in the same folder with the name `polite_strategies_label_x.csv`, x = {{0, 1}}\n")
  logging.info(
      "politeness words lists are stored in the same folder with the name `politeness_words_marked_sorted.txt`\n"
  )
  logging.info(
      "politeness words plots are stored in the same folder with the name `labelx_politeness.pdf`, x = {{0, 1}}\n"
  )


if __name__ == "__main__":
  [comments, _] = receive_data.receive_data()
  comments["text"] = comments["text"].replace(np.nan, "-")
  corpus = convo_politeness.prepare_corpus(comments)
  word_freq(corpus)
  politeness_hist(corpus)
