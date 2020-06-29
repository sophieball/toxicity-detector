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
import pandas as pd
import receive_data
import spacy
import text_cleaning

nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tokenizer = nltk.RegexpTokenizer(r"\w+")


# compare ngram in toxic and non-toxic comments
def word_freq(corpus):
  # fighting words
  fw = fightingWords.FightingWords(ngram_range=(1, 5))
  toxic_comments = lambda utt: utt.meta["label"] == 1.0
  non_toxic_comments = lambda utt: utt.meta["label"] == 0.0
  fw.fit(
      corpus=corpus, class1_func=toxic_comments, class2_func=non_toxic_comments)
  summary = fw.summarize(corpus)
  summary = summary.loc[abs(summary["z-score"]) >= 1.645].reset_index()
  # class1 has positive z-scores
  class1 = summary.loc[summary["class"] == "class1"].sort_values(
      by="z-score", ascending=False)
  # class2 has negative z-scores
  class2 = summary.loc[summary["class"] == "class2"].sort_values(by="z-score")
  out = open("fighting_words_freq.csv", "w")
  out.write("label==1,z-score,label==0,z-score\n")
  for i in range(40):
    out.write(class1.iloc[i]["ngram"] + "," + str(class1.iloc[i]["z-score"]) +
              "," + class2.iloc[i]["ngram"] + "," +
              str(class2.iloc[i]["z-score"]) + "\n")
  out.close()
  logging.info("fighting words lists are stored in `fighting_words_freq.csv`")


def politeness_hist(corpus):
  parser = TextParser(verbosity=0)
  corpus = parser.transform(corpus)
  ps = PolitenessStrategies()
  corpus_ps = ps.transform(corpus, markers=True)
  pos_query = lambda x: x.meta["label"] == 1
  neg_query = lambda x: x.meta["label"] == 0
  positive_count = ps.summarize(corpus, pos_query)
  negative_count = ps.summarize(corpus, neg_query)
  y_lim = max(max(positive_count["Averages"]), max(
      negative_count["Averages"])) + 1
  positive_data = ps.summarize(corpus, pos_query, plot=True, y_lim=y_lim)
  negative_data = ps.summarize(corpus, neg_query, plot=True, y_lim=y_lim)

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
  logging.info(
      "politeness words lists are stored in `politeness_wrods_marked_sorted.txt`"
  )


if __name__ == "__main__":
  [comments, _] = receive_data.receive_data()
  comments = comments.dropna()
  corpus = convo_politeness.prepare_corpus(comments)
  word_freq(corpus)
  politeness_hist(corpus)
