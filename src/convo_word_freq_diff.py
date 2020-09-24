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
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pandas import DataFrame
from typing import List, Dict, Set
import convokit
import convo_politeness
import fighting_words_py3
import nltk
import numpy as np
import os
import pandas as pd
import receive_data
import spacy
import text_cleaning
from sklearn.feature_extraction.text import CountVectorizer

from src import sep_ngram
from src import plot_politeness

clean_str = lambda s: clean(s,
    fix_unicode=True,               # fix various unicode errors
    to_ascii=True,                  # transliterate to closest ASCII representation
    lower=True,                     # lowercase text
    no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                  # replace all URLs with a special token
    no_emails=True,                # replace all email addresses with a special token
    no_phone_numbers=True,         # replace all phone numbers with a special token
    no_numbers=False,               # replace all numbers with a special token
    no_digits=False,                # replace all digits with a special token
    no_currency_symbols=True,      # replace all currency symbols with a special token
    no_punct=False,                 # fully remove punctuation
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_number="<PHONE>",
    replace_with_number="<NUMBER>",
    replace_with_digit="0",
    replace_with_currency_symbol="<CUR>",
    lang="en"
    )

nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tokenizer = nltk.RegexpTokenizer(r"\w+")
NGRAM = 6


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
  toxic_comments = [clean_str(obj.text) for obj in toxic_comments]
  non_toxic_comments = [clean_str(obj.text) for obj in non_toxic_comments]

  # find words
  summary = fighting_words_py3.bayes_compare_language(toxic_comments,
                                                     non_toxic_comments,
                                                     NGRAM)
  summary["abs_z-score"] = abs(summary["z-score"])
  summary = summary.sort_values(by="abs_z-score", ascending=False)
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
  plot_politeness.save_plot(count_df, "label1_politeness.pdf", 0.2)

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
