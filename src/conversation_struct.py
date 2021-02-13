# Lint as: python3
"""Use ConvoKit to construct Conversation"""

from collections import defaultdict
from convokit import Corpus, Speaker, Utterance
from convokit import PolitenessStrategies
from convokit.text_processing import TextParser
from datetime import datetime
from nltk.tokenize import sent_tokenize
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from src import receive_data
import logging
import markdown
import numpy as np
import pandas as pd
import pickle
import re
import string


def isascii(s):
  return all(ord(c) < 128 for c in s)


convert = lambda x:datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ")
seconds = lambda x: timedelta(days=0,
                    seconds=x.seconds, 
                    microseconds=x.microseconds).total_seconds()/60 if x.days < 0 \
                    else x.total_seconds()/60 


# https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate
translator = str.maketrans("", "", string.punctuation)

test_size = 0.2

# load bot list
f = open("src/data/speakers_bots_full.list")
bots = [l.strip() for l in f.readlines()]
f.close()


# Creating Speakers from the list of utterances
def create_speakers(comments):
  # create speakers
  speaker_meta = {}
  count = 0
  for i, row in comments.iterrows():
    login = row["author"]
    if login in bots:
      continue
    if login not in speaker_meta:
      speaker_meta[login] = {
          "id": str(count),
          "login": login,
          "associations": set(),
          "num_comments": 1
      }
      count += 1
    else:
      speaker_meta[login]["num_comments"] += 1
    try:
      # add tuple (owner/repo, association)
      speaker_meta[login]["associations"].add(
          ("/".join(row["thread_id"].split("/")[:-1]),
           row["author_association"]))
    except:
      speaker_meta[login]["associations"].add(
          (row["thread_id"], row["author_association"]))

  corpus_speakers = {k: Speaker(id=k, meta=v) for k, v in speaker_meta.items()}

  # I can sort speakers by the number of comments and verify if the uses with
  # top most comments are bots
  speaker_out = open("speakers.list", "w")
  for s in speaker_meta:
    speaker_out.write(
        str(speaker_meta[s]["num_comments"]) + "," + str(s) + "\n")
  speaker_out.close()
  return corpus_speakers


class PullRequest:

  def __init__(self, root_id, first_comment_id):
    self.root_id = str(root_id)
    self.comment_count = 0
    # first_comment_id is the root comment's id
    self.first_comment_id = first_comment_id
    # {"reply_to_1": latest_id_1, "reply_to_2": latest_id_2}:
    self.sub_conversation_id = {root_id: self.first_comment_id}
    self.first_sentences = {}
    self.authors = {}  # author_id: id
    self.comment_text = {}  # id: text
    self.prev_id = root_id
    self.rounds = 0
    self.shepherd_time = 0

  # only needed for GH
  def find_reply_to(self, row):
    reply_to = row["reply_to"].replace(".0", "")
    # if there's quote, use that to map to the reply to
    # find the first sentence of the outer layer quote
    text = row["text"]
    quotes = [
        x for x in text.split("\n")
        if len(x) > 2 and x[:2] == "> " and x[2] != ">"
    ]
    if len(quotes) > 0:
      quote = quotes[0][2:].strip()
      for k in self.comment_text:
        if quote in self.comment_text[k]:
          reply_to = self.root_id + "/" + k
          self.sub_conversation_id[k] = str(row["thread_id"])
          return reply_to
    # @
    if "@" in text:
      # find " @xxx ", strip space before and after, remove @
      match_login = r"@[a-z|A-Z|0-9][a-z|A-Z|0-9|\-]*[a-z|A-Z|0-9]?"
      mention = re.match(match_login, text)
      if mention is not None:
        mention_login = mention.group(0).strip()[1:]
        if mention_login in self.authors:
          reply_to = self.root_id + "/" + self.authors[mention_login]
          self.sub_conversation_id[self.authors[mention_login]] = str(
              row["thread_id"])
          return reply_to

    if reply_to in self.sub_conversation_id:
      # there's already a thread
      reply_to = self.root_id + "/" + self.sub_conversation_id[reply_to]
    else:
      # new thread
      reply_to = self.root_id + "/" + str(
          self.sub_conversation_id[self.root_id])
      self.sub_conversation_id[self.root_id] = str(row["thread_id"])
    return reply_to

  # set author
  # increment comment count
  # set comment_text
  # update sub_conversation_id[reply_to] and a new entry of current id
  def add_comment(self, row, google):
    utt_id = self.root_id + "/" + str(row["thread_id"])
    if type(row["text"]) == str:
      self.comment_text[str(row["thread_id"])] = row["text"]
    else:
      self.authors[row["author"]] = str(row["thread_id"])
      self.comment_count += 1
      reply_to = self.root_id
      self.prev_id = utt_id
      # update the end of root' thread to be this comment
      self.sub_conversation_id[self.root_id] = str(row["thread_id"])
      # each comment also starts its own thread
      self.sub_conversation_id[str(row["thread_id"])] = str(row["thread_id"])
      return reply_to, utt_id

    if row["reply_to"] == "ROOT":  # initial PR description
      current_reply_to = None
      reply_to = None
      utt_id = self.root_id  # root comment doesn't need id
      self.sub_conversation_id[self.root_id] = str(row["thread_id"])
    elif row["reply_to"] == "NONE":
      # discussion comments, on GH usually not a code review
      if self.comment_count == 1:
        reply_to = self.root_id
      else:
        if google:  # Google's comments don't have quotes or @s
          reply_to = self.prev_id
        else:
          reply_to = self.find_reply_to(row)
      self.sub_conversation_id[self.root_id] = str(row["thread_id"])
      self.sub_conversation_id[str(row["thread_id"])] = str(row["thread_id"])
    else:  # numbers
      # on GH: code comments, a subthread
      # on G: all reply_to are nums,
      #   replying to first_comment_id -> new subthread
      if not google:
        if self.comment_count == 1:
          # conversation structure doesn't store root comment id
          # also in our later analysis, we ignore this root comment
          reply_to = self.root_id
        else:
          reply_to = self.find_reply_to(row)
      else:
        if row["reply_to"] == self.first_comment_id:
          # parallel subthreads
          reply_to = self.root_id
        else:
          # in a subthread
          reply_to = self.root_id + "/" + row["reply_to"]

      # these are used for GH: non-code-review comments have replyto as NONE,
      # also, code comments all have reply_to point to the root of the
      # subthread, so I use the dict to record what's the latest comment id in that thread
      self.sub_conversation_id[reply_to] = str(row["thread_id"])
      self.sub_conversation_id[row["reply_to"].replace(".0", "")] = str(
          row["thread_id"])

    self.prev_id = utt_id
    self.authors[row["author"]] = str(row["thread_id"])
    self.comment_count += 1
    return reply_to, utt_id

  def get_comment_count(self):
    return self.comment_count

  def get_root_id(self):
    return self.root_id


# input: a string
# output: an array of word tokens
def preprocess_text(cur_text):
  # remove block quote
  cur_text, _ = re.subn(r">*.+\n", "", cur_text)

  # remove new lines
  cur_text = cur_text.replace("\n", " ")
  # remove code
  (alpha_text, _) = re.subn(r"```[\s|\S]+?```", "CODE", cur_text)
  num_sentences = len(sent_tokenize(alpha_text))
  # remove punctuation - this will mess up urls
  alpha_text = alpha_text.translate(translator)
  # keep words with only letters
  alpha_only = [x for x in alpha_text.split() if isascii(x) and not x.isdigit()]
  return num_sentences, alpha_only



# input: pd.DataFrame, convokit.Speakers
# output: convokit.Corpus
def prepare_corpus(comments, corpus_speakers, google):
  comments["text"] = comments["text"].replace(np.nan, "-")
  comments = comments.sort_values(by=["thread_id", "created_at"])
  comments["reply_to"] = comments["reply_to"].map(lambda x: str(x))
  utterance_corpus = {}
  # keep track of the root of each code review
  # _id is owner_repo_prid, id is the numeric id of the comment
  prev_repo = comments.iloc[0]["thread_id"]
  pr = PullRequest(prev_repo, str(prev_repo))

  conversation_label = {}
  conversation_thread_label = {}
  for idx, row in comments.iterrows():
    try:
      [owner, repo, CR_id] = row["thread_id"].split("/")
    except:
      owner = ""
      repo = ""
      CR_id = row["thread_id"]
    # update conversation root if we are entering a new code review
    if row["thread_id"] != prev_repo:
      pr = PullRequest(row["thread_id"], str(row["thread_id"]))
    prev_repo = row["thread_id"]

    # ignore bots
    if row["author"] in bots:
      continue

    # group comments by their reply_to
    reply_to, utt_id = pr.add_comment(row, google)
    if type(row["text"]) == str:
      num_sentences, alpha_text = preprocess_text(row["text"])
      num_words = len(alpha_text)
      alpha_text = " ".join(alpha_text)
    else:
      alpha_text = ""
      num_words = 0
      num_sentences = 0

    if num_sentences > 0:
      sent_len = num_words / num_sentences
    else:
      sent_len = 0

    meta = {
        "owner": owner,
        "repo": repo,
        "code_review_id": CR_id,
        "pos_in_conversation": pr.get_comment_count(),
        "num_sents": num_sentences,
        "num_words": num_words,
        "sent_len": sent_len,
        "original_text": row["text"],
    }

    # for training data
    if "label" in comments.columns:
      meta["label"] = row["label"]
      meta["thread_label"] = row["thread_label"]
      conversation_label[pr.get_root_id()] = row["label"]
      conversation_thread_label[pr.get_root_id()] = row["thread_label"]

    utterance_corpus[utt_id] = Utterance(
        id=utt_id,
        speaker=corpus_speakers[row["author"]],
        text=alpha_text,
        root=pr.get_root_id(),
        reply_to=reply_to,
        timestamp=row["created_at"],
        meta=meta)

  # Create corpus from list of utterances
  utterance_list = utterance_corpus.values()
  corpus = Corpus(utterances=utterance_list)

  # Update conversation labels
  for convo in corpus.iter_conversations():
    convo_id = convo.get_id()
    if convo_id in conversation_label:
      convo.meta["label"] = conversation_label[convo_id]
      convo.meta["thread_label"] = conversation_thread_label[convo_id]
      cur_conv_utts = convo.get_utterance_ids()
      rounds = len(cur_conv_utts)
      convo.meta["rounds"] = rounds
      first_comment = convo.get_utterance(cur_conv_utts[0])
      last_comment = convo.get_utterance(cur_conv_utts[-1])
      shepherd_time = convert(last_comment.timestamp) \
                    - convert(first_comment.timestamp)
      convo.meta["shepherd_time"] = seconds(shepherd_time)


  # print out the conversation structure of the first conversation
  first_conv = corpus.get_conversation(corpus.get_conversation_ids()[0])
  logging.info("check integrity: %s"%(first_conv.check_integrity()))
  logging.info("reply chain of the first conversation:")
  logging.info(first_conv.get_utterance(first_conv.get_utterance_ids()[0]).reply_to)
  logging.info(first_conv.print_conversation_structure())

  return corpus


if __name__ == "__main__":
  comments = pd.read_csv("src/data/pr_body_comments.csv")
  speakers = create_speakers(comments)
  prepare_corpus(comments, speakers)
