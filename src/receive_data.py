# Lint as: python3
"""
Receive data from standard input in CSV format
"""
import io
import logging
import pandas as pd
import sys

def receive_data():
  logging.info("Reading data.")
  data = pd.read_csv(io.StringIO(sys.stdin.read()), sep=",")
  # rename id column -> CMU MongoDB uses _id as unique comment id
  data = data.rename(columns={"id": "_id"})
  # convert _id to str, because convokit requires id to be str
  data["_id"] = data["_id"].apply(str)
  # convert T/F to 1/0 because other parts of the code use 1/0
  data["label"] = data["label"].map({True: 1, False: 0})
  data["training"] = data["training"].map({True: 1, False: 0})

  training = data.loc[data["training"] == 1]
  unlabeled = data.loc[data["training"] == 0].drop(["label"], axis = 1)
  return [training, unlabeled]
