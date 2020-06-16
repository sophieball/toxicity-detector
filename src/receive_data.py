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
  data = pd.read_csv(io.StringIO(sys.stdin.read()), sep=",").reset_index()
  data = data.rename(columns={"id": "_id"})
  data["label"] = data["label"].map({True: 1, False: 0})
  data["training"] = data["training"].map({True: 1, False: 0})

  training = data.loc[data["training"] == 1]
  testing = data.loc[data["training"] == 0]
  return [training, testing]
