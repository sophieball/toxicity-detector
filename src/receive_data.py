# Lint as: python3
"""
Receive data from R script
"""
import io
import pandas as pd
import sys

def receive_data():
  print("Reading data.")
  data = pd.read_csv(io.StringIO(sys.stdin.read()), sep=",")
  data = data.rename(columns={"id": "_id"})
  data["label"] = data["label"].map({True: 1, False: 0})
  data["training"] = data["training"].map({True: 1, False: 0})

  training = data.loc[data["training"] == 1]
  testing = data.loc[data["training"] == 0]
  return [training, testing]
