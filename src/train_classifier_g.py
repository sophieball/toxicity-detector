# main program for training a classifier and applies it to new data

import download_data
download_data.download_data()

from suite import Suite
import pandas as pd
from classifiers import *
from pathlib import Path
from os.path import join
import os
path = Path(os.path.abspath("."))

""" connect to SQL 

"""
def train_model():
  s = Suite()

  print("Loading training data.")
  # executes a SQL query and returns a pd.DataFrame
  # 3 columns: _id, text, label(0/1)
  train_dat = pd.read_sql_query("SQL_QUERY_FOR_TRANING_DATA")
  s.set_train_set(train_dat)
  # 2 columns: _id, text
  test_dat = pd.read_sql_query("SQL_QUERY_FOR_TEST_DATA")
  s.set_test_set(test_dat)
  s.set_model(svm_model)

  # list the set of parameters you want to try out
  s.set_ratios([2])
  s.add_parameter("C", [.05])
  s.add_parameter("gamma", [2])

  s.features = ["perspective_score", "stanford_polite"]
  s.nice_features = ["perspective_score", "stanford_polite"]

  # train the model, test all combinations of hyper parameter
  s.self_issue_classification_all()
  # fit the model on test data
  result = s.test_issue_classifications_from_comments_all()

  # only write the id and label to file
  result = result[["_id", "prediction"]]
  result.to_csv("PATH_TO_OUTPUT_FILE", index=False)

train_model()
