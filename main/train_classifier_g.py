# main program for training a classifier and applies it to new data

from src import download_data
download_data.download_data()

from src import classifiers
from src import suite
import pandas as pd
from pathlib import Path
import pickle
from os.path import join
import os
path = Path(os.path.abspath("."))

# train the classifier using the result of a SQL query
def train_model(train_file, predict_file):
  s = suite.Suite()

  print("Loading data.")
  # 3 columns: _id, text, label(0/1)
  # this will later be split into train/test data
  s.set_train_set(pd.read_csv(train_file))

  # 2 columns: _id, text
  s.set_unlabeled_set(pd.read_csv(predict_file))

  # select model
  s.set_model(classifiers.svm_model)

  # list the set of parameters you want to try out
  s.set_ratios([2])
  s.add_parameter("C", [.05])
  s.add_parameter("gamma", [2])

  # select features
  s.features = ["perspective_score", "stanford_polite"]
  s.nice_features = ["perspective_score", "stanford_polite"]

  # train the model, test all combinations of hyper parameter
  model = s.self_issue_classification_all()

  # fit the model on test data
  result = s.test_issue_classifications_from_comments_all()

  # only write the id and label to file
  result = result[["_id", "prediction"]]
  result.to_csv("PATH_TO_OUTPUT_FILE", index=False)

train_model("data/training/training_data.csv",
            "data/testing/test_data.csv")
