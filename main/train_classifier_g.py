# main program for training a classifier and applies it to new data
from src import download_data
download_data.download_data()

import logging
logging.basicConfig(
    filename="main/train_classifier.log", filemode="w", level=logging.INFO)
logging.basicConfig(level=logging.INFO)
from src import receive_data
from src import classifiers
from src import suite
import pandas as pd
import pickle


# train the classifier using the result of a SQL query
def train_model(training_data, unlabeled_data):
  s = suite.Suite()

  logging.info("Loading data.")
  # 3 columns: _id, text, label(0/1)
  # this will later be split into train/test data
  s.set_train_set(training_data)

  # 2 columns: _id, text
  s.set_unlabeled_set(unlabeled_data)

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
  result = result[["_id", "perspective_score", "stanford_polite", "prediction"]]
  result.to_csv("PATH_TO_OUTPUT_FILE", index=False)


[training, unlabeled] = receive_data.receive_data()
train_model(training, unlabeled)
