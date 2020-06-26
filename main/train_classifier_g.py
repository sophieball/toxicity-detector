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
import numpy as np


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
  s.set_model(classifiers.svm_model)  #random_forest_model)

  # list the set of parameters you want to try out
  s.set_ratios([2])
  s.add_parameter("C", [.05])
  s.add_parameter("gamma", [2])
  # RF params
  #s.add_parameter("n_estimators",
  #                [int(x) for x in np.linspace(start=200, stop=2000, num=10)])
  #s.add_parameter("max_features", ["auto", "sqrt"])
  #s.add_parameter("max_depth", [int(x) for x in np.linspace(10, 110, num=11)])

  # select features
  s.features = ["perspective_score", "politeness"]
  s.nice_features = ["perspective_score", "politeness"]

  # train the model, test all combinations of hyper parameter
  model = s.self_issue_classification_all()
  # save the model
  model_out = open("src/pickles/RF_model.p", "wb")
  pickle.dump(model, model_out)
  model_out.close()

  # fit the model on test data
  result = s.test_issue_classifications_from_comments_all()

  # only write the id and label to file
  if G_data:
    result = result.rename(columns={"_id": "id"})
  result = result[[
      "id", "perspective_score", "politeness", "raw_prediction", "prediction",
      "is_SE", "self_angry"
  ]]
  result.to_csv("classification_results.csv", index=False)
  logging.info("Number of 1's in raw prediction: {}.".format(
      sum(result["raw_prediction"])))
  logging.info("Number of data flipped due to SE: {}.".format(
      sum(result["is_SE"])))
  logging.info("Number of data flipped due to self angry: {}.".format(
      sum(result["self_angry"])))


G_data = True
[training, unlabeled] = receive_data.receive_data()
train_model(training, unlabeled)
