# main program for training a classifier and applies it to new data
from src import download_data
download_data.download_data()

import logging
logging.basicConfig(
    filename="train_classifier.log", filemode="w", level=logging.INFO)

from src import receive_data
from src import classifiers
from src import suite
import pandas as pd
import pathlib
import pickle
import numpy as np
import sys
import time

features = ["perspective_score", "num_url", "politeness"]


# train the classifier using the result of a SQL query
def train_model(training_data, model_name="svm", pretrain=False):
  s = suite.Suite()

  logging.info("Loading data.")
  # 4 columns: _id, text, label(0/1), training(1)
  # this will later be split into train/test data
  s.set_train_set(training_data)

  # select model
  if model_name == "svm":
    s.set_model(classifiers.svm_model)

    # list the set of parameters you want to try out
    s.set_ratios([1, 5, 10])
    s.set_parameters({
        "C": [0.05, 0.1, 0.5, 1, 10, 20],
        "gamma": [1, 2, 2.5, 3],
        "kernel": ["sigmoid"]
    })
  elif model_name == "rf":
    s.set_model(classifiers.random_forest_model)
    # RF params
    s.add_parameter("n_estimators",
                    [int(x) for x in np.linspace(start=200, stop=2000, num=10)])
    s.add_parameter("max_features", ["auto", "sqrt"])
    s.add_parameter("max_depth", [int(x) for x in np.linspace(10, 110, num=11)])

  # select features
  s.features = features
  s.nice_features = features
  logging.info("Features: {}".format(", ".join(features)))

  # train the model, test all combinations of hyper parameter
  model = s.self_issue_classification_all()
  # save the model
  if pretrain:
    model_out = open(
        "src/pickles/" + model_name.upper() + "_pretrained_model.p", "wb")
  else:
    model_out = open("src/pickles/" + model_name.upper() + "_model.p", "wb")
  pickle.dump(model, model_out)
  model_out.close()
  logging.info("Model is stored at {}.".format(
      str(pathlib.Path(__file__).parent.name) + "/src/pickles/"))
  result = s.all_train_data
  logging.info("Number of 1's in raw prediction: {}.".format(
      sum(result["raw_prediction"])))
  logging.info("Number of data flipped due to SE: {}.".format(
      len(result.loc[result["is_SE"] == 1])))
  logging.info("Number of data flipped due to self angry: {}.".format(
      len(result.loc[result["self_angry"] == "self"])))
  return model


def predict_unlabeled(unlabeled_data, trained_model, G_data=True):
  s = suite.Suite()
  # 3 columns: _id, text, training(0)
  s.set_unlabeled_set(unlabeled_data)

  s.set_trained_model(trained_model)
  # select features
  s.features = features
  s.nice_features = features
  logging.info("Features: {}".format(", ".join(features)))

  # fit the model on test data
  result = s.test_issue_classifications_from_comments_all()

  # remove text from the output
  if G_data:
    result = result.rename(columns={"_id": "id"})
  result = result.drop(["text", "original_text"], axis=1)
  result.to_csv("classification_results.csv", index=False)


if __name__ == "__main__":
  start_time = time.time()
  if len(sys.argv) > 1:
    if sys.argv[1] == "pretrain":
      logging.info("Training on GH data.")
      [training, _] = receive_data.receive_data()
      train_model(training, pretrain=True)
    if sys.argv[1] == "test":
      logging.info("Applying pre-trained GH model.")
      if len(sys.argv) >= 2:
        # set model
        model_file = open(sys.argv[2], "rb")
        trained_model = pickle.load(model_file)
        model_file.close()
        [_, unlabeled] = receive_data.receive_data()
        predict_unlabeled(unlabeled, trained_model)
      else:
        logging.info("Please povide the name of the model's pickle file.")
        logging.info(
            "The file should be among the data dependencies in the BUILD file.")
  else:
    logging.info("Training the model and predicting labels.")
    [training, unlabeled] = receive_data.receive_data()
    trained_model = train_model(training)
    predict_unlabeled(unlabeled, trained_model)
    logging.info("Trained model saved in {}".format(
        "`bazel-bin/main/feed_data.runfiles/__main__/src/pickles/"))
  logging.info(
        "Prepared training dataset, it took {} seconds".format(time.time() - \
                                                               start_time))
  print("Log saved in {}".format(
      "`bazel-bin/main/feed_data.runfiles/__main__/train_classifier.log`"))
  print("Prediction result saved in {}".format(
      "`bazel-bin/main/feed_data.runfiles/__main__/classification_results.csv`")
       )
