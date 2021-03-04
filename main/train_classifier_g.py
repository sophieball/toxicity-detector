# main program for training a classifier and applies it to new data
from src import download_data
download_data.download_data()

import logging
logging.basicConfig(
    filename="train_classifier.log", filemode="w", level=logging.INFO)
import os

from src import receive_data
from src import classifiers
from src import suite
import pandas as pd
import pathlib
import pickle
import numpy as np
import sys
import time

feature_set = [["perspective_score", "politeness"], 
               ["perspective_score", "identity_attack", "politeness", "rounds", "shepherd_time"],
               ["rounds", "shepherd_time"],
               ["perspective_score", 
                "identity_attack",
                "politeness",
                "Please", 
                "Please_start", 
                "HASHEDGE", 
                #"Indirect_(btw)", 
                #"Hedges", 
                #"Factuality",
       					#"Deference", 
                "Gratitude", 
                #"Apologizing", 
                #"1st_person_pl.", 
                "1st_person",
       					"1st_person_start", 
                "2nd_person", 
                "2nd_person_start",
       					#"Indirect_(greeting)", 
                #"Direct_question", 
                #"Direct_start", 
                "HASPOSITIVE",
       					"HASNEGATIVE", 
                #"SUBJUNCTIVE", 
                #"INDICATIVE",
                #"num_words", 
                "length"
              ],

              ["perspective_score", 
                "identity_attack",
                "Please", 
                "Please_start", 
                "HASHEDGE", 
                #"Indirect_(btw)", 
                #"Hedges", 
                #"Factuality",
       					#"Deference", 
                "Gratitude", 
                #"Apologizing", 
                #"1st_person_pl.", 
                "1st_person",
       					"1st_person_start", 
                "2nd_person", 
                "2nd_person_start",
       					#"Indirect_(greeting)", 
                #"Direct_question", 
                #"Direct_start", 
                "HASPOSITIVE",
       					"HASNEGATIVE", 
                #"SUBJUNCTIVE", 
                #"INDICATIVE",
                "length"
              ],

              ["perspective_score", 
                "identity_attack",
                "Please", 
                "Please_start", 
                "HASHEDGE", 
                #"Indirect_(btw)", 
                #"Hedges", 
                #"Factuality",
       					#"Deference", 
                "Gratitude", 
                #"Apologizing", 
                #"1st_person_pl.", 
                "1st_person",
       					"1st_person_start", 
                "2nd_person", 
                "2nd_person_start",
       					#"Indirect_(greeting)", 
                #"Direct_question", 
                #"Direct_start", 
                "HASPOSITIVE",
       					"HASNEGATIVE", 
                #"SUBJUNCTIVE", 
                #"INDICATIVE",
              ],

              ["perspective_score", 
                "identity_attack",
                "Please", 
                "Please_start", 
                "HASHEDGE", 
                #"Indirect_(btw)", 
                #"Hedges", 
                #"Factuality",
       					#"Deference", 
                "Gratitude", 
                #"Apologizing", 
                #"1st_person_pl.", 
                "1st_person",
       					"1st_person_start", 
                "2nd_person", 
                "2nd_person_start",
       					#"Indirect_(greeting)", 
                #"Direct_question", 
                #"Direct_start", 
                "HASPOSITIVE",
       					"HASNEGATIVE", 
                #"SUBJUNCTIVE", 
                #"INDICATIVE",
                "rounds",
                "shepherd_time",
                "review_time"
              ],
              ]


# train the classifier using the result of a SQL query
def train_model(training_data, model_name="svm", pretrain=False, G=False):
  s = suite.Suite()
  s.set_G(G)

  logging.info("Loading data.")
  # 4 columns: _id, text, label(0/1), training(1)
  # this will later be split into train/test data
  s.set_train_set(training_data)

  # select model
  if model_name == "svm":
    s.set_model_function(classifiers.svm_model)

    # list the set of parameters you want to try out
    s.set_ratios([1, 2, 3, 5, 10])
    s.set_parameters({
        "C": [0.05, 0.1, 0.5, 1, 10, 20, 25, 30, 50],
        "gamma": [1, 2, 2.5, 3, 5],
        "kernel": ["sigmoid", "rbf"]
    })
  elif model_name == "rf":
    s.set_model_function(classifiers.random_forest_model)
    # RF params
    s.add_parameter("n_estimators",
                    [int(x) for x in np.linspace(start=200, stop=2000, num=10)])
    s.add_parameter("max_features", ["auto", "sqrt"])
    s.add_parameter("max_depth", [int(x) for x in np.linspace(10, 110, num=11)])
  elif model_name == "lg":
    s.set_model_function(classifiers.logistic_model)
    # RF params
    s.add_parameter("penalty", ["l1", "l2", "13", "20", "5"])
    s.add_parameter("C", np.logspace(-4, 4, 60))

  # select features
  for fid, features in enumerate(feature_set[:1]):
    s.features = features
    s.nice_features = features
    logging.info("Features: {}".format(", ".join(features)))

    # train the model, test all combinations of hyper parameter
    model = s.self_issue_classification_all(model_name, fid)
    # save the model
    if pretrain:
      model_out = open(
          "src/pickles/{}_pretrained_model_{}.p".format(model_name.upper(),
                                                        str(fid)), "wb")
    else:
      model_out = open(
          "src/pickles/{}_model_{}.p".format(model_name.upper(), str(fid)),
          "wb")
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


def predict_unlabeled(unlabeled_data, trained_model, features, G_data=True):
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
    logging.info("Training the model and predicting labels.")
    [training, unlabeled] = receive_data.receive_data()
    trained_model = train_model(training, G=False)
    #trained_model = train_model(training, model_name="rf", G=False)
    #trained_model = train_model(training, model_name="lg")
    logging.info("Trained model saved in {}".format("`" + os.getcwd() +
                                                    "/src/pickles/"))
  else:
    logging.info("Training the model and predicting labels.")
    [training, unlabeled] = receive_data.receive_data()
    trained_model = train_model(training, G=True)
    trained_model = train_model(training, model_name="rf", G=True)
    #trained_model = train_model(training, model_name="lg")
    logging.info("Trained model saved in {}".format("`" + os.getcwd() +
                                                    "/src/pickles/"))
  logging.info(
        "Prepared training dataset, it took {} seconds".format(time.time() - \
                                                               start_time))
  print("Log saved in {}".format("`" + os.getcwd() + "/train_classifier.log`"))
  print("Prediction result saved in {}".format("`" + os.getcwd() +
                                               "/classification_results.csv`"))
