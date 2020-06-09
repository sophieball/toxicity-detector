from suite import Suite
import pandas as pd
from classifiers import *
from pathlib import Path
from os.path import join
import os
path = Path(os.path.abspath("."))

def train_model():
  s = Suite()

  print("Loading training data.")
  train_dat = pd.read_csv(join(path, "data/training_data_label.csv"))
  s.set_train_set(train_dat)
  test_dat = pd.read_csv(join(path, "data/test_comments.csv"))
  s.set_test_set(test_dat)
  s.set_model(svm_model)

  s.set_ratios([2])
  s.add_parameter("C", [.05])
  s.add_parameter("gamma", [2])

  s.features = ["perspective_score", "stanford_polite"]
  s.nice_features = ["perspective_score", "stanford_polite"]

  # train the model, test all combinations of hyper parameter
  s.self_issue_classification_all()
  # fit the model on test data
  result = s.test_issue_classifications_from_comments_all()
  result.to_csv("classification_results.csv", index=False)

train_model()
