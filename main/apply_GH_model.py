# Lint as: python3
"""Apply pre-trained model on test data"""

from src import download_data
download_data.download_data()

from src import receive_data
from src import suite
import logging
import pickle

logging.basicConfig(
    filename="tests/test_model.log", filemode="w", level=logging.INFO)
logging.basicConfig(level=logging.INFO)


# input: a pd.DataFrame
def apply_model(unlabeled_data):
  s = suite.Suite()

  # process data
  s.set_unlabeled_set(unlabeled_data)

  # set model
  model_file = open("src/pickles/SVM_model.p", "rb")
  model = pickle.load(model_file)
  model_file.close()
  s.set_trained_model(model)

  # select features
  s.features = ["perspective_score", "stanford_polite"]
  s.nice_features = ["perspective_score", "stanford_polite"]

  # fit the model on test data
  result = s.test_issue_classifications_from_comments_all()

  # only write the id and label to file
  if "_id" in result.columns:
    result = result.rename(columns={"_id": "id"})
  result = result.rename(columns={"stanford_polite": "politeness_score"})
  result = result[[
      "id", "perspective_score", "politeness_score", "prediction", "is_SE",
      "self_angry"
  ]]
  result.to_csv("main/output/classification_results.csv", index=False)


if __name__ == "__main__":
  [_, unlabeled_data] = receive_data.receive_data()
  apply_model(unlabeled_data)
