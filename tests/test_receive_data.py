# Lint as: python3
"""Test receive data from standard input in CSV format"""

import unittest
from src import receive_data
import sys
import io
import pandas as pd


class TestReceiveData(unittest.TestCase):

  def test_receive_data(self):
    dat = []
    dat.append({"id": 1, "text": "great", "label": False, "training": True})
    dat.append({"id": 2, "text": "not great", "label": False, "training": True})
    dat.append({"id": 3, "text": "you jerk", "label": True, "training": True})
    dat.append({"id": 4, "text": "bananas", "label": False, "training": True})
    dat.append({
        "id": 5,
        "text": "orange jerks",
        "label": None,
        "training": False
    })
    df = pd.DataFrame(
        dat, columns=["id", "text", "label",
                      "training"]).to_csv(index=False).strip("\n").split("\n")

    sys.stdin = io.StringIO("\n".join(df))
    [training, unlabeled] = receive_data.receive_data()

    self.assertTrue("_id" in training.columns)
    self.assertFalse("label" in unlabeled.columns)
    self.assertEqual(len(unlabeled), 1)
    self.assertEqual(len(training), 4)
    self.assertEqual(len(training.loc[training["label"] == 1]), 1)
    self.assertEqual(type(training.iloc[0]["_id"]), str)
