# Lint as: python3
"""Test applying GH model on some dummy data"""

import unittest
from main import apply_GH_model
import sys
import io
import pandas as pd


class TestApplyGHModel(unittest.TestCase):

  def test_apply_GH_model(self):
    dat = []
    dat.append({"id": 1, "text": "great", "label": False, "training": True})
    dat.append({
        "id":
            2,
        "text":
            "What is not subject to translations is your -1 reaction and your closing of the Issue",
        "label":
            False,
        "training":
            True
    })
    dat.append({
        "id":
            3,
        "text":
            "If you want to make it quicker you can allways provide the code here and I Will implement it in seconds. And if not, after telling you that I Will do, you come here with hurry and get dissapointed because I tell you the priorities of the mission development????? May I say that I am not your fucking slave or is it incorrect?",
        "label":
            True,
        "training":
            True
    })
    dat.append({"id": 4, "text": "bananas", "label": False, "training": True})
    dat.append({
        "id": 5,
        "text": "orange jerks",
        "label": None,
        "training": False
    })
    df = pd.DataFrame(dat)
    apply_GH_model.apply_model(df)

    # test results
    res = pd.read_csv("main/output/classification_results.csv")
    self.assertEqual(len(res), 5)
    self.assertTrue(math.isnan(res["prediction"]))
    self.assertEqual(res.loc[res["_id" == 3]]["prediction"], 1)
    self.assertEqual(res.loc[res["_id" == 2]]["prediction"], 0)
