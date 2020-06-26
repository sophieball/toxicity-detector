# Lint as: python3
"""Test applying GH model on some dummy data"""

import unittest
import sys
import io
import pandas as pd
import pickle


class TestGHModel(unittest.TestCase):

  def test_GH_model(self):
    # set model
    model_file = open("src/pickles/SVM_pretrained_model.p", "rb")
    model = pickle.load(model_file)
    model_file.close()

    # load sample data
    test_data = [
    {
        "_id": "jekyll/jekyll/6948/386652799",
        "perspective_score": 0.060521,
        "politeness": 0.095238
    },
    {
        "_id": "GoMint/GoMint/406/434020009",
        "perspective_score": 0.051550,
        "politeness": 0.00000
    },
    {
        "_id": "bootstrap/twbs/3057/5142140",
        "perspective_score": 0.531060,
        "politeness": 0.428571
    },
    {
        "_id": "openvpn-client/dperson/165/445486679",
        "perspective_score": 0.222210,
        "politeness": 0.190476
    }]


    features = ["perspective_score", "politeness"]
    test_list = [list(x) for x in test_data[features].values]

    test_data["prediction"] = model.predict(test_list)
    self.assertTrue(math.isnan(test_data["prediction"]))
    self.assertEqual(test_data.loc[test_data["_id" == "openvpn-client/dperson/165/445486679"]]["prediction"], 1)
    self.assertEqual(test_data.loc[test_data["_id" == "bootstrap/twbs/3057/5142140"]]["prediction"], 0)
