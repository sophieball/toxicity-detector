# Lint as: python3
"""Test receive data from standard input in CSV format
"""

import receive_data
import sys
import io
import pandas as pd

def test_receive_data():
  [training, testing] = receive_data.receive_data()
  assert len(testing) == 1
  assert len(training) == 4
  assert len(training.loc[training["label"] == 1]) == 1

def main():
  dat = []
  dat.append({"id": 1, "text": "great", "label": False, "training": True})
  dat.append({"id": 2, "text": "not great", "label": False, "training": True})
  dat.append({"id": 3, "text": "you jerk", "label": True, "training": True})
  dat.append({"id": 4, "text": "bananas", "label": False, "training": True})
  dat.append({"id": 5, "text": "orange jerks", "label": None, "training": False})
  df = pd.DataFrame(dat, columns=["id", "text", "label", "training"]).to_csv(index = False).strip("\n").split("\n")
  #sys.stdin = io.StringIO("\n".join([",".join(str(i) for i in k) for k in df.iterrows()]))
  sys.stdin = io.StringIO("\n".join(df))
  test_receive_data()

if __name__ == "__main__":
  main()
