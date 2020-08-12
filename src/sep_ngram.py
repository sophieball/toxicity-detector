# Lint as: python3
"""separate fighting words output into 3 cols
"""

import pandas as pd
import sys

if len(sys.argv) < 3:
  print("Usage: python3 src/sep_ngram.py <input> <output>")
  exit(0)

dat = pd.read_csv(sys.argv[1])

x = lambda a: len(str(a).split(" "))
dat["n"] = dat["ngram"].map(x)
dat = dat.sort_values(["n", "abs_z-score"], ascending=False)
dat1 = dat.loc[dat["n"] == 1]
dat2 = dat.loc[dat["n"] == 2]
dat3 = dat.loc[dat["n"] >= 3]


def convert_name(x):
  if x == "class1":
    return "pushback"
  else:
    return "non-pushback"


out = open(sys.argv[2], "w")
out.write(
    "unigram,label,abs(z-score),bigram,label,abs(z-score),ngram,label,abs(z-score)\n"
)
for i in range(20):
  row1 = dat1.iloc[i]
  out.write("{},{},{},".format(row1["ngram"], convert_name(row1["class"]),
                               row1["abs_z-score"]))
  row1 = dat2.iloc[i]
  out.write("{},{},{},".format(row1["ngram"], convert_name(row1["class"]),
                               row1["abs_z-score"]))
  row1 = dat3.iloc[i]
  out.write("{},{},{}\n".format(row1["ngram"], convert_name(row1["class"]),
                                row1["abs_z-score"]))
