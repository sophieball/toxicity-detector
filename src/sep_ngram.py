# Lint as: python3
"""separate fighting words output into 3 cols"""

import pandas as pd
import sys


def convert_name(x):
  if x == "class1":
    return "pushback code review comments"
  else:
    return "non-pushback code review comments"


# input: pd.DataFrame, str, int
def sep_ngram(dat, out_name, top_n):
  x = lambda a: len(str(a).split(" "))
  dat["n"] = dat["ngram"].map(x)
  dat = dat.sort_values(["z-score"], ascending=False)
  dat1 = dat.loc[dat["n"] == 1]
  dat2 = dat.loc[dat["n"] == 2]
  dat3 = dat.loc[dat["n"] >= 3]

  out = open(out_name, "w")
  out.write(
      "unigram,label,z-score,count in class1,count in class2,")
  out.write("bigram,label,z-score,count in class1,count in class2,")
  out.write("ngram,label,z-score,count in class1,count in class2\n")
  top_n = min(top_n, len(dat3))
  for i in range(int(top_n/2)):
    row = dat1.iloc[i]
    out.write("{},{},{},{},{},".format(row["ngram"], convert_name(row["class"]),
                                 row["z-score"],
                                 row["count in class1"],
                                 row["count in class2"]))
    row = dat2.iloc[i]
    out.write("{},{},{},{},{},".format(row["ngram"], convert_name(row["class"]),
                                 row["z-score"],
                                 row["count in class1"],
                                 row["count in class2"]))
    row = dat3.iloc[i]
    out.write("{},{},{},{},{}\n".format(row["ngram"], convert_name(row["class"]),
                                 row["z-score"],
                                 row["count in class1"],
                                 row["count in class2"]))
  for i in range(int(top_n/2), 0, -1):
    row = dat1.iloc[-i]
    out.write("{},{},{},{},{},".format(row["ngram"], convert_name(row["class"]),
                                 row["z-score"],
                                 row["count in class1"],
                                 row["count in class2"]))
    row = dat2.iloc[-i]
    out.write("{},{},{},{},{},".format(row["ngram"], convert_name(row["class"]),
                                 row["z-score"],
                                 row["count in class1"],
                                 row["count in class2"]))
    row = dat3.iloc[-i]
    out.write("{},{},{},{},{}\n".format(row["ngram"], convert_name(row["class"]),
                                 row["z-score"],
                                 row["count in class1"],
                                 row["count in class2"]))

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage: python3 src/sep_ngram.py <input> <output>")
    exit(0)

  dat = pd.read_csv(sys.argv[1])
  sep_ngrams(dat, sys.argv[2])
