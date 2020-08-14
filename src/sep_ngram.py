# Lint as: python3
"""separate fighting words output into 3 cols
"""

import pandas as pd
import sys



def convert_name(x):
  if x == "class1":
    return "label==1"
  else:
    return "label==0"


# input: pd.DataFrame, str, int
def sep_ngram(dat, out_name, top_n):
  x = lambda a: len(str(a).split(" "))
  print(dat.columns)
  dat["n"] = dat["ngram"].map(x)
  dat = dat.sort_values(["n", "abs_z-score"], ascending=False)
  dat1 = dat.loc[dat["n"] == 1]
  dat2 = dat.loc[dat["n"] == 2]
  dat3 = dat.loc[dat["n"] >= 3]

  out = open(out_name, "w")
  out.write(
      "unigram,label,abs(z-score),bigram,label,abs(z-score),ngram,label,abs(z-score)\n"
  )
  for i in range(top_n):
    row = dat1.iloc[i]
    out.write("{},{},{},".format(row["ngram"], convert_name(row["class"]),
                                 row["abs_z-score"]))
    row = dat2.iloc[i]
    out.write("{},{},{},".format(row["ngram"], convert_name(row["class"]),
                                 row["abs_z-score"]))
    row = dat3.iloc[i]
    out.write("{},{},{}\n".format(row["ngram"], convert_name(row["class"]),
                                  row["abs_z-score"]))

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("Usage: python3 src/sep_ngram.py <input> <output>")
    exit(0)

  dat = pd.read_csv(sys.argv[1])
  sep_ngrams(dat, sys.argv[2])
