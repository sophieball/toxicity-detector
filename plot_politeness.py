# Lint as: python3
"""Collect fighting words and politeness stats"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

type_colors = {
    "Gratitude": "pos",
    "Deference": "pos",
    "Indirect_(greeting)": "pos",
    "HASPOSITIVE": "pos",
    "HASNEGATIVE": "imp",
    "Apologizing": "neg",
    "Please": "neg",
    "Please_start": "imp",
    "Indirect_(btw)": "neg",
    "Direct_question": "imp",
    "Direct_start": "imp",
    "SUBJUNCTIVE": "neg",
    "INDICATIVE": "neg",
    "1st_person_start": "neg",
    "1st_person_pl.": "neg",
    "1st_person": "neg",
    "2nd_person": "neg",
    "2nd_person_start": "imp",
    "Hedges": "neg",
    "Factuality": "imp"
}

colors = {"pos": "g", "neg": "yellow", "imp": "w"}
hatches = {"pos": "\\", "neg": "/", "imp": "x"}
ecs = {"pos": "w", "neg": "black", "imp": "black"}

name = {
    "Gratitude": "Gratitude",
    "Deference": "Deference",
    "Indirect_(greeting)": "Greeting",
    "HASPOSITIVE": "Positive lexicon",
    "HASNEGATIVE": "Negative lexicon",
    "Apologizing": "Apologizing",
    "Please": "Please",
    "Please_start": "Please start",
    "Indirect_(btw)": "Indirect (btw)",
    "Direct_question": "Direct question",
    "Direct_start": "Direct start",
    "SUBJUNCTIVE": "Counterfactual modal",
    "INDICATIVE": "Indicative modal",
    "1st_person_start": "1st person start",
    "1st_person_pl.": "1st person pl.",
    "1st_person": "1st person",
    "2nd_person": "2nd person",
    "2nd_person_start": "2nd person start",
    "Hedges": "Hedges",
    "Factuality": "Factuality"
}

order = [
    "Gratitude", "Deference", "Indirect_(greeting)", "HASPOSITIVE",
    "Apologizing", "Please", "Indirect_(btw)", "SUBJUNCTIVE", "INDICATIVE",
    "1st_person_start", "1st_person_pl.", "1st_person", "2nd_person", "Hedges",
    "Factuality", "HASNEGATIVE", "Please_start", "Direct_question",
    "Direct_start", "2nd_person_start"
]


def save_plot(scores, file_name, y_lim):
  print([scores[t] for t in order])
  #scores = scores.to_dict()["Averages"]
  plt.figure(dpi=200, figsize=(9, 6))
  bars = plt.bar(
      list(range(len(scores))), [scores[t] for t in order],
      color=[colors[type_colors[t]] for t in order],
      ec=[ecs[type_colors[t]] for t in order],
      tick_label=order,
      align="edge")
  bar_hatches = [hatches[type_colors[t]] for t in order]
  for bar, hatch in zip(bars, bar_hatches):
    bar.set_hatch(hatch)
  pos_patch = mpatches.Patch(
      color=colors["pos"],
      hatch=hatches["pos"],
      ec=ecs["pos"],
      label="Positive politeness")
  neg_patch = mpatches.Patch(
      color=colors["neg"],
      hatch=hatches["neg"],
      ec=ecs["neg"],
      label="Negative politeness")
  imp_patch = mpatches.Patch(
      color=colors["imp"],
      hatch=hatches["imp"],
      ec=ecs["imp"],
      label="Impoliteness")
  plt.legend(handles=[pos_patch, neg_patch, imp_patch])
  plt.xticks(
      np.arange(.4,
                len(scores) + .4),
      labels=[name[t] for t in order],
      rotation=45,
      ha="right",
      size=9)
  plt.ylabel("Occurrences per Utterance", size=10)
  plt.yticks(size=15)
  plt.tight_layout()
  if y_lim != None:
    plt.ylim(0, y_lim)
  plt.savefig(file_name)


scores = pd.read_csv("~/Desktop/polite_strategies_label_1 (1).csv", index_col=0)
#scores = pd.read_csv("G_polite_1.csv")
print(scores)
scores = scores.drop(index=["HASHEDGE"], axis=0)
scores = scores.to_dict("dict")["Averages"]
#scores = scores.rename(columns={"index": "type", "Average": "Average"})
#get_polite = lambda a:type_colors[a]
#scores["politeness"] = scores["type"].map(get_polite)
print(scores)
save_plot(scores, "G_polite_1.pdf", 10)
