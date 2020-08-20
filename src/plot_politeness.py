# Lint as: python3
"""Collect fighting words and politeness stats"""

import sys
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

#colors = {"pos": "g", "neg": "yellow", "imp": "w"}
colors = {"label=1": "yellow", "label=0": "g"}
hatches = {"pos": "\\", "neg": "/", "imp": "x"}
#hatches = {"label=1": "\\", "label=0": ""}
ecs = {"pos": "black", "neg": "black", "imp": "black"}

pos = {
    "Gratitude": "Gratitude",
    "Deference": "Deference",
    "Indirect_(greeting)": "Greeting",
}

neg = {
    "Apologizing": "Apologizing",
    "Please": "Please",
    "Indirect_(btw)": "By the way",
    "SUBJUNCTIVE": "Counterfactual",# modal",
    "INDICATIVE": "Indicative modal",
    "1st_person_pl.": "1st person pl.",
    "1st_person": "1st person",
    "2nd_person": "2nd person",
    "Hedges": "Hedges",
    "1st_person_start": "1st person start",
}

imp = {
    "Factuality": "Factuality",
    "Please_start": "Please start",
    "2nd_person_start": "2nd person start",
    "Direct_question": "Direct question",
    "Direct_start": "Direct start",
}

order = {
    "Gratitude": "Gratitude",
    "Deference": "Deference",
    "Indirect_(greeting)": "Greeting",
    "Apologizing": "Apologizing",
    "Please": "Please",
    "Indirect_(btw)": "By the way",
    "SUBJUNCTIVE": "Counterfactual",# modal",
    "INDICATIVE": "Indicative modal",
    "1st_person_pl.": "1st person pl.",
    "1st_person": "1st person",
    "2nd_person": "2nd person",
    "Hedges": "Hedges",
    "1st_person_start": "1st person start",
    "Factuality": "Factuality",
    "Please_start": "Please start",
    "2nd_person_start": "2nd person start",
    "Direct_question": "Direct question",
    "Direct_start": "Direct start",
}


def make_patch():
  patch1 = mpatches.Patch(
      color=colors["label=1"],
      #hatch=hatches["pos"],
      #ec=ecs["pos"],
      label="Pushback")
  patch0 = mpatches.Patch(
      color=colors["label=0"],
      #hatch=hatches["neg"],
      #ec=ecs["neg"],
      label="Non-pushback")
  patch2 = mpatches.Patch(
      color="w",#colors["imp"],
      hatch=hatches["pos"],
      ec=ecs["imp"],
      label="Positive politeness")
  patch3 = mpatches.Patch(
      color="w",#colors["imp"],
      hatch=hatches["neg"],
      ec=ecs["pos"],
      label="Negative politeness")
  patch4 = mpatches.Patch(
      color="w",#colors["pos"],
      hatch=hatches["imp"],
      ec=ecs["neg"],
      label="Impoliteness")
  return [patch0, patch1, patch2, patch3, patch4]


# input: pd.DataFrame, str, int
def save_plot(scores, file_name, x_lim):
  scores = scores.drop(index=["HASHEDGE"], axis=0)
  scores_d = scores.to_dict()
  scores1 = scores_d["label=1"]
  scores0 = scores_d["label=0"]
  plt.figure(dpi=200, figsize=(60, 75))
  bar_height = 2.5

  ind = np.arange(0, 6*len(order), 6)
  for i in range(len(pos), len(pos)+len(neg)):
    ind[i] += 4
  for i in range(len(pos)+len(neg), len(order)):
    ind[i] += 8

  bars = plt.barh(
      ind, [scores1[t] for t in order],
      height=bar_height,
      color=[colors["label=1"]],#type_colors[t]] for t in order],
      ec=[ecs[type_colors[t]] for t in order],
      tick_label=[order[t] for t in order],
      align="edge")
  bar_hatches = [hatches[type_colors[t]] for t in order]
  for bar, hatch in zip(bars, bar_hatches):
    bar.set_hatch(hatch)

  # label = 0
  bars = plt.barh(
      ind + bar_height, [scores0[t] for t in order],
      height=bar_height,
      color=[colors["label=0"]],#type_colors[t]] for t in order],
      ec=[ecs[type_colors[t]] for t in order],
      tick_label=[order[t] for t in order],
      align="edge")
  bar_hatches = [hatches[type_colors[t]] for t in order]
  for bar, hatch in zip(bars, bar_hatches):
    bar.set_hatch(hatch)

  patch = make_patch()
  plt.legend(handles=patch, fontsize=80)
  plt.xlabel("Occurrences per Sentence", size=80)

  x_lim = 0.035
  plt.xticks(np.arange(0, x_lim+0.005, 0.002))
  plt.tick_params(labelsize=50)
  #plt.tight_layout()
  ax = plt.gca()
  ax.xaxis.grid(True)
  plt.xlim(0, x_lim)

  plt.savefig(file_name)


if __name__ == "__main__":
  if len(sys.argv) == 1:
    print("Usage: python3 plot_politeness.py <input> <output>")
  else:
    i_f = sys.argv[1]
    o_f = sys.argv[2]
  scores = pd.read_csv(i_f, index_col=0)
  save_plot(scores, o_f, 0.1)
