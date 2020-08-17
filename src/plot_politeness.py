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

colors = {"pos": "g", "neg": "yellow", "imp": "w"}
colors = {"label=1": "yellow", "label=0": "g"}
hatches = {"pos": "\\", "neg": "/", "imp": "x"}
hatches = {"label=1": "\\", "label=0": ""}
ecs = {"pos": "w", "neg": "black", "imp": "black"}

pos = [{
    "Gratitude": "Gratitude",
    "Deference": "Deference",
    "Indirect_(greeting)": "Greeting",
    "HASPOSITIVE": "Positive lexicon",
}, {}]

neg = [{
    "Apologizing": "Apologizing",
    "Please": "Please",
    "Indirect_(btw)": "Indirect (btw)",
    "SUBJUNCTIVE": "Counterfactual modal",
    "INDICATIVE": "Indicative modal",
    "1st_person_pl.": "1st person pl.",
    "1st_person": "1st person",
    "2nd_person": "2nd person",
    "Hedges": "Hedges",
}, {
    "1st_person_start": "1st person start",
}]

imp = [{
    "Factuality": "Factuality",
    "HASNEGATIVE": "Negative lexicon",
}, {
    "Please_start": "Please start",
    "2nd_person_start": "2nd person start",
    "Direct_question": "Direct question",
    "Direct_start": "Direct start",
}]

order = [
    "Gratitude", "Deference", "Indirect_(greeting)", "HASPOSITIVE",
    "Apologizing", "Please", "Indirect_(btw)", "SUBJUNCTIVE", "INDICATIVE",
    "1st_person_start", "1st_person_pl.", "1st_person", "2nd_person", "Hedges",
    "Factuality", "HASNEGATIVE", "Please_start", "Direct_question",
    "Direct_start", "2nd_person_start"
]


def make_patch(polite_type, polite_name):
  patch1 = mpatches.Patch(
      color=colors["label=1"],#polite_type],
      hatch=hatches["label=1"],#polite_type],
      #ec=ecs[polite_type],
      label="label=1")
  patch0 = mpatches.Patch(
      color=colors["label=0"],#polite_type],
      hatch=hatches["label=0"],#polite_type],
      #ec=ecs[polite_type],
      label="label=0")
  """
  patch = mpatches.Patch(
      color=colors,#polite_type],
      hatch=hatches,#polite_type],
      #ec=ecs[polite_type],
      label=["label=1", "label=0"])
  """
  return [patch0, patch1]


# input: pd.DataFrame, str, int
def save_plot(scores, file_name, x_lim):
  scores = scores.drop(index=["HASHEDGE"], axis=0)
  scores_d = scores.to_dict()
  scores1 = scores_d["label=1"]
  scores0 = scores_d["label=0"]
  #plt.figure(dpi=200, figsize=(35, 15))
  fig, ax = plt.subplots(3, 2)
  fig.tight_layout()
  fig.subplots_adjust(left=0.18, right=0.95, wspace=0.4)
  bar_height = 0.35

  def plot_subplots(i, j, keys, polite_type, polite_name):
    if len(keys) == 0:
      fig.delaxes(ax[i, j])
      return
    # subplots
    ind = np.arange(len(keys))
    bars = ax[i, j].barh(
        ind, [scores1[t] for t in keys],
        height=bar_height,
        color=[colors["label=1"] for t in keys],#type_colors[t]] for t in keys],
        ec="black",
        #ec=[ecs[type_colors[t]] for t in keys],
        tick_label=list(keys.values()),
        align="edge")
    bar_hatches = [hatches["label=1"] for t in keys]#type_colors[t]] for t in keys]
    for bar, hatch in zip(bars, bar_hatches):
      bar.set_hatch(hatch)

    bars = ax[i, j].barh(
        ind + bar_height, [scores0[t] for t in keys],
        height=bar_height,
        color=[colors["label=0"] for t in keys],#type_colors[t]] for t in keys],
        #ec=[ecs[type_colors[t]] for t in keys],
        ec="black",
        tick_label=list(keys.values()),
        align="edge")

    patch = make_patch(polite_type, polite_name)
    ax[i, j].legend(handles=patch, fontsize=5)
    if j == 0:
      ax[i, j].set_ylabel(polite_name, size=8)
    if i == 2:
      ax[i, j].set_xlabel("Occurrences per Utterance", size=5)

    if j == 1:
      x_lim = 0.1
      ax[i, j].set_xticks(np.arange(0, x_lim, 0.02))
    else:
      x_lim = 0.01
      ax[i, j].set_xticks(np.arange(0, x_lim, 0.002))
    ax[i, j].tick_params(labelsize=5)
    #ax[i, j].tight_layout()
    ax[i, j].xaxis.grid()
    if x_lim != None:
      ax[i, j].set_xlim(0, x_lim)

  polite_types = ["pos", "neg", "imp"]
  polite_names = ["Positive politeness", "Negative politeness", "Impoliteness"]
  polite_keys = [pos, neg, imp]
  for i in range(3):
    for j in range(2):
      plot_subplots(i, j, polite_keys[i][j], polite_types[i], polite_names[i])
  plt.savefig(file_name)


if __name__ == "__main__":
  if len(sys.argv) == 1:
    print("Usage: python3 plot_politeness.py <input> <output>")
  else:
    i_f = sys.argv[1]
    o_f = sys.argv[2]
  scores = pd.read_csv(i_f, index_col=0)
  save_plot(scores, o_f, 0.1)
