# Lint as: python3
"""
Helper functions for predicting conversation faillure
"""

from src import download_data
download_data.download_data()
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedShuffleSplit
from sklearn.feature_selection import f_classif, SelectPercentile, chi2

from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from convokit import download
from convokit.prompt_types import PromptTypeWrapper
from convokit import PolitenessStrategies
from convokit import Corpus

import os
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
from src import conversation_struct


def compare_tox(df_ntox, df_tox, min_n=0):
  df_ntox.to_csv("df_ntox.csv")
  df_tox.to_csv("df_tox.csv")
  cols = df_ntox.columns
  num_feats_in_tox = df_tox[cols].sum().astype(int).rename("num_feat_tox")
  num_nfeats_in_tox = (1 -
                       df_tox[cols]).sum().astype(int).rename("num_nfeat_tox")
  num_feats_in_ntox = df_ntox[cols].sum().astype(int).rename("num_feat_ntox")
  num_nfeats_in_ntox = (
      1 - df_ntox[cols]).sum().astype(int).rename("num_nfeat_ntox")
  prop_tox = df_tox[cols].mean().rename("prop_tox")
  ref_prop_ntox = df_ntox[cols].mean().rename("prop_ntox")
  n_tox = len(df_tox)
  df = pd.concat([
      num_feats_in_tox,
      num_nfeats_in_tox,
      num_feats_in_ntox,
      num_nfeats_in_ntox,
      prop_tox,
      ref_prop_ntox,
  ],
                 axis=1)
  df.to_csv("tox.csv")
  df["num_total"] = df.num_feat_tox + df.num_feat_ntox
  df["log_odds"] = np.log(df.num_feat_tox) - np.log(df.num_nfeat_tox) \
      + np.log(df.num_nfeat_ntox) - np.log(df.num_feat_ntox)
  df["abs_log_odds"] = np.abs(df.log_odds)
  df["binom_p"] = df.apply(
      lambda x: stats.binom_test(x.num_feat_tox, n_tox, x.prop_tox), axis=1)
  df = df[df.num_total >= min_n]
  df["p"] = df["binom_p"].apply(lambda x: "%.3f" % x)
  df["pstars"] = df["binom_p"].apply(get_p_stars)
  return df.sort_values("log_odds", ascending=False)


# we are now ready to plot these comparisons. the following (rather intimidating) helper function
# produces a nicely-formatted plot:
def draw_figure(ax,
                first_cmp,
                second_cmp,
                title="",
                prompt_types=6,
                min_log_odds=.2,
                max_log_odds=np.Inf,
                min_freq=2,
                xlim=3):

  # selecting and sorting the features to plot, given minimum effect sizes and statistical significance
  # effect size >- 0.2, p < 0.05
  frequent_feats = first_cmp[first_cmp.num_total >= min_freq].index.union(
      second_cmp[second_cmp.num_total >= min_freq].index)
  lrg_effect_feats = first_cmp[
      (first_cmp.abs_log_odds >= min_log_odds)
      & (first_cmp.binom_p < .05)
      & (first_cmp.abs_log_odds != np.Inf)].index.union(
          second_cmp[(second_cmp.abs_log_odds >= min_log_odds)
                     & (second_cmp.binom_p < .05)
                     & (second_cmp.abs_log_odds != np.Inf)].index)
  feats_to_include = frequent_feats.intersection(lrg_effect_feats)
  feat_order = sorted(
      feats_to_include, key=lambda x: first_cmp.loc[x].log_odds, reverse=True)

  # parameters determining the look of the figure
  colors = ["darkorchid", "seagreen"]
  shapes = ["d", "s"]
  eps = .02
  star_eps = .035
  xlim = xlim
  min_log = .2
  gap_prop = 2
  label_size = 7
  title_size = 10
  radius = 144
  features = feat_order
  logging.info(features)
  ax.invert_yaxis()
  ax.plot([0, 0], [0, len(features) / gap_prop], color="black")

  # for each figure we plot the point according to effect size in the first and second comment,
  # and add axis labels denoting statistical significance
  yticks = []
  yticklabels = []
  for f_idx, feat in enumerate(features):
    curr_y = (f_idx + .5) / gap_prop
    yticks.append(curr_y)
    try:

      first_p = first_cmp.loc[feat].binom_p
      second_p = second_cmp.loc[feat].binom_p
      if first_cmp.loc[feat].abs_log_odds < min_log:
        first_face = "white"
      elif first_p >= 0.05:
        first_face = "white"
      else:
        first_face = colors[0]
      if second_cmp.loc[feat].abs_log_odds < min_log:
        second_face = "white"
      elif second_p >= 0.05:
        second_face = "white"
      else:
        second_face = colors[1]
      ax.plot([-1 * xlim, xlim], [curr_y, curr_y],
              "--",
              color="grey",
              zorder=0,
              linewidth=.5)

      ax.scatter([first_cmp.loc[feat].log_odds], [curr_y + eps],
                 s=radius,
                 edgecolor=colors[0],
                 marker=shapes[0],
                 zorder=20,
                 facecolors=first_face)
      ax.scatter([second_cmp.loc[feat].log_odds], [curr_y + eps],
                 s=radius,
                 edgecolor=colors[1],
                 marker=shapes[1],
                 zorder=10,
                 facecolors=second_face)

      first_pstr_len = len(get_p_stars(first_p))
      second_pstr_len = len(get_p_stars(second_p))
      p_str = np.array([" "] * 8)
      if first_pstr_len > 0:
        p_str[:first_pstr_len] = "*"
      if second_pstr_len > 0:
        p_str[-second_pstr_len:] = "⁺"

      feat_str = feat + "\n" + "".join(p_str)
      yticklabels.append(feat_str)
    except Exception as e:
      yticklabels.append("")

  # add the axis labels
  ax.margins(0.1)
  ax.set_xlabel("log-odds ratio", fontsize=8, family="serif")
  ax.set_xticks(
      [-xlim - .05, -2.5, -2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 2.5, xlim])
  ax.set_xticklabels(
      ["on-track", -2.5, -2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 2.5, "toxic"],
      fontsize=8,
      family="serif")
  ax.set_yticks(yticks)
  ax.set_yticklabels(yticklabels, fontsize=8, family="serif")
  ax.tick_params(axis="both", which="both", bottom="off", top="off", left="off")
  if title != "":
    ax.text(
        0,
        (len(features) + 2.25) / gap_prop,
        title,
        fontsize=title_size,
        family="serif",
        horizontalalignment="center",
    )
  return feat_order


# compare features exhibited
def clean_feature_name(feat):
  new_feat = feat.replace("feature_politeness",
                          "").replace("==", "").replace("_", "")
  split = new_feat.split()
  first, rest = split[0], "".join(split[1:]).lower()
  if first[0].isalpha():
    first = first.title()
  if "Hashedge" in first:
    return "Hedge (lexicon)"
  if "Hedges" in first:
    return "Hedge (dep. tree)"
  if "greeting" in feat:
    return "Greetings"
  cleaner_str = first + " " + rest
  return cleaner_str


def get_p_stars(x):
  if x < .001:
    return "***"
  elif x < .01:
    return "**"
  elif x < .05:
    return "*"
  else:
    return ""


def mode(seq):
  vals, counts = np.unique(seq, return_counts=True)
  return vals[np.argmax(counts)]


def run_pred_single(inputs, X, y):
  f_idx, (train_idx, test_idx) = inputs

  X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
  y_train, y_test = y[train_idx], y[test_idx]

  # use ANOVA F-value to select features
  base_clf = Pipeline([("scaler", StandardScaler()),
                       ("featselect", SelectPercentile(f_classif, 10)),
                       ("logreg", LogisticRegression(solver="liblinear"))])
  clf = GridSearchCV(
      base_clf, {
          "logreg__C": [10**i for i in range(-4, 4)],
          "featselect__percentile": list(range(10, 110, 10))
      },
      cv=3)

  clf.fit(X_train, y_train)

  y_scores = clf.predict_proba(X_test)[:, 1]
  y_pred = clf.predict(X_test)

  feature_weights = clf.best_estimator_.named_steps["logreg"].coef_.flatten()
  feature_mask = clf.best_estimator_.named_steps["featselect"].get_support()

  hyperparams = clf.best_params_
  logging.info(hyperparams)

  return (y_pred, y_scores, feature_weights, hyperparams, feature_mask)


def run_pred(X, y, fnames, groups):
  feature_weights = {}
  scores = np.asarray([np.nan for i in range(len(y))])
  y_pred = np.zeros(len(y))
  hyperparameters = defaultdict(list)
  #splits = list(enumerate(LeaveOneGroupOut().split(X, y, groups)))
  splits = list(
      enumerate(
          StratifiedShuffleSplit(n_splits=5, test_size=0.2,
                                 random_state=0).split(X, y)))
  chi_score, p_score = chi2(X, y)
  chi2_df = pd.DataFrame({
      "chi_score": chi_score,
      "p_score": p_score
  },
                         index=X.columns)
  accs = []

  with Pool(os.cpu_count()) as p:
    prediction_results = p.map(partial(run_pred_single, X=X, y=y), splits)

  fselect_pvals_all = []
  for i in range(len(splits)):
    f_idx, (train_idx, test_idx) = splits[i]
    y_pred_i, y_scores_i, weights_i, hyperparams_i, mask_i = prediction_results[
        i]
    y_pred[test_idx] = y_pred_i
    scores[test_idx] = y_scores_i
    feature_weights[f_idx] = np.asarray([np.nan for _ in range(len(fnames))])
    feature_weights[f_idx][mask_i] = weights_i
    for param in hyperparams_i:
      hyperparameters[param].append(hyperparams_i[param])

  acc = np.mean(y_pred == y)
  pvalue = stats.binom_test(sum(y_pred == y), n=len(y), alternative="greater")

  coef_df = pd.DataFrame(feature_weights, index=fnames)
  coef_df["mean_coef"] = coef_df.apply(np.nanmean, axis=1)
  coef_df["std_coef"] = coef_df.apply(np.nanstd, axis=1)
  return acc, coef_df[["mean_coef", "std_coef"
                      ]], scores, pd.DataFrame(hyperparameters), pvalue, chi2_df


def get_labeled_pairs(pairs_df):
  paired_labels = []
  c0s = []
  c1s = []
  page_ids = []
  for i, row in enumerate(pairs_df.itertuples()):
    if i % 2 == 0:
      c0s.append(row.conversation_id)
      c1s.append(row.bad_conversation_id)
    else:
      c0s.append(row.bad_conversation_id)
      c1s.append(row.conversation_id)
    paired_labels.append(i % 2)
    page_ids.append(row.page_id)
  return pd.DataFrame({
      "c0": c0s,
      "c1": c1s,
      "first_convo_toxic": paired_labels,
      "page_id": page_ids
  })


def get_feature_subset(feature_table, feature_list):
  prompt_type_names = ["km_%d_dist" % i for i in range(6)
                      ] + ["km_%d_dist_second" % i for i in range(6)]
  politeness_names = [
      f for f in feature_table.columns if f not in prompt_type_names
  ]

  features_to_use = []
  if "prompt_types" in feature_list:
    features_to_use += prompt_type_names
  if "politeness_strategies" in feature_list:
    features_to_use += politeness_names

  feature_subset = feature_table[features_to_use]

  return feature_subset, features_to_use


def run_pipeline(feature_table, feature_set):
  logging.info("Running prediction task for feature set", "+".join(feature_set))
  logging.info("Generating labels...")
  logging.info("Computing paired features...")
  X, feature_names = get_feature_subset(
      feature_table.drop(columns=["conversation_id", "slug", "label"], axis=1),
      feature_set)
  #X = X_c1 - X_c0
  logging.info("Using", X.shape[1], "features")
  #y = labeled_pairs_df.first_convo_toxic.values
  y_train = feature_table["label"]
  y = LabelEncoder().fit_transform(y_train)
  logging.info("Running leave-one-page-out prediction...")
  accuracy, coefs, scores, hyperparams, pvalue, chi2_df = run_pred(
      X, y, feature_names, feature_table.slug)  #, labeled_pairs_df.page_id)
  logging.info("Accuracy:", accuracy)
  logging.info("p-value: %.4e" % pvalue)
  logging.info("C (mode):", mode(hyperparams.logreg__C))
  logging.info("Percent of features (mode):", mode(hyperparams.featselect__percentile))
  logging.info("Coefficents:")
  coefs.join(chi2_df).sort_values(by="mean_coef").round(3).to_csv("_".join(feature_set) +
                                                         ".csv")
  return accuracy
