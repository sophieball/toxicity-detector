load("@rules_python//python:defs.bzl", "py_binary")
load("@deps//:requirements.bzl", "requirement")

py_binary(
    name = "train_classifier_g",
    srcs = ["train_classifier_g.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//src:classifiers",
        "//src:download_data",
        "//src:receive_data",
        "//src:suite",
        requirement("pandas"),
    ],
)

py_binary(
    name = "train_polite_score",
    srcs = ["train_polite_score.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//src:download_data",
        "//src:convo_politeness",
        "//src:receive_data",
        requirement("sklearn"),
    ],
)

load(
    "@com_grail_rules_r//R:defs.bzl",
    "r_binary",
    "r_library",
    "r_pkg",
    "r_pkg_test",
    "r_unit_test",
)
load("@com_grail_rules_r//R:defs.bzl", "r_package")
load("@com_grail_rules_r//R:defs.bzl", "r_package_with_test")
load("@r_repositories_bzl//:r_repositories.bzl", "r_repositories")

r_binary(
    name = "politeness_logi",
    src = "politeness_logi.R",
    data = [
        "//src:data/both_t_data_subset.csv",
        "//src:convo_politeness",
    ],
    deps = [
        "@R_plyr",
        "@R_readr",
        "@R_pscl",
        "@R_testit",
    ],
)

r_binary(
    name = "factor",
    src = "factor.R",
    deps = [
        "@R_readr",
        "@R_nFactors",
        "@R_DataCombine",
        "@R_psych",
    ],
)

r_binary(
    name = "factor_lan",
    src = "factor_lan.R",
    deps = [
        "@R_caret",
        "@R_foreach",
        "@R_withr",
        "@R_pROC",
        "@R_iterators",
        "@R_codetools",
        "@R_recipes",
        "@R_ggplot2",
        "@R_ipred",
        "@R_ModelMetrics",
        "@R_reshape2",
        "@R_scales",
        "@R_stringr",
        "@R_viridisLite",
        "@R_isoband",
        "@R_prodlim",
        "@R_RColorBrewer",
        "@R_labeling",
        "@R_gtable",
        "@R_farver",
        "@R_testthat",
        "@R_munsell",
        "@R_evaluate",
        "@R_KernSmooth",
        "@R_lava",
        "@R_colorspace",
        "@R_survival",
        "@R_praise",
        "@R_SQUAREM",
        "@R_class",
        "@R_numDeriv",
        "@R_pkgload",
        "@R_pkgbuild",
        "@R_rstudioapi",
        "@R_rprojroot",
        "@R_desc",
        "@R_rpart",
        "@R_backports",
        "@R_callr",
        "@R_processx",
        "@R_ps",
        "@R_tidyr",
        "@R_gower",
        "@R_lubridate",
        "@R_timeDate",
    ]
)

r_binary(
    name = "feed_data",
    src = "feed_data.R",
    data = [
        ":factor",
        "factor_lan.R",
        ":factor_lan",
        ":train_classifier_g",
        ":train_polite_score",
        "//src:convo_word_freq_diff",
        "//src:data/both_t_data_subset.csv",
    ],
    deps = [
        ":politeness_logi",
        "@R_plyr",
        "@R_readr",
    ],
)
