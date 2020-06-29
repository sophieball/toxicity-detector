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
        "//src:data/both_t_data.csv",
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
    name = "feed_data",
    src = "feed_data.R",
    data = [
        ":train_classifier_g",
        ":train_polite_score",
        "//src:convo_word_freq_diff",
        "//src:data/both_t_data.csv",
    ],
    deps = [
        ":politeness_logi",
        "@R_plyr",
        "@R_readr",
    ],
)
