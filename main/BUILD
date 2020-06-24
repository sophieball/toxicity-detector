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
    name = "test_model",
    srcs = ["test_model.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//src:download_data",
        "//src:receive_data",
        "//src:suite",
    ],
    data = [
        "//src:pickles/SVM_model.p",
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
    name = "feed_data",
    src = "feed_data.R",
    data = [
        ":train_classifier_g",
        ":test_model",
        "//src:data/both_t_data.csv",
    ],
    deps = [
        "@R_plyr",
        "@R_readr",
    ],
)
