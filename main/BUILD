load("@rules_python//python:defs.bzl", "py_binary")
load("@deps//:requirements.bzl", "requirement")

py_binary(
     name = "get_feature_set",
     srcs = ["get_feature_set.py"],
)

py_binary(
    name = "train_classifier_g",
    srcs = ["train_classifier_g.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//main:get_feature_set",
        "//src:classifiers",
        "//src:download_data",
        "//src:receive_data",
        "//src:suite",
        requirement("pandas"),
    ],
)

py_binary(
    name ="train_prompt_types",
    srcs = ["train_prompt_types.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//src:predict_bad_conver_helpers",
        "//src:download_data",
        "//src:conversation_struct",
        "//src:receive_data",
        requirement("pandas"),
        requirement("markdown"),
        requirement("numpy"),
        requirement("scipy"),
        requirement("sklearn"),
        requirement("convokit"),
    ],
)

py_binary(
    name ="predict_bad_conver",
    srcs = ["predict_bad_conver.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//src:predict_bad_conver_helpers",
        "//src:download_data",
        "//src:conversation_struct",
        "//src:receive_data",
        requirement("pandas"),
        requirement("markdown"),
        requirement("numpy"),
        requirement("scipy"),
        requirement("sklearn"),
        requirement("convokit"),
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

py_binary(
    name = "train_polite_prompt_classifier",
    srcs = ["train_polite_prompt_classifier.py"],
    data = [
        "pt_model_10K.files",
    ],
    deps = [
        "//src:conversation_struct",
        "//src:download_data",
        "//src:predict_bad_conver_helpers",
        requirement("convokit"),
        requirement("numpy"),
        requirement("pandas"),
        requirement("sklearn"),
        requirement("spacy"),
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
    name = "feed_data",
    src = "feed_data.R",
    data = [
        ":train_classifier_g",
        ":train_polite_score",
        "//src:convo_word_freq_diff",
        "//src:find_SE_words",
        "//src:data/both_t_data_subset.csv",
        "//main:train_prompt_types",
        "//main:train_polite_prompt_classifier",
    ],
    deps = [
        ":politeness_logi",
        "@R_plyr",
        "@R_readr",
    ],
)
