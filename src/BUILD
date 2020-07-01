load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_python//python:defs.bzl", "py_test")
load("@deps//:requirements.bzl", "requirement")

exports_files(
    glob([
        "data/*.csv",
        "pickles/*.p",
    ]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "receive_data",
    srcs = ["receive_data.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "download_data",
    srcs = ["download_data.py"],
    deps = [
        requirement("numpy"),
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "statistics",
    srcs = ["statistics.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("pandas"),
        requirement("sklearn"),
        requirement("imblearn"),
    ],
)

py_library(
    name = "text_modifier",
    srcs = ["text_modifier.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":download_data",
        requirement("sklearn"),
        requirement("nltk"),
        requirement("pandas"),
        requirement("spacy"),
        requirement("numpy"),
    ],
)

py_binary(
    name = "convo_word_freq_diff",
    srcs = ["convo_word_freq_diff.py"],
    visibility = ["//visibility:public"],
    data = ["data/both_t_data.csv"],
    deps = [
        ":download_data",
        ":receive_data",
        requirement("blis"),
        requirement("catalogue"),
        requirement("convokit"),
        requirement("cycler"),
        requirement("cymem"),
        requirement("ftfy"),
        requirement("importlib_metadata"),
        requirement("kiwisolver"),
        requirement("markdown"),
        requirement("murmurhash"),
        requirement("numpy"),
        requirement("pandas"),
        requirement("plac"),
        requirement("preshed"),
        requirement("python-dateutil"),
        requirement("pytz"),
        requirement("spacy"),
        requirement("srsly"),
        requirement("thinc"),
        requirement("threadpoolctl"),
        requirement("wasabi"),
        requirement("zipp"),
    ],
)

py_binary(
    name = "convo_politeness",
    srcs = ["convo_politeness.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":download_data",
        ":receive_data",
        requirement("blis"),
        requirement("catalogue"),
        requirement("convokit"),
        requirement("cycler"),
        requirement("cymem"),
        requirement("ftfy"),
        requirement("importlib_metadata"),
        requirement("kiwisolver"),
        requirement("murmurhash"),
        requirement("numpy"),
        requirement("pandas"),
        requirement("plac"),
        requirement("preshed"),
        requirement("pytz"),
        requirement("spacy"),
        requirement("srsly"),
        requirement("thinc"),
        requirement("threadpoolctl"),
        requirement("wasabi"),
        requirement("zipp"),
    ],
)

py_library(
    name = "create_features",
    srcs = ["create_features.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":convo_politeness",
        ":text_cleaning",
        ":text_modifier",
        ":text_parser",
        ":util",
        requirement("pandas"),
        requirement("nltk"),
    ],
)

py_library(
    name = "text_cleaning",
    srcs = ["text_cleaning.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("nltk"),
        requirement("markdown"),
    ],
)

py_library(
    name = "text_parser",
    srcs = ["text_parser.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("nltk"),
        requirement("markdown"),
        requirement("importlib_metadata"),
        requirement("zipp"),
    ],
)

py_library(
    name = "cross_validate",
    srcs = ["cross_validate.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("sklearn"),
        ":classifiers",
        ":text_modifier",
    ],
)

py_library(
    name = "suite",
    srcs = ["suite.py"],
    data = glob([
        "data/*.csv",
        "pickles/*.p",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":classifiers",
        ":convo_politeness",
        ":create_features",
        ":cross_validate",
        ":text_modifier",
        ":util",
        requirement("gensim"),
        requirement("nltk"),
        requirement("pandas"),
        requirement("sklearn"),
        requirement("smart_open"),
        requirement("textblob"),
        requirement("wordfreq"),
    ],
)

py_library(
    name = "classifiers",
    srcs = ["classifiers.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("sklearn"),
        ":statistics",
    ],
)
