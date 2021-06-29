load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_python//python:defs.bzl", "py_test")
load("@deps//:requirements.bzl", "requirement")

exports_files(
    glob([
        "data/*.csv",
        "pickles/*.p",
        "pickles/pt_model_10K.files/*",
        "model/*",
        "vocabulary/*",
        "Senti4SD-fast.jar",
        "dsm.bin",
        "Senti4SD.model",
    ]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "sentiment_classification",
    srcs = ["sentiment_classification.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/senti_core:classification",
        "//src/senti_core/utils:csv_utils",
        "//src/senti_core/utils:core_utils",
    ],
)

py_library(
    name = "receive_data",
    srcs = ["receive_data.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("pandas"),
        requirement("numpy"),
        requirement("pytz"),
        requirement("python-dateutil"),
    ],
)

py_library(
    name = "download_data",
    srcs = ["download_data.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("numpy"),
        requirement("setuptools"),
    ],
)

py_library(
    name = "fighting_words",
    srcs = ["fighting_words_py3.py"],
    deps = [
        ":receive_data",
        requirement("numpy"),
        requirement("sklearn"),
    ],
)

py_binary(
    name = "find_SE_words",
    srcs = ["find_SE_words.py"],
    data = [
        "data/kaggle_toxicity_subset.csv",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":fighting_words",
        requirement("langdetect"),
        requirement("pandas"),
        requirement("scipy"),
        requirement("joblib"),
    ],
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
        requirement("numpy"),
    ],
)

py_library(
    name = "plot_politeness",
    srcs = ["plot_politeness.py"],
    deps = [
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("pandas"),
    ],
)

py_library(
    name = "sep_ngram",
    srcs = ["sep_ngram.py"],
    deps = [requirement("pandas")],
)

py_binary(
    name = "convo_word_freq_diff",
    srcs = ["convo_word_freq_diff.py"],
    data = ["data/both_t_data.csv"],
    visibility = ["//visibility:public"],
    deps = [
        ":download_data",
        ":receive_data",
        ":text_parser",
        ":sep_ngram",
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
        requirement("matplotlib"),
        requirement("Pillow"),
        requirement("numpy"),
        requirement("pandas"),
        requirement("plac"),
        requirement("preshed"),
        requirement("python-dateutil"),
        requirement("pytz"),
        requirement("sklearn"),
        requirement("srsly"),
        requirement("thinc"),
        requirement("threadpoolctl"),
        requirement("wasabi"),
        requirement("zipp"),
    ],
)

py_binary(
    name = "conversation_struct_g",
    srcs = ["conversation_struct_g.py"],
    data = [
        "data/pr_body_comments.csv",
        "data/speakers_bots_full.list",
    ],
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
        requirement("markdown"),
        requirement("matplotlib"),
        requirement("Pillow"),
        requirement("murmurhash"),
        requirement("numpy"),
        requirement("pandas"),
        requirement("plac"),
        requirement("preshed"),
        requirement("python-dateutil"),
        requirement("pytz"),
        requirement("srsly"),
        requirement("thinc"),
        requirement("threadpoolctl"),
        requirement("wasabi"),
        requirement("zipp"),
    ],
)

py_binary(
    name = "conversation_struct",
    srcs = ["conversation_struct.py"],
    data = [
        "data/pr_body_comments.csv",
        "data/speakers_bots_full.list",
    ],
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
        requirement("markdown"),
        requirement("matplotlib"),
        requirement("Pillow"),
        requirement("murmurhash"),
        requirement("numpy"),
        requirement("pandas"),
        requirement("plac"),
        requirement("preshed"),
        requirement("python-dateutil"),
        requirement("pytz"),
        requirement("srsly"),
        requirement("thinc"),
        requirement("threadpoolctl"),
        requirement("wasabi"),
        requirement("zipp"),
    ],
)

py_library(
    name ="predict_bad_conver_helpers",
    srcs = ["predict_bad_conver_helpers.py"],
    data = [
        "data/pr_body_comments.csv",
        "data/both_t_data.csv",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":download_data",
        ":conversation_struct",
        requirement("pandas"),
        requirement("numpy"),
        requirement("scipy"),
        requirement("sklearn"),
        requirement("convokit"),
    ],
)

py_binary(
    name ="convo_forecast",
    srcs = ["convo_forecast.py"],
    data = ["data/pr_body_comments.csv"],
    visibility = ["//visibility:public"],
    deps = [
        ":download_data",
        ":conversation_struct",
        requirement("pandas"),
        requirement("numpy"),
        requirement("scipy"),
        requirement("sklearn"),
        requirement("convokit"),
    ],
)

py_binary(
    name = "convo_politeness",
    srcs = ["convo_politeness.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":download_data",
        ":receive_data",
        ":text_parser",
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
    data = [
        ":Senti4SD-fast.jar",
    ],
    deps = [
        ":config",
        ":convo_politeness",
        ":sentiment_classification",
        ":text_parser",
        requirement("pandas"),
        requirement("nltk"),
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
        "data/SE_words_G.list",
        "pickles/*.p",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":classifiers",
        ":convo_politeness",
        ":create_features",
        ":cross_validate",
        ":text_modifier",
        ":text_parser",
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
