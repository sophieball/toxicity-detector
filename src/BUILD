load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_python//python:defs.bzl", "py_test")
load("@deps//:requirements.bzl", "requirement")

py_library(
    name = "download_data",
    srcs = ["download_data.py"]
)

py_library(
    name = "config",
    srcs = ["config.py"]
)

py_library(
    name = "perspective",
    srcs = ["perspective.py"],
    deps = [
        ":config",
        requirement("requests"),
        requirement("markdown"),
    ]
)

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        requirement("pandas"),
        ":get_data",
        requirement("sklearn"),
        requirement("imblearn")
    ]
)

py_library(
    name = "lexicon",
    srcs = ["lexicon.py"],
    deps = [
        requirement("nltk"),
        requirement("pandas"),
        ":text_modifier"
    ]
)

py_library(
    name = "text_modifier",
    srcs = ["text_modifier.py"],
    deps = [
        requirement("sklearn"),
        requirement("nltk"),
        requirement("pandas"),
        requirement("spacy"),
        requirement("numpy"),
        requirement("en_core_web_md")
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "convo_politeness",
    srcs = ["convo_politeness.py"],
    deps = [
        requirement("convokit"),
        requirement("pandas"),
        requirement("spacy"),
        requirement("en_core_web_sm"),
        requirement("pytz"),
        requirement("cycler"),
        requirement("kiwisolver"),
        requirement("thinc"),
        requirement("catalogue"),
        requirement("importlib_metadata"),
        requirement("zipp"),
        requirement("srsly"),
        requirement("cymem"),
        requirement("preshed"),
        requirement("murmurhash"),
        requirement("blis"),
        requirement("wasabi"),
        requirement("plac"),
        requirement("ftfy"),
        requirement("threadpoolctl"),
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "create_features",
    srcs = ["create_features.py"],
    deps = [
        ":get_data",
        ":convo_politeness",
        ":text_parser",
        ":text_cleaning",
        ":text_modifier",
        ":util",
        requirement("pandas"),
        requirement("nltk")
    ]
)

py_library(
    name = "text_cleaning",
    srcs = ["text_cleaning.py"],
    deps = [
        requirement("nltk"),
        requirement("markdown")
    ]
)

py_library(
    name = "text_parser",
    srcs = ["text_parser.py"],
    deps = [
        requirement("nltk"),
        requirement("markdown")
    ]
)

py_library(
    name = "cross_validate",
    srcs = ["cross_validate.py"],
    deps = [
        requirement("sklearn"),
        ":get_data",
        ":classifiers",
        ":text_modifier"
    ]
)

py_library(
    name = "get_data",
    srcs = ["get_data.py"],
    deps = [
        requirement("pandas")
    ]
)

py_library(
    name = "suite",
    srcs = ["suite.py"],
    deps = [
        ":text_modifier",
        ":convo_politeness",
        ":create_features",
        ":util",
        ":cross_validate",
        ":classifiers",
        ":perspective",
        requirement("nltk"),
        requirement("pandas"),
        requirement("textblob"),
        requirement("sklearn"),
        requirement("gensim"),
        requirement("smart_open"),
        requirement("wordfreq")
    ]
)

py_library(
    name = "classifiers",
    srcs = ["classifiers.py"],
    deps = [requirement("sklearn")]
)

# in progress
py_test(
    name = "test_requirements",
    srcs = ["test_requirements.py"]
)

py_binary(
    name = "train_classifier_g",
    srcs = ["train_classifier_g.py"],
    deps = [
        ":download_data",
        ":suite",
        ":classifiers",
        requirement("pandas")
    ],
)
