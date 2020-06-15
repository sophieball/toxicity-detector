load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_python//python:defs.bzl", "py_test")
load("@deps//:requirements.bzl", "requirement")

py_library(
    name = "receive_data",
    srcs = ["receive_data.py"],
    visibility = ["//visibility:public"]
)

py_library(
    name = "download_data",
    srcs = ["download_data.py"],
    visibility = ["//visibility:public"]
)

py_library(
    name = "statistics",
    srcs = ["statistics.py"],
    visibility = ["//visibility:public"]
)

py_library(
    name = "config",
    srcs = ["config.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        requirement("pandas"),
        requirement("sklearn"),
        requirement("imblearn"),
    ],
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
    ],
    visibility = ["//visibility:public"]
)

py_library(
    name = "convo_politeness",
    srcs = ["convo_politeness.py"],
    deps = [
        requirement("convokit"),
        requirement("pandas"),
        requirement("spacy"),
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
)

py_library(
    name = "create_features",
    srcs = ["create_features.py"],
    deps = [
        ":config",
        ":convo_politeness",
        ":text_parser",
        ":text_cleaning",
        ":text_modifier",
        ":util",
        requirement("pandas"),
        requirement("nltk")
    ],
)

py_library(
    name = "text_cleaning",
    srcs = ["text_cleaning.py"],
    deps = [
        requirement("nltk"),
        requirement("markdown")
    ],
    visibility = ["//visibility:public"]
)

py_library(
    name = "text_parser",
    srcs = ["text_parser.py"],
    deps = [
        requirement("nltk"),
        requirement("markdown"),
        requirement("importlib_metadata"),
        requirement("zipp")
    ],
    visibility = ["//visibility:public"]
)

py_library(
    name = "cross_validate",
    srcs = ["cross_validate.py"],
    deps = [
        requirement("sklearn"),
        ":classifiers",
        ":text_modifier"
    ],
)

py_library(
    name = "suite",
    srcs = ["suite.py"],
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
        requirement("wordfreq")
    ],
    data = glob([
        "data/*.csv",
        "pickles/*.p"
    ]),
    visibility = ["//visibility:public"]
)

py_library(
    name = "classifiers",
    srcs = ["classifiers.py"],
    deps = [
        requirement("sklearn"),
        ":statistics"
    ],
    visibility = ["//visibility:public"]
)

py_test(
    name = "test_receive_data",
    srcs = ["test_receive_data.py"],
    main = "test_receive_data.py",
    deps = [
        ":receive_data",
        requirement("pandas")
    ]
)
