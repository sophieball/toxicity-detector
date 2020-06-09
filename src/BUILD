load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_python//python:defs.bzl", "py_test")
#load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@my_deps//:requirements.bzl", "requirement")
#load("@py_model//:requirements.bzl", requirement_model = "requirement")

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
        #requirement("en-core-web-sm-abd")
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "convo_politeness",
    srcs = ["convo_politeness.py"],
    deps = [
        requirement("convokit"),
        requirement("pandas"),
        #"@en_core_web_md"
        #requirement("en-core-web-sm-abd"),
        requirement("spacy"),
        #requirement_model("en_core_web_sm")
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "create_features",
    srcs = ["create_features.py"],
    deps = [
        ":get_data",
        ":convo_politeness",
        ":TextParser",
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
    name = "TextParser",
    srcs = ["TextParser.py"],
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
        requirement("gensim")
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
        ":suite",
        ":classifiers",
        requirement("pandas")
    ],
)
