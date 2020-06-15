load("@rules_python//python:defs.bzl", "py_binary")
load("@deps//:requirements.bzl", "requirement")

py_binary(
    name = "train_classifier_g",
    srcs = ["train_classifier_g.py"],
    deps = [
        "//src:receive_data",
        "//src:download_data",
        "//src:suite",
        "//src:classifiers",
        requirement("pandas")
    ],
)

