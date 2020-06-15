load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_python//python:defs.bzl", "py_test")
load("@deps//:requirements.bzl", "requirement")

py_test(
    name = "main_test",
    srcs = ["main_test.py"],
    main = "main_test.py",
    deps = [
        "//src:text_parser",
        "//src:text_modifier",
        "//src:text_cleaning",
    ]
)
