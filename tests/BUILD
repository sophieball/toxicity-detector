load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_python//python:defs.bzl", "py_test")
load("@deps//:requirements.bzl", "requirement")

py_test(
    name = "test_GH_model",
    srcs = ["test_GH_model.py"],
    deps = [
        "//src:receive_data",
        "//main:apply_GH_model",
    ],
)

py_test(
    name = "test_receive_data",
    srcs = ["test_receive_data.py"],
    deps = ["//src:receive_data"],
)

py_test(
    name = "main_test",
    srcs = [
        "main_test.py",
        "test_receive_data.py",
    ],
    main = "main_test.py",
    deps = [
        "//src:receive_data",
        "//src:text_parser",
    ],
)
