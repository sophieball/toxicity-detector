load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_python//python:defs.bzl", "py_test")
load("@deps//:requirements.bzl", "requirement")

py_test(
    name = "main_test",
    srcs = ["main_test.py",
            "test_receive_data.py"],
    main = "main_test.py",
    deps = [
        "//src:receive_data",
        "//src:text_parser",
    ]
)
