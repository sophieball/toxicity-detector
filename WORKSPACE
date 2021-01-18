############
# For python
############
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "rules_python",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.1.0/rules_python-0.1.0.tar.gz",
    sha256 = "b6d46438523a3ec0f3cead544190ee13223a52f6a6765a29eae7b7cc24cc83a0",
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

# Only needed if using the packaging rules.
load("@rules_python//python:pip.bzl", "pip_repositories")

pip_repositories()

load("@rules_python//python:pip.bzl", "pip_install")

# Create a central repo that knows about the dependencies needed for
# requirements.txt.
pip_install(
   name = "deps",
   requirements = "requirements.txt",
)

#######
# for R
#######
http_archive(
    name = "com_grail_rules_r",
    strip_prefix = "rules_r-master",
    urls = ["https://github.com/grailbio/rules_r/archive/master.tar.gz"],
)

load("@com_grail_rules_r//R:dependencies.bzl", "r_register_toolchains", "r_rules_dependencies")

r_rules_dependencies()

r_register_toolchains()

# R packages with standard sources.
load("@com_grail_rules_r//R:repositories.bzl", "r_repository", "r_repository_list")

r_repository_list(
    name = "r_repositories_bzl",
    package_list = "//:r_packages.csv",
    remote_repos = {
        "CREAB": "https://cloud.r-project.org",
    },
)

# source: https://github.com/grailbio/rules_r/issues/38
load("@r_repositories_bzl//:r_repositories.bzl", "r_repositories")

r_repositories()

load("@com_grail_rules_r//R:dependencies.bzl", "r_coverage_dependencies")

r_coverage_dependencies()
