#!/bin/bash
bazel build //main:train_classifier_g

echo "Done building. To run:"
echo "bazel-bin/main/train_classifier_g"

