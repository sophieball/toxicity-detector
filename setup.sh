#!/bin/bash
bazel build //src:train_classifier_g

# check if spacy en is linked
SPACY_PATH=bazel-bin/src/train_classifier_g.runfiles/deps_pypi__spacy_2_2_4/spacy/data/

FILE="$SPACY_PATH""en"
if [ ! -d "$FILE" ]; then
    cd "$SPACY_PATH"
    ln -s ../../../deps_pypi__en_core_web_sm_2_2_0/en_core_web_sm/ ./en
    cd -
else
    echo
fi

echo "Done building. To run:"
echo "bazel-bin/src/train_classifier_g"

