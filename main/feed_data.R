library(plyr)
library(readr)

# the combined training data and test data used in the NIER2019 paper:
# https://cmustrudel.github.io/papers/raman20toxicity.pdf
df <- read.csv("src/data/both_t_data.csv")
df <- rename(df, c("X_id" = "id"))

#system2("main/train_polite_score",#train_classifier_g",#apply_GH_model",
system2("main/train_classifier_g",
        args = c("test", "src/pickles/SVM_pretrained_model.p"),
        stdout = "",
        input = format_csv(df))
