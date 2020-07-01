library(plyr)
library(readr)

# the combined training data and test data used in the NIER2019 paper:
# https://cmustrudel.github.io/papers/raman20toxicity.pdf
df <- read.csv("src/data/both_t_data_subset.csv")
df <- rename(df, c("X_id" = "id"))

system2("main/train_classifier_g",
#system2("main/politeness_logi",
#system2("src/convo_word_freq_diff",
        stdout = "",
        input = format_csv(df))
