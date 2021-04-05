library(plyr)
library(readr)

# the combined training data and test data used in the NIER2019 paper:
# https://cmustrudel.github.io/papers/raman20toxicity.pdf

# Load data
#df <- read.csv("src/data/both_t_data_thread.csv")
df <- read.csv("src/data/pr_body_comments.csv")

df <- rename(df, c("X_id" = "id"))

# I leave them as comment here so I can easily switch
system2("main/train_classifier_g",
#system2("src/find_SE_words",
#system2("main/politeness_logi",
#system2("src/convo_word_freq_diff",
        args = "prs",
        stdout = "",
        input = format_csv(df))
