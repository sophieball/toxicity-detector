library(plyr)
library(readr)

# the combined training data and test data used in the NIER2019 paper:
# https://cmustrudel.github.io/papers/raman20toxicity.pdf

# Load data
#df <- read.csv("src/data/issues_matched_by_length.csv")
#df <- read.csv("src/data/prs_matched_by_length.csv")
#df <- read.csv("src/data/too_heated_prs_matched_by_length.csv")
#df <- read.csv("src/data/pr_body_comments.csv")
#df <- read.csv("src/data/fake_G_pushback.csv")
#df <- read.csv("src/data/pushback_survey_prs_apr_8.csv")
#df <- read.csv("src/data/pushback_survey_prs_sampled.csv")
df <- read.csv("src/data/pushback_reported_prs_w_neg.csv")
#df <- read.csv("src/data/issues_prs.csv")
#df <- read.csv("src/data/random_sample_10000_prs_body_comments.csv")

df <- rename(df, c("X_id" = "id"))

# I leave them as comment here so I can easily switch
#system2("main/train_prompt_types",
#system2("main/train_polite_prompt_classifier",
system2("main/train_classifier_g",
#system2("src/find_SE_words",
#system2("src/convo_politeness",
#system2("main/politeness_logi",
#system2("src/convo_word_freq_diff",
        args = "prs",
        #args = "issues",
        stdout = "",
        input = format_csv(df))
