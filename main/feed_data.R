library(plyr)
library(readr)

# the combined training data and test data used in the NIER2019 paper:
# https://cmustrudel.github.io/papers/raman20toxicity.pdf
df <- read.csv("./src/data/both_t_data.csv")
df <- rename(df, c("X_id" = "id"))

system2("main/test_model",
        stdout = "",#test_out",
        input = format_csv(df))
