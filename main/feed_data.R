library(plyr)
library(readr)

# a csv containing BOTH Training and Testing DATA
# 4 columns: _id, text, training (True/False), label (True/False)
df <- read.csv("./src/data/both_t_data.csv")
df <- rename(df, c("X_id" = "id"))

system2("main/train_classifier_g",
        stdout = "test_out",
        input = format_csv(df))
