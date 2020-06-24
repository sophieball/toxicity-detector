library(plyr)
library(readr)

# the combined training data nad test data used in the NIER2019 paper
df <- read.csv("./src/data/both_t_data.csv")
df <- rename(df, c("X_id" = "id"))

system2("main/test_model",
        stdout = "",#test_out",
        input = format_csv(df))
