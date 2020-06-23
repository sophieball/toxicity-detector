library(plyr)
library(readr)

df <- data.frame(
  id = c(1,2,3,4,5),
  text = c("great","not great", "you jerk", "bananas", "orange jerks"),
  label = c(F, F, T, F, NA),
  training = c(T, T, T, T, F)
)
getwd()
df <- read.csv("./src/data/both_t_data.csv")
#df <- read.csv("~/Desktop/example_data.csv")
df <- rename(df, c("X_id" = "id"))

system2("../../main/train_classifier_g",
        stdout = "test_out",
        input = format_csv(df))
