df <- data.frame(
  id = c(1,2,3,4,5),
  text = c("great","not great", "you jerk", "bananas", "orange jerks"),
  toxic = c(F, F, T, F, NA),
  training = c(T, T, T, T, F)
)

library(readr)
system2("bazel-bin/main/train_classifier_g",
        stdout = "test_out",
        input = format_csv(df))
