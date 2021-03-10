# redirect output to a file
sink("politeness_logi.out")

library(pscl)
library(testit)
library(plyr)
library(readr)

# get data
dat <- read.csv("/dev/stdin")
assert(length(dat) > 0)
# collect counts of marked politness words and store them in a file
system2("src/convo_politeness",
        input = format_csv(dat))

# read python's output from file
dat <- read.csv("politeness_features.csv", stringsAsFactors=FALSE)
dat[is.na(dat)] <- 0
sapply(dat, class)

m_pol <- glm(label ~
              log(1+length)
            + log(1 + HASHEDGE)
            + (Please > 0)
            + (Please_start > 0)
            + (Factuality > 0)
            + (Deference > 0)
            + (Gratitude > 0)
            + (Apologizing > 0)
            + log(1 + X1st_person_pl.)
            + log(1 + X1st_person)
            + log(1 + X1st_person_start)
            + log(1 + X2nd_person)
            + (X2nd_person_start > 0)
            + (Indirect_.greeting. > 0)
            + (Direct_question > 0)
            + log(1 + Direct_start)
            + log(1 + HASPOSITIVE)
            + log(1 + HASNEGATIVE)
            + (SUBJUNCTIVE > 0)
            + (INDICATIVE > 0)
            + log(1 + rounds)
            + log(1 + shepherd_time)
            + log(1 + review_time)
            , data = dat
            , family = "binomial")

summary(m_pol)
pR2(m_pol)
write(paste("Done. Results are stored in `", getwd(), "/politeness_logi.out`",  sep=""), stderr())
