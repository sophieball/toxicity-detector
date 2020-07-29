# Code by Lan Cheng
library(caret)

# redirect output to a file
sink("factor.out")

# Factor analysis to pushback metrics to 3-7 factors
dat <- read.csv("/dev/stdin")
dat$comb <- 2*dat$mot1 + 3*dat$mot2

# Check variance
print("initial dataframe shape:")
print(dim(dat))
no_var <- as.numeric(which(apply(dat, 2, var) == 0))
print("columns with no variance:")
print(no_var)
# Remove columns with 0 variance
if (length(no_var) > 0){
  dat <- dat[ - no_var]
}
print(paste("dataframe shape after droping no var cols:", dim(dat)))

linear_combos <- findLinearCombos(dat)
linear_combos
# remove columns that are linear combos
dat <- dat[, - as.numeric(linear_combos$remove)]
print("dataframe shape after dropping linear combination cols:")
print(dim(dat))


# Use elbow method to determine the optimal number of factors. Here the elbow
# point is determined by visual inspection
png(paste("scree", "_", toString(format(Sys.time(), format="%y-%m-%d_%H:%M:%S")), ".png", sep=""))
plot(eigen(cor(dat))$values)

for (numFactor in 3:7){
  fa.result <- factanal(dat,
                        factors = numFactor,
                        rotation = "varimax",
                        scores = "Bartlett"
  )
  fa.result.loading <- fa.result$loadings
  fa.result.scores <- data.frame(fa.result$scores)
  fa.result.scores$fingerprint <- dat$fingerprint
  print(paste("Trying ", numFactor, " factors:"))
  print(fa.result, digits=2, cutoff=0.3, sort=TRUE)
  cat("\n\n")

  # plot first 2 factors
  load <- fa.result$loadings[,1:2]
  png(paste("factors", numFactor, "_", toString(format(Sys.time(), format="%y-%m-%d_%H:%M:%S")), ".png", sep=""))
  plot(load, xlab="Factor 1", ylab="Factor 2",
       ylim=c(-1, 1), xlim=c(-1, 1), main="Varimax rotation")
  text(
       fa.result$loadings[,1]-0.08,
       fa.result$loadings[,2]+0.08,
       colnames(dat),
       col="blue"
     )
}

write("Done.", stderr())
write(paste("\nPlots are stored in path `",
            getwd(),
            "` with names
            `scree_<time>.png`,
            `factors3_<time>.png`,
            `factors4_<time>.png`,
            `factors5_<time>.png`,
            `factors6_<time>.png`",
            sep=""), stderr())
write(paste("\nResults are stored in `", getwd(), "/factor.out`",  sep=""), stderr())
