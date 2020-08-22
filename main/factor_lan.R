#https://journal.r-project.org/archive/2016/RJ-2016-021/RJ-2016-021.pdf

# Code by Lan Cheng
library(mclust)
library(caret)

# redirect output to a file
sink("factor.out")

# Factor analysis to pushback metrics to 3-7 factors
dat <- read.csv("/dev/stdin")
dat[is.na(dat)] <- 0

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
print("dataframe shape after droping no var cols:")
print(dim(dat))

# find linear combinations in dat
#linear_combos <- findLinearCombos(dat)
#cat("\n\n")
#print("Columns that are linear combos:")
#print(linear_combos)

# remove columns that are linear combos
#if (length(linear_combos$remove) > 0){
#  dat <- dat[, - as.numeric(linear_combos$remove)]
#}
print("dataframe shape after checking linear combination cols:")
print(dim(dat))


# Use elbow method to determine the optimal number of factors. Here the elbow
# point is determined by visual inspection
png(paste("scree", "_", toString(format(Sys.time(), format="%y-%m-%d_%H:%M:%S")), ".png", sep=""))
#plot(eigen(cor(dat))$values)

for (numFactor in 3:4){
  fa.result <- factanal(dat,
                        factors = numFactor,
                        rotation = "varimax",
                        scores = "Bartlett"
  )
  fa.result.loading <- fa.result$loadings
  fa.result.scores <- data.frame(fa.result$scores)
  fa.result.scores$fingerprint <- dat$fingerprint
  cat("\n\n")
  print(paste("Trying ", numFactor, " factors:"))
  print(fa.result, digits=2, cutoff=0.3, sort=TRUE)

  # Clustering analysis using GMM model
  # Determine the optimal number of clusters
  dataModel <- fa.result.scores#[, -which(colnames(fa.result.scores) %in% c("fingerprint"))]
  # The range of number of clusters is set between 1 and 30. Results with more
  # than 30 clusters are not meaningful.
  gMin <- 1
  gMax <- 30
  # Calculate BIC curve, using VVV model
  set.seed(10086)
  BIC <- mclustBIC(dataModel, G = gMin:gMax, modelNames = "VVV")
  plot(BIC)
  # Use GMM clustering model with optimal number of clusters
  set.seed(10086)
  optimalMod <- Mclust(dataModel, x = BIC)
  print(summary(optimalMod))
  clusterResult <- dataModel
  print(clusterResult)
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
