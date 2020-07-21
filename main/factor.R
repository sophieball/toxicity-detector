# redirect output to a file
sink("factor.out")

library(readr)
library(nFactors)
library(DataCombine)

# get data
dat <- read.csv("/dev/stdin")
write(paste("Number of rows in the data:", nrow(dat)), stdout())
dat <- DropNA(dat)
write(paste("Number of rows after droping NAs:", nrow(dat)), stdout())
write(paste("Number of variables for factor analysis:", ncol(dat)), stdout())

# determine the number of factors
ev <- eigen(cor(dat, method="spearman", use="complete.obs"))
write(paste("Eigen values:", ev$values), stdout())

ap <- parallel(subject=nrow(dat), var=ncol(dat), rep=100, cent=0.05)
nS <- nScree(x=ev$values, aparallel=ap$eigen$qevpea)
png(paste("scree", "_", toString(format(Sys.time(), format="%y-%m-%d_%H:%M:%S")), ".png", sep=""))
plotnScree(nS)

# try different numbers of factors
for (n_fac in 3:6){
  fit <- factanal(dat, n_fac, rotation="varimax")
  print(fit, digits=2, cutoff=0.3, sort=TRUE)
  # plot first 2 factors
  load <- fit$loadings[,1:2]
  png(paste("factors", n_fac, "_", toString(format(Sys.time(), format="%y-%m-%d_%H:%M:%S")), ".png", sep=""))
  plot(load, xlab="Factor 1", ylab="Factor 2",
       ylim=c(-1, 1), xlim=c(-1, 1), main="Varimax rotation")
  text(
       fit$loadings[,1]-0.08,
       fit$loadings[,2]+0.08,
       colnames(dat),
       col="blue"
     )
}

write(paste("Done. Plots are stored in path `",
            getwd(),
            "` with names `scree_<time>.pdf`, `factors3_<time>.png`, `factors4_<time>.png`, `factors5_<time>.png`, `factors6_<time>.png`",
            sep=""), stderr())
write(paste("Results are stored in `", getwd(), "/factor.out`",  sep=""), stderr())
