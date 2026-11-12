# Usage: Rscript epidish_solve.R ref.csv bulk.csv METHOD CONSTRAINT out.csv
# ref: markers x types, bulk: markers x samples (first column = probe id).
suppressPackageStartupMessages(library(EpiDISH))
a <- commandArgs(trailingOnly = TRUE)
ref <- as.matrix(read.csv(a[1], row.names = 1, check.names = FALSE))
bulk <- as.matrix(read.csv(a[2], row.names = 1, check.names = FALSE))
method <- a[3]
if (method == "CP") {
  est <- epidish(beta.m = bulk, ref.m = ref, method = "CP", constraint = a[4])$estF
} else {
  est <- epidish(beta.m = bulk, ref.m = ref, method = method)$estF
}
write.csv(est, a[5])
