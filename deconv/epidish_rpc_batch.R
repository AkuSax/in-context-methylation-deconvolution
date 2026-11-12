# Read one independent reference and one bulk profile per task.
suppressPackageStartupMessages(library(EpiDISH))
args <- commandArgs(trailingOnly = TRUE)
tasks <- as.integer(args[3])
markers <- as.integer(args[4])
types <- as.integer(args[5])
values <- readBin(args[1], "double", n = tasks * markers * (types + 1), endian = "little")
values <- array(values, dim = c(types + 1, markers, tasks))

probes <- sprintf("cg%08d", seq_len(markers))
labels <- paste0("type", seq_len(types))
estimates <- matrix(NA_real_, nrow = types, ncol = tasks)
for (task in seq_len(tasks)) {
  profiles <- t(matrix(values[, , task], nrow = types + 1))
  rownames(profiles) <- probes
  colnames(profiles) <- c(labels, "bulk")
  ref <- profiles[, labels, drop = FALSE]
  bulk <- profiles[, "bulk", drop = FALSE]
  estimates[, task] <- epidish(beta.m = bulk, ref.m = ref, method = "RPC")$estF[1, labels]
}
writeBin(as.double(estimates), args[2], endian = "little")
