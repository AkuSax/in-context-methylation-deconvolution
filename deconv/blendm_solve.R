# Run vendored BLEND-M (vendor/BLENDM/BLENDM.R) on many tasks in one R session.
# Usage: Rscript blendm_solve.R <task_dir>
# <task_dir>/tasks.txt lists one subdirectory per line. Each task directory holds
#   ref.csv     markers x purified samples (first column = CpG id)
#   labels.csv  columns sample,type (order matches ref.csv columns)
#   bulk.csv    markers x bulk samples (same CpG order)
# and receives blendm_unknown.csv / blendm_nounknown.csv (samples x types), or a .err file.
args <- commandArgs(trailingOnly = TRUE)
root <- args[1]
here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
suppressPackageStartupMessages(library(nnls))
source(file.path(here, "vendor", "BLENDM", "BLENDM.R"))

for (task in readLines(file.path(root, "tasks.txt"))) {
  d <- file.path(root, task)
  ref <- as.matrix(read.csv(file.path(d, "ref.csv"), row.names = 1, check.names = FALSE))
  lab <- read.csv(file.path(d, "labels.csv"), stringsAsFactors = FALSE)
  bulk <- as.matrix(read.csv(file.path(d, "bulk.csv"), row.names = 1, check.names = FALSE))
  stopifnot(identical(colnames(ref), lab$sample), identical(rownames(ref), rownames(bulk)))
  types <- unique(lab$type)
  ref_list <- setNames(lapply(types, function(t) ref[, lab$type == t, drop = FALSE]), types)
  for (variant in c("unknown", "nounknown")) {
    out <- file.path(d, paste0("blendm_", variant, ".csv"))
    res <- tryCatch(BLENDM(bulk, ref_list, unknown.include = (variant == "unknown")),
                    error = function(e) e)
    if (inherits(res, "error")) {
      writeLines(conditionMessage(res), sub("\\.csv$", ".err", out))
    } else {
      rownames(res) <- colnames(bulk)
      write.csv(res, out)
    }
  }
}
