# Usage: Rscript noob_idats.R IDAT_DIR OUT_PREFIX [DETP_MAX]
#
# One preprocessing recipe for every series, starting from raw IDATs:
#   minfi preprocessNoob (background subtraction + single-sample dye-bias
#   correction), beta = M / (M + U), and beta set to NA where the detection
#   p-value exceeds DETP_MAX (default 0.01).
# Every step works on one array at a time, so a sample's betas do not depend on
# which other samples sit in the directory.
#
# Writes
#   OUT_PREFIX.f32.bin      float32, column-major, probes x samples
#   OUT_PREFIX.probes.txt   one probe id per line
#   OUT_PREFIX.samples.txt  one IDAT basename per line (GSM_sentrix_position)
#   OUT_PREFIX.qc.csv       per-sample array type, detection failures, median signal
suppressPackageStartupMessages(library(minfi))

a <- commandArgs(trailingOnly = TRUE)
if (length(a) < 2) stop("usage: Rscript noob_idats.R IDAT_DIR OUT_PREFIX [DETP_MAX]")
dir <- a[1]
out <- a[2]
detp_max <- if (length(a) >= 3) as.numeric(a[3]) else 0.01

# minfi 1.44 reaches matrixStats with useNames = NA, which the installed
# matrixStats rejects. For this session only, turn that NA into TRUE (names are
# kept, values are unchanged) so preprocessNoob runs with the packages as they are.
local({
  ns <- asNamespace("matrixStats")
  for (f in getNamespaceExports(ns)) {
    orig <- get(f, ns)
    if (!is.function(orig) || !"useNames" %in% names(formals(orig))) next
    wrapped <- local({
      inner <- orig
      function(...) {
        args <- list(...)
        if (!is.null(args$useNames) && is.na(args$useNames)) args$useNames <- TRUE
        do.call(inner, args)
      }
    })
    assignInNamespace(f, wrapped, ns)
  }
})

# minfi::detectionP(type = "m+u") written out with base R: total signal against
# the negative controls.
detection_p <- function(rg) {
  neg <- getControlAddress(rg, controlType = "NEGATIVE")
  r <- getRed(rg)
  g <- getGreen(rg)
  r_mu <- apply(r[neg, , drop = FALSE], 2, median)
  r_sd <- apply(r[neg, , drop = FALSE], 2, mad)
  g_mu <- apply(g[neg, , drop = FALSE], 2, median)
  g_sd <- apply(g[neg, , drop = FALSE], 2, mad)
  one_red <- getProbeInfo(rg, type = "I-Red")
  one_grn <- getProbeInfo(rg, type = "I-Green")
  two <- getProbeInfo(rg, type = "II")
  p <- matrix(NA_real_, nrow = nrow(one_red) + nrow(one_grn) + nrow(two), ncol = ncol(rg),
              dimnames = list(c(one_red$Name, one_grn$Name, two$Name), sampleNames(rg)))
  for (i in seq_len(ncol(rg))) {
    p[one_red$Name, i] <- pnorm(r[one_red$AddressA, i] + r[one_red$AddressB, i],
                                mean = 2 * r_mu[i], sd = 2 * r_sd[i], lower.tail = FALSE)
    p[one_grn$Name, i] <- pnorm(g[one_grn$AddressA, i] + g[one_grn$AddressB, i],
                                mean = 2 * g_mu[i], sd = 2 * g_sd[i], lower.tail = FALSE)
    p[two$Name, i] <- pnorm(r[two$AddressA, i] + g[two$AddressA, i],
                            mean = r_mu[i] + g_mu[i], sd = r_sd[i] + g_sd[i], lower.tail = FALSE)
  }
  p
}

grn <- sort(list.files(dir, pattern = "_Grn\\.idat(\\.gz)?$", full.names = TRUE))
if (length(grn) == 0) stop("no *_Grn.idat files in ", dir)
base <- sub("_Grn\\.idat(\\.gz)?$", "", grn)

rg <- read.metharray(base, force = TRUE)
cat("array:", annotation(rg)[["array"]], " samples:", ncol(rg), "\n")

detp <- detection_p(rg)
ms <-preprocessNoob(rg, dyeCorr = TRUE, dyeMethod = "single")
beta <- getBeta(ms)
detp <- detp[rownames(beta), colnames(beta), drop = FALSE]
fail <- detp > detp_max
beta[fail] <- NA

con <- file(paste0(out, ".f32.bin"), "wb")
for (j in seq_len(ncol(beta))) writeBin(as.numeric(beta[, j]), con, size = 4)
close(con)
writeLines(rownames(beta), paste0(out, ".probes.txt"))
writeLines(colnames(beta), paste0(out, ".samples.txt"))

qc <- data.frame(
  sample = colnames(beta),
  array = annotation(rg)[["array"]],
  n_probes = nrow(beta),
  frac_detp_fail = colMeans(fail),
  median_meth = apply(getMeth(ms), 2, median),
  median_unmeth = apply(getUnmeth(ms), 2, median),
  minfi = as.character(packageVersion("minfi")),
  detp_max = detp_max
)
write.csv(qc, paste0(out, ".qc.csv"), row.names = FALSE)
cat("wrote", out, " probes:", nrow(beta), " masked:", sum(fail), "\n")
