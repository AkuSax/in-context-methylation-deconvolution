# Usage: Rscript paper_solvers.R ref.csv bulk.csv outdir METHOD[,METHOD...]
# ref: markers x types, bulk: markers x samples (first column = probe id).
# Writes outdir/<METHOD>.csv (samples x types) or outdir/<METHOD>.err.
#
# Settings copy the De Ridder et al. 2024 benchmark scripts
# (github.com/FunctionalEpigeneticsLab/DNAme-deconvolution-benchmarking,
# Scripts/IC_deconvolution_workflow_part2.r):
#   EMeth_*       emeth(eta = 0, aber = FALSE, init = "default", nu = 0, maxiter = 50);
#                 Normal: V = "c", family "normal"; Laplace: V = "c", family "laplace";
#                 Binom: V = "b", family "normal". Fractions sum to 1.
#   ICeDT         ICeDT(rhoConverge = 0.0015, maxIter_PP = 30); the "tumor" row (its
#                 unexplained component) is dropped, so cell types sum to <= 1.
#   FARDEEP       fardeep(ref, bulk)$abs.beta (absolute, not renormalised).
#   LTS           robustbase::ltsReg(y ~ types, alpha = 0.5) with intercept; slopes kept.
#                 This is what the paper ran under the name "MethylResolver" (the
#                 package's own alpha search and normalisation were stripped).
a <- commandArgs(trailingOnly = TRUE)
ref <- as.matrix(read.csv(a[1], row.names = 1, check.names = FALSE))
bulk <- as.matrix(read.csv(a[2], row.names = 1, check.names = FALSE))
outdir <- a[3]
methods <- strsplit(a[4], ",")[[1]]
bulk <- bulk[rownames(ref), , drop = FALSE]

emeth_fit <- function(V, family) {
  suppressPackageStartupMessages(library(EMeth))
  capture.output(r <- emeth(Y = bulk, eta = rep(0, ncol(bulk)), mu = ref, aber = FALSE,
                            V = V, init = "default", family = family, nu = 0, maxiter = 50,
                            verbose = FALSE)$rho)
  r
}

lts_fit <- function() {
  types <- colnames(ref)
  sig <- as.data.frame(ref)
  names(sig) <- make.names(types)
  f <- as.formula(paste("y ~", paste(names(sig), collapse = " + ")))
  t(sapply(seq_len(ncol(bulk)), function(j) {
    set.seed(20261111 + j)
    d <- cbind(sig, y = bulk[, j])
    robustbase::ltsReg(f, data = d, alpha = 0.5)$coefficients[-1]
  }))
}

fit <- function(m) {
  switch(m,
    EMeth_Normal = emeth_fit("c", "normal"),
    EMeth_Laplace = emeth_fit("c", "laplace"),
    EMeth_Binom = emeth_fit("b", "normal"),
    ICeDT = {
      suppressPackageStartupMessages(library(ICeDT))
      capture.output(rho <- ICeDT(bulk, ref, rhoConverge = 0.0015, maxIter_PP = 30)$rho)
      t(rho[colnames(ref), , drop = FALSE])
    },
    FARDEEP = {
      suppressPackageStartupMessages(library(FARDEEP))
      as.matrix(fardeep(ref, bulk)$abs.beta)
    },
    LTS = lts_fit(),
    stop("unknown method ", m))
}

for (m in methods) {
  res <- tryCatch(fit(m), error = function(e) e)
  if (inherits(res, "error")) {
    writeLines(conditionMessage(res), file.path(outdir, paste0(m, ".err")))
    next
  }
  res <- matrix(as.numeric(res), nrow = ncol(bulk), dimnames = list(colnames(bulk), colnames(ref)))
  write.csv(res, file.path(outdir, paste0(m, ".csv")))
}
