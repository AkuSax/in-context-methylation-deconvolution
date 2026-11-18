# Vendored from https://github.com/Penghuihuang2000/BLENDM at 90546bc3e75bf4fb3be50c1b0336b7650241f074 (MIT, see LICENSE.md).
# Only change: rlist::list.cbind/list.rbind -> do.call(cbind/rbind, .) because rlist is not installed.
#' Select Marker CpG Sites for Cellular Deconvolution
#'
#' Identifies marker CpG sites whose DNA methylation (DNAm) levels are
#' cell-type-specific, using a two-step filtering procedure. This function is
#' intended to be applied to a reference panel of purified cell-type samples
#' before running \code{\link{BLENDM}}.
#'
#' @details
#' The selection proceeds in two steps:
#'
#' \strong{Step 1 — ANOVA on M-values.}
#' For each CpG, a one-way ANOVA is fitted with cell type as the grouping
#' factor and DNAm M-values (\eqn{\log_2(\beta / (1 - \beta))}) as the
#' response. CpGs with a Benjamini-Hochberg (BH) adjusted p-value below
#' \code{pval_cutoff} are retained as having statistically significant
#' cell-type variation.
#'
#' \strong{Step 2 — Effect size filter.}
#' Among the CpGs passing Step 1, the maximum pairwise difference between
#' cell-type mean beta values is normalized by the pooled standard deviation.
#' CpGs whose normalized maximum group difference exceeds \code{diff_cutoff} are
#' returned as marker CpGs.
#'
#'
#' @param beta_val A numeric matrix of DNAm beta values with CpG sites as rows
#'   and purified reference samples as columns.
#' @param cell_type A character or factor vector of cell type labels for the
#'   purified reference samples. Must have the same length as
#'   \code{ncol(beta_val)} and its order must match the column order of
#'   \code{beta_val}.
#' @param pval_cutoff Numeric scalar. BH-adjusted p-value threshold for the
#'   M-value ANOVA in Step 1. CpGs with adjusted p-value strictly below this
#'   threshold are retained. Default is \code{0.05}.
#' @param diff_cutoff Numeric scalar. Threshold for the normalized maximum
#'   inter-cell-type group difference in Step 2. CpGs with a normalized
#'   difference strictly greater than this threshold are returned as markers.
#'   Default is \code{10}.
#'
#' @return A character vector of CpG names (row names of \code{beta_val})
#'   that pass both filtering steps and are selected as marker CpGs.
#'
#'
#'
#' @export
select_marker_cpg <- function(beta_val, cell_type, pval_cutoff = 0.05, diff_cutoff = 10) {
  eps <- 1e-6
  beta_val <- pmin(pmax(beta_val, eps), 1 - eps)

  ## Internal helper: run one-way ANOVA, return p-value and residual SE
  ANOVA_site <- function(x, group) {
    df <- data.frame("ct" = as.factor(group), "methyl" = x)
    res_ANOVA <- summary(aov(methyl ~ ct, data = df))
    p_val <- res_ANOVA[[1]]$`Pr(>F)`[1]
    se    <- sqrt(res_ANOVA[[1]]$`Mean Sq`[2])
    return(c(p_val, se))
  }

  ## Pooled SD from beta-value ANOVA (used for normalization in Step 2)
  run.ANOVA <- t(apply(beta_val, 1, function(x) { ANOVA_site(x, cell_type) }))
  ANOVA.se  <- as.vector(run.ANOVA[, 2])
  names(ANOVA.se) <- rownames(beta_val)

  ## Normalize beta values by pooled SD, then compute max inter-group mean difference
  beta_val_normalized <- sweep(beta_val, 1, ANOVA.se, "/")
  max_normalized_mean_diff <- function(marker) {
    cat_mean <- tapply(beta_val_normalized[marker, ], cell_type, mean)
    max(cat_mean) - min(cat_mean)
  }

  ## Step 1: ANOVA on M-values with BH correction
  beta_val_Mval <- log2(beta_val / (1 - beta_val))
  run.mANOVA    <- t(apply(beta_val_Mval, 1, function(x) { ANOVA_site(x, cell_type) }))
  mANOVA.pval   <- as.vector(run.mANOVA[, 1])
  names(mANOVA.pval) <- rownames(beta_val_Mval)

  mANOVA.pval.BH    <- p.adjust(mANOVA.pval, method = "BH")
  marker.mANOVA_005 <- names(mANOVA.pval.BH)[mANOVA.pval.BH < pval_cutoff]

  ## Step 2: effect size filter on ANOVA-significant CpGs
  marker.diff_norm <- sapply(marker.mANOVA_005, max_normalized_mean_diff)
  marker.selected  <- names(marker.diff_norm)[marker.diff_norm > diff_cutoff]

  return(marker.selected)
}


#' Estimate Beta Distribution Dispersion Parameter for a Single CpG
#'
#' Given observed beta values and their fitted means for a single CpG site
#' across multiple samples, estimates the CpG-specific precision (dispersion)
#' parameter \eqn{\phi_g} of the beta distribution by maximum likelihood.
#'
#'
#' @param obs.data A numeric vector of observed beta values for a single CpG
#'   across all bulk samples. All values must be strictly between 0 and 1.
#' @param mean A numeric vector of fitted mean beta values for the same CpG,
#'   of the same length as \code{obs.data}. All values must be strictly between
#'   0 and 1.
#' @param lower Numeric scalar. Lower bound of the search interval for the
#'   dispersion parameter. Default is \code{1e-6}.
#' @param upper Numeric scalar. Initial upper bound of the search interval.
#'   This is automatically doubled until the score function changes sign,
#'   so it is safe to leave at the default of \code{1e3}.
#'
#' @return A single numeric value: the maximum likelihood estimate of the
#'   dispersion parameter \eqn{\phi_g} for the given CpG.
#'
#'
#'
#' @export
estimate_dispersion <- function(obs.data, mean, lower = 1e-6, upper = 1e3) {
  x <- obs.data
  stopifnot(
    length(x) == length(mean),
    all(x    > 0 & x    < 1),
    all(mean > 0 & mean < 1)
  )
  G <- length(x)

  ## Score function (first derivative of log-likelihood w.r.t. dispersion)
  score <- function(dispersion) {
    G * digamma(dispersion) -
      sum(mean       * digamma(mean       * dispersion) +
          (1 - mean) * digamma((1 - mean) * dispersion)) +
      sum(mean * log(x) + (1 - mean) * log(1 - x))
  }

  ## Expand upper bound until root is bracketed
  s_low <- score(lower)
  s_up  <- score(upper)
  while (s_low * s_up > 0) {
    upper <- 2 * upper
    s_up  <- score(upper)
  }

  res <- uniroot(score, lower = lower, upper = upper)
  return(res$root)
}


#' BLEND-M: Cellular Deconvolution with Personalized DNA Methylation References
#'
#' Estimates cell type fractions in bulk DNA methylation (DNAm) samples using
#' the BLEND-M two-step inverse-variance weighted non-negative least squares
#' estimator. BLEND-M learns a personalized cell-type-specific (CTS)
#' reference profile for each bulk sample by blending available purified
#' reference data, and down-weights CpG sites with high measurement variance.
#'
#'
#' @param mixture_sample A numeric matrix of DNAm beta values for bulk samples,
#'   with CpG sites as rows and bulk samples as columns. Row names must match
#'   those of the matrices in \code{reference.list}. Values equal to exactly 0
#'   or 1 are clamped internally.
#' @param reference.list A named list of numeric matrices, one per cell type.
#'   Each matrix has CpG sites as rows (with row names matching those of
#'   \code{mixture_sample}) and purified reference samples as columns. The
#'   names of the list define the cell type labels returned in the output.
#' @param unknown.include TRUE or FALSE, decide if we consider a cell type with unknown reference.
#'  Default as TRUE, recommended for real data application.
#' @return A numeric matrix of estimated cell type fractions with bulk samples
#'   as rows and cell types as columns. Each row sums to 1. Column names
#'   correspond to the names of \code{reference.list}.
#'
#'
#'
#' @importFrom nnls nnls
#' @importFrom rlist list.cbind list.rbind
#' @export
BLENDM <- function(mixture_sample, reference.list, unknown.include = TRUE) {
  if(unknown.include){
    unknown.ct <- matrix(1, nrow = nrow(reference.list[[1]]), ncol=2)
    unknown.ct[,2] <- 1
    colnames(unknown.ct) <- c("unknown0","unknown1") # just to have two columns, same 1 vectors
    rownames(unknown.ct) <- rownames(reference.list[[1]])
    reference.list <- c(reference.list, list("unknown"=unknown.ct))
  }
  
  M <- unlist(lapply(reference.list, ncol))  # number of purified samples per cell type
  G <- nrow(mixture_sample)                  # number of CpGs
  N <- ncol(mixture_sample)                  # number of bulk samples

  ## Build full reference matrix (G x sum(M))
  reference.mtx <- as.matrix(do.call(cbind, reference.list))
  
  ## Clamp boundary beta values
  mixture_sample[mixture_sample == 1] <- 1 - 1e-15
  mixture_sample[mixture_sample == 0] <- 1e-15

  ## --- Step I: initial NNLS estimation ---
  one_run <- function(ref.mtx, mix.sample) {
    nnls_res <- lapply(seq_len(ncol(mix.sample)), function(i) {
      nnls::nnls(ref.mtx, mix.sample[, i])
    })
    frac_est <- do.call(rbind, lapply(nnls_res, function(res) res$x))
    frac_est <- t(apply(frac_est, 1, function(x) x / sum(x)))
    list(frac = frac_est)
  }

  initialize  <- one_run(reference.mtx, mixture_sample)
  fitted_mean <- reference.mtx %*% t(initialize$frac)  # G x N
  rm(initialize)

  ## MLE for CpG-specific precision parameters
  dispersion_est <- sapply(seq_len(G), function(i) {
    estimate_dispersion(mixture_sample[i, ], fitted_mean[i, ])
  })

  ## Per-CpG, per-sample standard deviations
  beta_sd <- function(mean_vec, dispersion_vec) {
    sqrt((mean_vec * (1 - mean_vec)) / dispersion_vec)
  }
  sd_est         <- sapply(seq_len(N), function(i) {
    beta_sd(fitted_mean[, i], dispersion_est)
  })  # G x N
  inverse_cpg_sd <- 1 / sd_est
  rm(sd_est)

  ## --- Step II: inverse-variance weighted NNLS ---
  one_sample_wnnls <- function(single_sample, ref.mtx, inverse_sd) {
    nnls::nnls(
      apply(ref.mtx, 2, function(x) x * inverse_sd),
      single_sample * inverse_sd
    )$x
  }

  refit <- t(sapply(seq_len(N), function(i) {
    one_sample_wnnls(mixture_sample[, i], reference.mtx, inverse_cpg_sd[, i])
  }))

  ## Recover cell type fractions by summing Psi over purified samples per cell type
  group_labels <- rep(names(M), times = M)
  frac_est_new <- t(apply(refit, 1, function(x) tapply(x, group_labels, sum)))
  all_res      <- t(apply(frac_est_new, 1, function(x) x / sum(x)))

  return(all_res)
}
