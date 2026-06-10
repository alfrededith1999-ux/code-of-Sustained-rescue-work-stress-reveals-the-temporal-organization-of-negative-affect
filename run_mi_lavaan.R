args <- commandArgs(trailingOnly = TRUE)
manifest_path <- args[1]
out_csv <- args[2]

suppressMessages(library(lavaan))
suppressMessages(library(semTools))
suppressMessages(library(readr))

man <- read_csv(manifest_path, show_col_types = FALSE)

fit_one <- function(data_path, model_syntax, estimator, ordinal_flag) {
  dat <- read_csv(data_path, show_col_types = FALSE)
  dat$wave <- as.factor(dat$wave)
  items <- grep("^y\\d+$", names(dat), value=TRUE)

  # ordinal: ordered=items; scalar 用 thresholds
  if (ordinal_flag) {
    ordered_items <- items
    param <- "theta"
  } else {
    ordered_items <- NULL
    param <- "delta"
  }

  fit0 <- cfa(model_syntax, data=dat, group="wave",
              estimator=estimator, meanstructure=TRUE, std.lv=TRUE,
              ordered=ordered_items, parameterization=param)

  fit1 <- cfa(model_syntax, data=dat, group="wave",
              estimator=estimator, meanstructure=TRUE, std.lv=TRUE,
              ordered=ordered_items, parameterization=param,
              group.equal=c("loadings"))

  if (ordinal_flag) {
    fit2 <- cfa(model_syntax, data=dat, group="wave",
                estimator=estimator, meanstructure=TRUE, std.lv=TRUE,
                ordered=ordered_items, parameterization=param,
                group.equal=c("loadings","thresholds"))
  } else {
    fit2 <- cfa(model_syntax, data=dat, group="wave",
                estimator=estimator, meanstructure=TRUE, std.lv=TRUE,
                ordered=ordered_items, parameterization=param,
                group.equal=c("loadings","intercepts"))
  }

  get_fit <- function(fit) {
    fm <- fitMeasures(fit, c("chisq","df","pvalue","cfi","tli","rmsea","srmr","aic","bic"))
    as.list(fm)
  }

  list(
    configural = get_fit(fit0),
    metric     = get_fit(fit1),
    scalar     = get_fit(fit2)
  )
}

rows <- list()

for (i in 1:nrow(man)) {
  scale <- man$scale[i]
  data_path <- man$data_csv[i]
  estimator <- man$estimator[i]
  ordinal_flag <- man$ordinal[i]
  model_syntax <- man$model[i]
  waves <- man$waves[i]

  cat(sprintf("[R] scale=%s | waves=%s | estimator=%s | ordinal=%s\\n",
              scale, waves, estimator, ordinal_flag))

  ok <- TRUE
  res <- NULL
  err <- NULL
  tryCatch({
    res <- fit_one(data_path, model_syntax, estimator, ordinal_flag)
  }, error=function(e){
    ok <<- FALSE
    err <<- as.character(e)
  })

  if (!ok) {
    for (m in c("configural","metric","scalar")) {
      rows[[length(rows)+1]] <- data.frame(
        scale=scale, waves=waves, model=m,
        chisq=NA, df=NA, pvalue=NA, cfi=NA, tli=NA, rmsea=NA, srmr=NA, aic=NA, bic=NA,
        delta_cfi=NA, delta_rmsea=NA, delta_srmr=NA,
        pass=FALSE, error=err, stringsAsFactors=FALSE
      )
    }
    next
  }

  cf <- res$configural
  mt <- res$metric
  sc <- res$scalar

  d_cfi_m  <- mt$cfi   - cf$cfi
  d_rm_m   <- mt$rmsea - cf$rmsea
  d_srm_m  <- mt$srmr  - cf$srmr

  d_cfi_s  <- sc$cfi   - mt$cfi
  d_rm_s   <- sc$rmsea - mt$rmsea
  d_srm_s  <- sc$srmr  - mt$srmr

  pass_metric <- (d_cfi_m >= -0.01) && (d_rm_m <= 0.015) && (d_srm_m <= 0.03)
  pass_scalar <- (d_cfi_s >= -0.01) && (d_rm_s <= 0.015) && (d_srm_s <= 0.01)

  rows[[length(rows)+1]] <- data.frame(
    scale=scale, waves=waves, model="configural",
    chisq=cf$chisq, df=cf$df, pvalue=cf$pvalue, cfi=cf$cfi, tli=cf$tli, rmsea=cf$rmsea, srmr=cf$srmr, aic=cf$aic, bic=cf$bic,
    delta_cfi=NA, delta_rmsea=NA, delta_srmr=NA,
    pass=TRUE, error="", stringsAsFactors=FALSE
  )
  rows[[length(rows)+1]] <- data.frame(
    scale=scale, waves=waves, model="metric",
    chisq=mt$chisq, df=mt$df, pvalue=mt$pvalue, cfi=mt$cfi, tli=mt$tli, rmsea=mt$rmsea, srmr=mt$srmr, aic=mt$aic, bic=mt$bic,
    delta_cfi=d_cfi_m, delta_rmsea=d_rm_m, delta_srmr=d_srm_m,
    pass=pass_metric, error="", stringsAsFactors=FALSE
  )
  rows[[length(rows)+1]] <- data.frame(
    scale=scale, waves=waves, model="scalar",
    chisq=sc$chisq, df=sc$df, pvalue=sc$pvalue, cfi=sc$cfi, tli=sc$tli, rmsea=sc$rmsea, srmr=sc$srmr, aic=sc$aic, bic=sc$bic,
    delta_cfi=d_cfi_s, delta_rmsea=d_rm_s, delta_srmr=d_srm_s,
    pass=pass_scalar, error="", stringsAsFactors=FALSE
  )
}

out <- do.call(rbind, rows)
write_csv(out, out_csv)
cat(sprintf("[R] DONE -> %s\\n", out_csv))
