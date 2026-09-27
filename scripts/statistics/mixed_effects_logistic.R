#!/usr/bin/env Rscript

# MRI-QRF logistic mixed-effects analysis
# Usage:
#   Rscript mixed_effects_logistic.R glmm_input.csv output_dir
#
# Required R packages:
#   lme4, emmeans, readr, dplyr

args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 2) stop("Usage: Rscript mixed_effects_logistic.R input.csv output_dir")

input_file <- args[1]
output_dir <- args[2]
dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)

suppressPackageStartupMessages({
  library(lme4)
  library(emmeans)
  library(readr)
  library(dplyr)
})

d <- read_csv(input_file, show_col_types=FALSE)
required <- c("dataset","model","condition","patient_key","training_seed","correct")
if (!all(required %in% names(d))) stop("Input is missing required columns")

all_est <- list()
all_con <- list()

groups <- d %>% distinct(dataset, model)

for (i in seq_len(nrow(groups))) {
  ds <- groups$dataset[i]
  mdl <- groups$model[i]
  z <- d %>% filter(dataset == ds, model == mdl)

  z$condition <- factor(z$condition)
  if (!("clean" %in% levels(z$condition))) stop(paste(ds, mdl, "has no clean condition"))
  z$condition <- relevel(z$condition, ref="clean")
  z$patient_key <- factor(z$patient_key)
  z$training_seed <- factor(z$training_seed)

  fit <- glmer(
    correct ~ condition + (1 | patient_key) + (1 | training_seed),
    data=z,
    family=binomial(link="logit"),
    control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=200000))
  )

  saveRDS(fit, file.path(output_dir, paste0(ds,"__",mdl,"__glmer.rds")))

  coef_tab <- as.data.frame(summary(fit)$coefficients)
  coef_tab$term <- rownames(coef_tab)
  coef_tab$dataset <- ds
  coef_tab$model <- mdl
  rownames(coef_tab) <- NULL
  all_est[[length(all_est)+1]] <- coef_tab

  emm <- emmeans(fit, ~ condition, type="response")
  # Treatment-vs-control contrasts; because clean was relevelled first, ref = 1.
  con <- as.data.frame(contrast(emm, method="trt.vs.ctrl", ref=1, adjust="holm"))
  con$dataset <- ds
  con$model <- mdl
  all_con[[length(all_con)+1]] <- con

  capture.output(summary(fit),
                 file=file.path(output_dir, paste0(ds,"__",mdl,"__summary.txt")))
}

write_csv(bind_rows(all_est), file.path(output_dir, "glmm_fixed_effects.csv"))
write_csv(bind_rows(all_con), file.path(output_dir, "glmm_clean_vs_corrupted_holm.csv"))
