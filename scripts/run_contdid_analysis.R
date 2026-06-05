# Continuous-treatment DiD event studies with bcallaway11/contdid.
#
# Run from repo root:
#   Rscript scripts/run_contdid_analysis.R
#
# This script requires R. If contdid is missing, it installs from the
# bcallaway11 r-universe repository.

options(stringsAsFactors = FALSE)

ROOT <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
DATA_PATH <- file.path(ROOT, "data", "processed", "sepe_cno4_monthly_ai_exposure.csv")
OUT_DIR <- file.path(ROOT, "analysis", "econometrics_outputs", "contdid")
INPUT_DIR <- file.path(OUT_DIR, "inputs")
FIG_DIR <- file.path(OUT_DIR, "Graficos")
RLIB_DIR <- file.path(ROOT, ".r_libs")
INTERVENTION_PERIOD <- "2022-09"
BOOTSTRAPS <- 5

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(INPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(RLIB_DIR, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(RLIB_DIR, .libPaths()))

ensure_package <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    if (pkg == "contdid") {
      install.packages(
        "contdid",
        repos = c("https://bcallaway11.r-universe.dev", "https://cloud.r-project.org")
      )
    } else {
      install.packages(pkg, repos = "https://cloud.r-project.org")
    }
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

ensure_package("contdid")
ensure_package("ggplot2")
ensure_package("svglite")
ensure_package("ptetools")
ensure_package("openxlsx")

read_panel <- function() {
  df <- read.csv(DATA_PATH, check.names = FALSE)
  df <- df[df$dimension == "total" & df$category == "Total" & df$gender == "Total", ]
  df$cno4 <- sprintf("%04d", as.integer(df$cno4))
  periods <- sort(unique(df$period))
  intervention_index <- match(INTERVENTION_PERIOD, periods)
  if (is.na(intervention_index)) {
    stop("INTERVENTION_PERIOD not found in period column.")
  }
  df$time_period <- match(df$period, periods)
  df$event_time <- df$time_period - intervention_index
  df$first_treat_period <- intervention_index
  df$Y_unemployment <- log1p(df$parados)
  df$Y_contracts <- df$contratos
  df
}

make_balanced <- function(df, yname, dname) {
  keep <- c("cno4", "occupation_title", "period", "time_period", "event_time",
            "first_treat_period", yname, dname)
  dat <- df[, keep]
  names(dat)[names(dat) == yname] <- "Y"
  names(dat)[names(dat) == dname] <- "D_raw"
  dat <- dat[is.finite(dat$Y) & is.finite(dat$D_raw), ]

  periods <- sort(unique(dat$period))
  counts <- tapply(dat$period, dat$cno4, function(x) length(unique(x)))
  complete_units <- names(counts)[counts == length(periods)]
  dat <- dat[dat$cno4 %in% complete_units, ]

  exposure <- unique(dat[, c("cno4", "D_raw")])
  q25 <- as.numeric(quantile(exposure$D_raw, 0.25, na.rm = TRUE))
  q50 <- as.numeric(quantile(exposure$D_raw, 0.50, na.rm = TRUE))
  q75 <- as.numeric(quantile(exposure$D_raw, 0.75, na.rm = TRUE))

  zero_count <- length(unique(dat$cno4[dat$D_raw == 0]))
  if (zero_count > 0) {
    dat <- dat[dat$D_raw > 0 | dat$D_raw == 0, ]
    dat$G <- ifelse(dat$D_raw > 0, dat$first_treat_period, 0)
    dat$D <- ifelse(dat$D_raw > 0, dat$D_raw, 0)
    control_rule <- "zero-dose CNOs"
  } else {
    dat <- dat[dat$D_raw > q25 | dat$D_raw <= q25, ]
    dat$G <- ifelse(dat$D_raw > q25, dat$first_treat_period, 0)
    dat$D <- ifelse(dat$D_raw > q25, dat$D_raw, 0)
    control_rule <- "bottom-quartile CNOs set as zero-dose baseline"
  }

  dat$id <- as.integer(factor(dat$cno4))
  attr(dat, "q25") <- q25
  attr(dat, "q50") <- q50
  attr(dat, "q75") <- q75
  attr(dat, "control_rule") <- control_rule
  dat
}

airef_theme <- function() {
  ggplot2::theme_minimal(base_family = "sans", base_size = 9) +
    ggplot2::theme(
      text = ggplot2::element_text(color = "#404040", face = "bold"),
      axis.text = ggplot2::element_text(color = "#4D4D4D", face = "bold"),
      axis.title = ggplot2::element_text(color = "#404040", face = "bold"),
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_line(color = "#CCCCCC", linewidth = 0.35),
      axis.line = ggplot2::element_line(color = "#404040", linewidth = 0.35),
      axis.ticks = ggplot2::element_line(color = "#404040", linewidth = 0.35),
      panel.border = ggplot2::element_blank(),
      plot.background = ggplot2::element_rect(fill = "white", color = NA),
      panel.background = ggplot2::element_rect(fill = "white", color = NA),
      plot.title = ggplot2::element_blank(),
      legend.title = ggplot2::element_blank(),
      legend.background = ggplot2::element_blank()
    )
}

clean_plot_data <- function(plot_data) {
  if (!is.null(plot_data) && nrow(plot_data) > 0) {
    names(plot_data) <- ifelse(is.na(names(plot_data)) | names(plot_data) == "",
                               "significance", names(plot_data))
  }
  plot_data
}

make_airef_event_plot <- function(plot_data) {
  plot_data <- clean_plot_data(plot_data)
  ggplot2::ggplot(plot_data, ggplot2::aes(x = e, y = att)) +
    ggplot2::geom_ribbon(
      ggplot2::aes(ymin = cil, ymax = ciu),
      fill = "#E397A0",
      alpha = 0.35,
      linewidth = 0
    ) +
    ggplot2::geom_line(color = "#83082A", linewidth = 0.9) +
    ggplot2::geom_point(color = "#83082A", size = 1.6) +
    ggplot2::geom_vline(xintercept = -0.5, linetype = "dashed", color = "#404040", linewidth = 0.35) +
    ggplot2::geom_hline(yintercept = 0, color = "#404040", linewidth = 0.35) +
    ggplot2::scale_x_continuous(breaks = seq(min(plot_data$e), max(plot_data$e), by = 6)) +
    ggplot2::labs(x = "Meses desde septiembre de 2022", y = "ACRT") +
    airef_theme()
}

make_airef_dose_plot <- function(plot_data, y_label) {
  plot_data <- clean_plot_data(plot_data)
  plot_data$cil <- plot_data$est - plot_data$crit * plot_data$se
  plot_data$ciu <- plot_data$est + plot_data$crit * plot_data$se
  ggplot2::ggplot(plot_data, ggplot2::aes(x = dose, y = est)) +
    ggplot2::geom_ribbon(
      ggplot2::aes(ymin = cil, ymax = ciu),
      fill = "#E397A0",
      alpha = 0.35,
      linewidth = 0
    ) +
    ggplot2::geom_line(color = "#83082A", linewidth = 0.9) +
    ggplot2::geom_hline(yintercept = 0, color = "#404040", linewidth = 0.35) +
    ggplot2::labs(x = "Exposicion a IA", y = y_label) +
    airef_theme()
}

save_plot <- function(plot_obj, plot_data, path_base, workbook_title, workbook_note) {
  ggplot2::ggsave(paste0(path_base, ".pdf"), plot_obj, width = 14.5, height = 7.25, units = "cm")
  ggplot2::ggsave(paste0(path_base, ".png"), plot_obj, width = 14.5, height = 7.25, units = "cm", dpi = 300)
  ggplot2::ggsave(paste0(path_base, ".svg"), plot_obj, width = 14.5, height = 7.25, units = "cm")

  plot_data <- clean_plot_data(plot_data)
  if (!is.null(plot_data) && nrow(plot_data) > 0) {
    wb <- openxlsx::createWorkbook()
    openxlsx::addWorksheet(wb, "Grafico")
    openxlsx::writeData(wb, "Grafico", workbook_title, startCol = 2, startRow = 2)
    openxlsx::writeData(wb, "Grafico",
                        "Fuente: AIReF a partir de SEPE y estimaciones con contdid.",
                        startCol = 2, startRow = 3)
    openxlsx::writeData(wb, "Grafico", workbook_note, startCol = 2, startRow = 4)
    openxlsx::writeData(wb, "Grafico", plot_data, startCol = 4, startRow = 5)
    header_style <- openxlsx::createStyle(
      fgFill = "#83082A", fontColour = "#FFFFFF", textDecoration = "bold"
    )
    text_style <- openxlsx::createStyle(fontColour = "#404040", fontName = "Century Gothic")
    openxlsx::addStyle(wb, "Grafico", text_style, rows = 1:(nrow(plot_data) + 8), cols = 1:12, gridExpand = TRUE)
    openxlsx::addStyle(wb, "Grafico", header_style, rows = 5, cols = 4:(ncol(plot_data) + 3), gridExpand = TRUE)
    openxlsx::setColWidths(wb, "Grafico", cols = 1:12, widths = "auto")
    openxlsx::saveWorkbook(wb, paste0(path_base, ".xlsx"), overwrite = TRUE)
  }
}

cont_did_pte <- function(dat, target_parameter, aggregation) {
  setup_wrapper <- function(
    yname, gname, tname, idname, data, xformula = ~1,
    dname = "D", degree = 3, num_knots = 1, ...
  ) {
    contdid:::setup_pte_cont(
      yname = yname,
      gname = gname,
      tname = tname,
      idname = idname,
      data = data,
      xformula = xformula,
      target_parameter = target_parameter,
      aggregation = aggregation,
      treatment_type = "continuous",
      dname = dname,
      dvals = NULL,
      degree = degree,
      num_knots = num_knots,
      ...
    )
  }

  gt_type <- ifelse(aggregation == "dose", "dose", "att")
  ptetools::pte(
    yname = "Y",
    gname = "G",
    tname = "time_period",
    idname = "id",
    data = dat,
    setup_pte_fun = setup_wrapper,
    subset_fun = contdid:::cont_two_by_two_subset,
    attgt_fun = contdid:::cont_did_acrt,
    gt_type = gt_type,
    cband = TRUE,
    biters = BOOTSTRAPS,
    cl = 1,
    dname = "D",
    dose_est_method = "parametric",
    degree = 3,
    num_knots = 1
  )
}

run_one <- function(df, outcome_key, yname, exposure_key, dname) {
  spec_name <- paste(outcome_key, exposure_key, "continuous", sep = "_")
  message("Running ", spec_name)
  spec_dir <- file.path(OUT_DIR, spec_name)
  fig_dir <- file.path(FIG_DIR, outcome_key)
  dir.create(spec_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

  dat <- make_balanced(df, yname, dname)
  write.csv(dat, file.path(INPUT_DIR, paste0(spec_name, ".csv")), row.names = FALSE)

  res <- cont_did_pte(dat, target_parameter = "slope", aggregation = "eventstudy")

  saveRDS(res, file.path(spec_dir, "contdid_result.rds"))
  capture.output(summary(res), file = file.path(spec_dir, "summary.txt"))

  event_plot_data <- contdid::ggcont_did(res)$data
  p <- make_airef_event_plot(event_plot_data)
  workbook_title <- paste("Event study contdid -", outcome_key, "-", exposure_key)
  workbook_note <- paste(
    "Nota: Efectos ACRT por periodo relativo a septiembre de 2022.",
    "Intervalos con bandas de confianza simultaneas.",
    "Regla de control:", attr(dat, "control_rule")
  )
  save_plot(
    p,
    event_plot_data,
    file.path(fig_dir, paste0("contdid_", exposure_key, "_continuous")),
    workbook_title,
    workbook_note
  )

  dose_res <- cont_did_pte(dat, target_parameter = "slope", aggregation = "dose")
  saveRDS(dose_res, file.path(spec_dir, "contdid_dose_result.rds"))
  capture.output(summary(dose_res), file = file.path(spec_dir, "summary_dose.txt"))

  dose_fig_dir <- file.path(FIG_DIR, outcome_key, "dose")
  dir.create(dose_fig_dir, recursive = TRUE, showWarnings = FALSE)
  dose_att_data <- contdid::ggcont_did(dose_res, type = "att")$data
  dose_att_plot <- make_airef_dose_plot(dose_att_data, "ATT(d)")
  save_plot(
    dose_att_plot,
    dose_att_data,
    file.path(dose_fig_dir, paste0("contdid_", exposure_key, "_dose_att")),
    paste("Dose aggregation ATT(d) -", outcome_key, "-", exposure_key),
    paste("Nota: Agregacion por dosis segun Case 1 del README de contdid.", "Regla de control:", attr(dat, "control_rule"))
  )

  dose_acrt_data <- contdid::ggcont_did(dose_res, type = "acrt")$data
  dose_acrt_plot <- make_airef_dose_plot(dose_acrt_data, "ACRT(d)")
  save_plot(
    dose_acrt_plot,
    dose_acrt_data,
    file.path(dose_fig_dir, paste0("contdid_", exposure_key, "_dose_acrt")),
    paste("Dose aggregation ACRT(d) -", outcome_key, "-", exposure_key),
    paste("Nota: Agregacion por dosis segun Case 1 del README de contdid.", "Regla de control:", attr(dat, "control_rule"))
  )

  data.frame(
    spec = spec_name,
    outcome = outcome_key,
    exposure = exposure_key,
    design = "continuous dose; positive/higher-dose CNOs adopt in 2022-09",
    control_rule = attr(dat, "control_rule"),
    n_units = length(unique(dat$id)),
    n_periods = length(unique(dat$time_period)),
    q25 = attr(dat, "q25"),
    q50 = attr(dat, "q50"),
    q75 = attr(dat, "q75"),
    bootstraps = BOOTSTRAPS
  )
}

panel <- read_panel()

specs <- list(
  list("unemployment", "Y_unemployment", "rf", "observed_exposure_rf"),
  list("unemployment", "Y_unemployment", "cosine_weighted", "observed_exposure_cosine_weighted"),
  list("unemployment", "Y_unemployment", "cosine_nearest", "observed_exposure_cosine_nearest"),
  list("contracts", "Y_contracts", "rf", "observed_exposure_rf"),
  list("contracts", "Y_contracts", "cosine_weighted", "observed_exposure_cosine_weighted"),
  list("contracts", "Y_contracts", "cosine_nearest", "observed_exposure_cosine_nearest")
)

metadata <- do.call(rbind, lapply(specs, function(s) {
  run_one(panel, s[[1]], s[[2]], s[[3]], s[[4]])
}))

write.csv(metadata, file.path(OUT_DIR, "contdid_run_metadata.csv"), row.names = FALSE)
message("Wrote outputs to ", OUT_DIR)
