suppressPackageStartupMessages({
  library(contdid)
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(tibble)
})

project_root <- "C:/Users/ngonzalezp/OneDrive - AIREF/Escritorio/Unempolyment_Benefits/ai_unemployment_analysis"

raw_path <- file.path(project_root, "data/raw/sepe_cno4_monthly_ai_exposure.csv")
tables_dir <- file.path(project_root, "output/tables")
figures_dir <- file.path(project_root, "output/figures")
processed_dir <- file.path(project_root, "data/processed")
dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)

exposure_var <- "observed_exposure_cosine_nearest"
event_period <- "2022-09"

safe_name <- function(x) {
  out <- as.character(x)
  out <- gsub("<", "lt", out)
  out <- gsub(">", "gt", out)
  out <- gsub("-", "_", out)
  out <- gsub(" ", "_", out)
  out <- gsub("[^A-Za-z0-9_]", "_", out)
  tolower(out)
}

cont_did_fixed <- function(
    yname,
    dname,
    gname,
    tname,
    idname,
    data,
    target_parameter = c("level", "slope"),
    aggregation = c("dose", "eventstudy"),
    treatment_type = "continuous",
    dose_est_method = "parametric",
    dvals = NULL,
    degree = 3,
    num_knots = 0,
    control_group = "nevertreated",
    anticipation = 0,
    alp = 0.05,
    cband = TRUE,
    boot_type = "multiplier",
    biters = 50,
    base_period = "varying",
    cl = 1,
    xformula = ~1) {
  target_parameter <- match.arg(target_parameter)
  aggregation <- match.arg(aggregation)

  if (treatment_type != "continuous") {
    stop("Only continuous treatment is implemented in this runner.")
  }

  attgt_fun <- contdid:::cont_did_acrt
  gt_type <- "dose"
  if (aggregation == "eventstudy" && target_parameter == "level") {
    attgt_fun <- function(gt_data, ...) {
      ptetools::did_attgt(gt_data = gt_data, xformula = xformula)
    }
    gt_type <- "att"
  }
  if (aggregation == "eventstudy" && target_parameter == "slope") {
    gt_type <- "att"
  }

  setup_fun <- function(
      yname, gname, tname, idname, data, panel = TRUE, cband = TRUE,
      alp = 0.05, boot_type = "multiplier", gt_type = "att",
      weightsname = NULL, ret_quantile = NULL, probs = NULL,
      biters = 100, cl = 1, call = NULL, ...) {
    ptep <- contdid:::setup_pte_cont(
      yname = yname,
      gname = gname,
      tname = tname,
      idname = idname,
      data = data,
      xformula = xformula,
      target_parameter = target_parameter,
      aggregation = aggregation,
      treatment_type = treatment_type,
      required_pre_periods = 1,
      anticipation = anticipation,
      base_period = base_period,
      cband = cband,
      alp = alp,
      boot_type = boot_type,
      weightsname = weightsname,
      gt_type = gt_type,
      biters = biters,
      cl = cl,
      dname = dname,
      dvals = dvals,
      degree = degree,
      num_knots = num_knots,
      panel = panel,
      ret_quantile = ret_quantile,
      probs = probs,
      call = call
    )
    ptep$control_group <- control_group
    ptep
  }

  ptetools::pte(
    yname = yname,
    gname = gname,
    tname = tname,
    idname = idname,
    data = data,
    setup_pte_fun = setup_fun,
    subset_fun = contdid:::cont_two_by_two_subset,
    attgt_fun = attgt_fun,
    cband = cband,
    alp = alp,
    boot_type = boot_type,
    gt_type = gt_type,
    biters = biters,
    cl = cl,
    control_group = control_group,
    anticipation = anticipation,
    base_period = base_period,
    dose_est_method = dose_est_method,
    dvals = dvals,
    degree = degree,
    num_knots = num_knots
  )
}

add_time_and_outcomes <- function(df) {
  df |>
    mutate(
      cno4 = sprintf("%04s", as.character(cno4)),
      parados = suppressWarnings(as.numeric(parados)),
      period_date = as.Date(paste0(period, "-01")),
      t_index = dense_rank(period_date),
      ln_parados = if_else(parados > 0, log(parados), NA_real_),
      exposure = suppressWarnings(as.numeric(.data[[exposure_var]])),
      exposure_10pp = exposure / 0.10
    )
}

prepare_contdid_panel <- function(df, sample, subgroup) {
  event_t <- df |>
    distinct(period, period_date, t_index) |>
    filter(period == event_period) |>
    pull(t_index)
  if (length(event_t) != 1) {
    stop("Could not locate event period in panel: ", sample, " / ", subgroup)
  }
  first_post <- event_t + 1
  out <- df |>
    filter(!is.na(ln_parados), !is.na(exposure), exposure >= 0) |>
    mutate(
      id = as.integer(factor(unit)),
      const = 1,
      dose = exposure_10pp,
      dose_binary = as.numeric(exposure > 0),
      g = if_else(exposure > 0, first_post, 0L),
      sample = sample,
      subgroup = subgroup
    ) |>
    select(sample, subgroup, id, unit, cno4, period, t_index, g, const, dose, dose_binary, ln_parados)
  out
}

read_dimension <- function(dimension) {
  read_csv(
    raw_path,
    show_col_types = FALSE,
    col_types = cols(.default = col_character()),
    progress = FALSE
  ) |>
    filter(dimension == !!dimension, nchar(cno4) <= 4) |>
    add_time_and_outcomes()
}

make_total_panel <- function() {
  read_csv(
    raw_path,
    show_col_types = FALSE,
    col_types = cols(.default = col_character()),
    progress = FALSE
  ) |>
    filter(dimension == "total", category == "Total", gender == "Total", nchar(cno4) <= 4) |>
    add_time_and_outcomes() |>
    mutate(unit = cno4) |>
    prepare_contdid_panel("total_cno4", "all")
}

make_province_panel <- function() {
  read_csv(
    raw_path,
    show_col_types = FALSE,
    col_types = cols(.default = col_character()),
    progress = FALSE
  ) |>
    filter(dimension == "province", nchar(cno4) <= 4) |>
    add_time_and_outcomes() |>
    mutate(unit = paste(category, cno4, sep = "__")) |>
    prepare_contdid_panel("province_cno4", "all")
}

make_age3_panel <- function() {
  age_map <- c("<18" = "<18 to 29", "18-24" = "<18 to 29", "25-29" = "<18 to 29", "30-39" = "30-39", "40-44" = "40 to >44", ">44" = "40 to >44")
  read_csv(
    raw_path,
    show_col_types = FALSE,
    col_types = cols(.default = col_character()),
    progress = FALSE
  ) |>
    filter(dimension == "age", nchar(cno4) <= 4, category %in% names(age_map)) |>
    mutate(age3 = unname(age_map[category])) |>
    add_time_and_outcomes() |>
    group_by(age3, cno4, period, period_date, t_index) |>
    summarise(
      parados = sum(parados, na.rm = TRUE),
      exposure = first(exposure),
      .groups = "drop"
    ) |>
    mutate(
      ln_parados = if_else(parados > 0, log(parados), NA_real_),
      exposure_10pp = exposure / 0.10,
      unit = cno4
    )
}

extract_summary <- function(obj, sample, subgroup, aggregation, target_parameter) {
  s <- summary(obj)
  txt <- capture.output(print(s))
  write_lines(txt, file.path(tables_dir, paste0("table_contdid_summary_", aggregation, "_", safe_name(sample), "_", safe_name(subgroup), ".txt")))
  tibble(
    sample = sample,
    subgroup = subgroup,
    aggregation = aggregation,
    target_parameter = target_parameter,
    summary_file = paste0("output/tables/table_contdid_summary_", aggregation, "_", safe_name(sample), "_", safe_name(subgroup), ".txt")
  )
}

object_to_table <- function(obj, sample, subgroup, aggregation, target_parameter) {
  pieces <- list()
  for (nm in names(obj)) {
    val <- obj[[nm]]
    if (is.atomic(val) && length(val) > 0 && length(val) <= 200) {
      pieces[[nm]] <- paste(val, collapse = ";")
    }
  }
  if (length(pieces) == 0) {
    return(tibble(
      sample = character(),
      subgroup = character(),
      aggregation = character(),
      target_parameter = character(),
      field = character(),
      value = character()
    ))
  }
  tibble(
    sample = sample,
    subgroup = subgroup,
    aggregation = aggregation,
    target_parameter = target_parameter,
    field = names(pieces),
    value = unlist(pieces, use.names = FALSE)
  )
}

write_event_table <- function(obj, sample, subgroup) {
  event <- obj$event_study
  crit <- event$crit.val.egt
  if (length(crit) == 1) {
    crit <- rep(crit, length(event$egt))
  }
  tbl <- tibble(
    sample = sample,
    subgroup = subgroup,
    event_time = event$egt,
    estimate = event$att.egt,
    std_error = event$se.egt,
    ci_low = estimate - crit * std_error,
    ci_high = estimate + crit * std_error,
    overall_att = event$overall.att,
    overall_se = event$overall.se
  )
  outfile <- file.path(tables_dir, paste0("table_contdid_eventstudy_", safe_name(sample), "_", safe_name(subgroup), ".csv"))
  write_csv(tbl, outfile)
  paste0("output/tables/", basename(outfile))
}

write_dose_table <- function(obj, sample, subgroup) {
  att_crit <- obj$att.d_crit.val
  acrt_crit <- obj$acrt.d_crit.val
  if (length(att_crit) == 1) {
    att_crit <- rep(att_crit, length(obj$dose))
  }
  if (length(acrt_crit) == 1) {
    acrt_crit <- rep(acrt_crit, length(obj$dose))
  }
  tbl <- tibble(
    sample = sample,
    subgroup = subgroup,
    dose_exposure_10pp = obj$dose,
    exposure_share = obj$dose * 0.10,
    att_dose = obj$att.d,
    att_std_error = obj$att.d_se,
    att_ci_low = att_dose - att_crit * att_std_error,
    att_ci_high = att_dose + att_crit * att_std_error,
    acrt_dose = obj$acrt.d,
    acrt_std_error = obj$acrt.d_se,
    acrt_ci_low = acrt_dose - acrt_crit * acrt_std_error,
    acrt_ci_high = acrt_dose + acrt_crit * acrt_std_error,
    overall_att = obj$overall_att,
    overall_att_se = obj$overall_att_se,
    overall_acrt = obj$overall_acrt,
    overall_acrt_se = obj$overall_acrt_se
  )
  outfile <- file.path(tables_dir, paste0("table_contdid_dose_response_", safe_name(sample), "_", safe_name(subgroup), ".csv"))
  write_csv(tbl, outfile)
  paste0("output/tables/", basename(outfile))
}

run_one_contdid <- function(panel, sample, subgroup, biters = 50) {
  panel <- panel |>
    group_by(id) |>
    filter(n_distinct(t_index) == n_distinct(panel$t_index)) |>
    ungroup()

  dose_vals <- panel |>
    filter(g > 0) |>
    distinct(id, dose) |>
    pull(dose)
  dvals <- as.numeric(quantile(dose_vals, probs = c(0.1, 0.25, 0.5, 0.75, 0.9), na.rm = TRUE))
  dvals <- sort(unique(dvals[dvals > 0 & is.finite(dvals)]))

  event_res <- cont_did_fixed(
    yname = "ln_parados",
    tname = "t_index",
    idname = "id",
    dname = "dose",
    gname = "g",
    data = panel,
    target_parameter = "slope",
    aggregation = "eventstudy",
    treatment_type = "continuous",
    dose_est_method = "parametric",
    dvals = dvals,
    degree = 1,
    num_knots = 0,
    control_group = "nevertreated",
    base_period = "varying",
    biters = biters,
    cband = TRUE,
    cl = 1
  )

  dose_res <- cont_did_fixed(
    yname = "ln_parados",
    tname = "t_index",
    idname = "id",
    dname = "dose",
    gname = "g",
    data = panel,
    target_parameter = "slope",
    aggregation = "dose",
    treatment_type = "continuous",
    dose_est_method = "parametric",
    dvals = dvals,
    degree = 1,
    num_knots = 0,
    control_group = "nevertreated",
    base_period = "varying",
    biters = biters,
    cband = TRUE,
    cl = 1
  )

  event_fig <- file.path(figures_dir, paste0("figure_contdid_eventstudy_", safe_name(sample), "_", safe_name(subgroup), ".png"))
  dose_fig <- file.path(figures_dir, paste0("figure_contdid_dose_response_", safe_name(sample), "_", safe_name(subgroup), ".png"))
  event_table <- write_event_table(event_res, sample, subgroup)
  dose_table <- write_dose_table(dose_res, sample, subgroup)
  png(event_fig, width = 1600, height = 900, res = 180)
  print(ggcont_did(event_res) + ggtitle(paste("ContDID event study:", sample, subgroup)))
  dev.off()
  png(dose_fig, width = 1600, height = 900, res = 180)
  print(ggcont_did(dose_res, type = "att") + ggtitle(paste("ContDID dose response:", sample, subgroup)))
  dev.off()

  bind_rows(
    extract_summary(event_res, sample, subgroup, "eventstudy", "slope"),
    extract_summary(dose_res, sample, subgroup, "dose", "slope")
  ) |>
    mutate(
      event_figure = paste0("output/figures/", basename(event_fig)),
      dose_figure = paste0("output/figures/", basename(dose_fig)),
      event_table = event_table,
      dose_table = dose_table,
      nobs = nrow(panel),
      units = n_distinct(panel$id),
      periods = n_distinct(panel$t_index),
      treated_units = n_distinct(panel$id[panel$g > 0]),
      control_units = n_distinct(panel$id[panel$g == 0]),
      biters = biters
    ) -> summary_rows

  detail_rows <- bind_rows(
    object_to_table(event_res, sample, subgroup, "eventstudy", "slope"),
    object_to_table(dose_res, sample, subgroup, "dose", "slope")
  )

  list(summary = summary_rows, details = detail_rows)
}

main <- function() {
  include_province <- Sys.getenv("CONTDID_INCLUDE_PROVINCE", unset = "0") == "1"
  total_panel <- make_total_panel()
  age3_raw <- make_age3_panel()

  panels <- list(
    list(sample = "total_cno4", subgroup = "all", data = total_panel, biters = 50)
  )
  if (include_province) {
    province_panel <- make_province_panel()
    panels[[length(panels) + 1]] <- list(
      sample = "province_cno4",
      subgroup = "all",
      data = province_panel,
      biters = 20
    )
  }
  for (bucket in c("<18 to 29", "30-39", "40 to >44")) {
    panels[[length(panels) + 1]] <- list(
      sample = "age3_cno4",
      subgroup = bucket,
      data = age3_raw |> filter(age3 == bucket) |> prepare_contdid_panel("age3_cno4", bucket),
      biters = 50
    )
  }

  all_summary <- list()
  all_details <- list()
  failures <- list()
  for (p in panels) {
    message("Running contdid for ", p$sample, " / ", p$subgroup)
    res <- tryCatch(
      run_one_contdid(p$data, p$sample, p$subgroup, biters = p$biters),
      error = function(e) {
        failures[[length(failures) + 1]] <<- tibble(sample = p$sample, subgroup = p$subgroup, error = conditionMessage(e))
        NULL
      }
    )
    if (!is.null(res)) {
      all_summary[[length(all_summary) + 1]] <- res$summary
      all_details[[length(all_details) + 1]] <- res$details
    }
  }

  summary_tbl <- bind_rows(all_summary)
  detail_tbl <- bind_rows(all_details)
  failure_tbl <- if (length(failures) == 0) {
    tibble(sample = character(), subgroup = character(), error = character())
  } else {
    bind_rows(failures)
  }

  write_csv(summary_tbl, file.path(tables_dir, "table_contdid_results_index.csv"))
  write_csv(detail_tbl, file.path(tables_dir, "table_contdid_object_details.csv"))
  write_csv(failure_tbl, file.path(tables_dir, "table_contdid_failures.csv"))

  metadata <- c(
    "{",
    '  "tables": [',
    '    "output/tables/table_contdid_results_index.csv",',
    '    "output/tables/table_contdid_object_details.csv",',
    '    "output/tables/table_contdid_failures.csv"',
    "  ],",
    '  "note": "contdid uses exposure_10pp as dose, positive exposure as treated from the first post-Sep-2022 month, and zero exposure as never-treated controls. Set CONTDID_INCLUDE_PROVINCE=1 to rerun the long province-CNO4 panel."',
    "}"
  )
  writeLines(metadata, file.path(processed_dir, "contdid_metadata.json"))
}

main()
