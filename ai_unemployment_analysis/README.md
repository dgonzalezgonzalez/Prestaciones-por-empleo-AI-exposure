# AI Exposure and Unemployment in Spain

This project studies whether occupations with higher AI exposure experienced larger post-2022 changes in registered unemployment (`parados`) using SEPE CNO4-month data.

The analysis follows the empirical spirit of Anthropic's labor-market impact framework: compare high-exposure occupations with low/no-exposure occupations before and after the diffusion of modern generative AI.

## Data

Raw input:

- `data/raw/sepe_cno4_monthly_ai_exposure.csv`

Processed panel:

- `data/processed/analysis_panel.csv`

Main sample:

- CNO4-month rows
- `dimension == total`
- `category == Total`
- `gender == Total`
- January 2021 through March 2026
- 502 occupations and 31,626 occupation-month rows

Main exposure:

- `observed_exposure_cosine_nearest`

Main treatment:

- Treated: exposure greater than the panel 75th percentile, 0.1169
- Control: exposure equal to zero
- Middle-exposure occupations are excluded from the binary DiD

Main post period:

- Event month: September 2022
- Post equals 1 for months after September 2022

## Reproduce

One-command run on this machine:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

Or run the main analysis directly with the Codex bundled Python runtime:

```powershell
& 'C:\Users\ngonzalezp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' code\run_analysis.py
& 'C:\Users\ngonzalezp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' code\run_next_steps.py
& 'C:\Users\ngonzalezp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' code\run_research_moves.py
& 'C:\Users\ngonzalezp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' code\run_synthetic_methods.py
& 'C:\Users\ngonzalezp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' code\run_continuous_event_studies.py
& 'C:\Users\ngonzalezp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' code\run_binary_event_study_samples.py
& 'C:\Users\ngonzalezp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' code\run_exposure_variation.py
& 'C:\Users\ngonzalezp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' code\run_memo_v1_assets.py
& 'C:\Users\ngonzalezp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' code\run_memo_v2_assets.py
& 'C:\Users\ngonzalezp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' code\run_memo_v2_event_assets.py
& 'C:\Users\ngonzalezp\AppData\Local\Programs\R\R-4.5.2\bin\Rscript.exe' code\run_contdid_analysis.R
```

The default ContDID run covers total CNO4 and the three grouped age-CNO4 panels. To rerun the long province-CNO4 ContDID panel, set `CONTDID_INCLUDE_PROVINCE=1` before running `code\run_contdid_analysis.R`.

Install dependencies if needed:

```powershell
& 'C:\Users\ngonzalezp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pip install -r requirements.txt
```

Run referee replication checks:

```powershell
& 'C:\Users\ngonzalezp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' code\replication\referee2_replicate_core.py
& 'C:\Users\ngonzalezp\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' code\replication\referee2_replicate_node_counts.js
```

## Key Outputs

Figures:

- `output/figures/figure_01_mean_log_parados_by_exposure_group.png`
- `output/figures/figure_02_indexed_total_log_parados_by_exposure_group.png`
- `output/figures/figure_03_exposure_distribution.png`
- `output/figures/figure_04_event_study_high_vs_zero_sep2022.png`
- `output/figures/figure_05_next_step_subgroup_parados.png`
- `output/figures/figure_06_next_step_robustness_diagnostics.png`
- `output/figures/figure_07_v1_cno2_month_fe_event_study.png`
- `output/figures/figure_08_recommended_age3039_cno2_month_fe_event_study.png`
- `output/figures/figure_09_recommended_honestdid_smoothness_original.png`
- `output/figures/figure_synthetic_total_ocup4d.png`
- `output/figures/figure_synthetic_province_ocup4d.png`
- `output/figures/figure_synthetic_age3039_ocup4d.png`
- `output/figures/figure_continuous_event_study_sample1_total_ocup4d.png`
- `output/figures/figure_continuous_event_study_sample2_province_ocup4d.png`
- `output/figures/figure_continuous_event_study_sample3_gender_ocup4d.png`
- `output/figures/figure_continuous_event_study_sample4_age_ocup4d.png`
- `output/figures/figure_binary_event_study_sample1_total_ocup4d.png`
- `output/figures/figure_binary_event_study_sample2_province_ocup4d.png`
- `output/figures/figure_binary_event_study_sample3_gender_ocup4d.png`
- `output/figures/figure_binary_event_study_sample4_age_ocup4d.png`
- `output/figures/figure_binary_event_study_sample4_age3039_ocup4d.png`
- `output/figures/figure_exposure_variation_within_cno2.png`
- `output/figures/figure_memo_v1_indexed_total_all_ln_parados.png`
- `output/figures/figure_memo_v1_indexed_total_all_ln_contratos.png`
- `output/figures/figure_memo_v1_indexed_total_all_ln_parados_contratos.png`
- `output/figures/figure_memo_v1_indexed_age_*.png`
- `output/figures/figure_memo_v2_indexed_total_all_ln_parados.png`
- `output/figures/figure_memo_v2_indexed_total_all_ln_contratos.png`
- `output/figures/figure_memo_v2_indexed_total_all_ln_parados_contratos.png`
- `output/figures/figure_memo_v2_indexed_age3_*.png`
- `output/figures/figure_memo_v2_twfe_event_*.png`
- `output/figures/figure_memo_v2_sdid_event_*.png`
- `output/figures/figure_contdid_eventstudy_*.png`
- `output/figures/figure_contdid_dose_response_*.png`

Tables:

- `output/tables/table_top50_exposed_occupations.csv`
- `output/tables/table_exposure_group_summary.csv`
- `output/tables/table_regression_specifications.csv`
- `output/tables/table_event_study_coefficients.csv`
- `output/tables/table_next_step_subgroup_heterogeneity.csv`
- `output/tables/table_next_step_province_fe.csv`
- `output/tables/table_next_step_cno2_rate_proxy.csv`
- `output/tables/table_next_step_timing_continuous_contracts.csv`
- `output/tables/table_next_step_exposure_validation_queue.csv`
- `output/tables/table_v1_cno2_month_fe_specifications.csv`
- `output/tables/table_v1_cno2_month_fe_event_study.csv`
- `output/tables/table_v1_exposure_match_validation_queue.csv`
- `output/tables/table_recommended_cno2_fwl_decomposition.csv`
- `output/tables/table_recommended_cno2_family_summary.csv`
- `output/tables/table_recommended_within_cno2_family_did.csv`
- `output/tables/table_recommended_age3039_cno2_month_fe.csv`
- `output/tables/table_recommended_age3039_cno2_month_fe_event_study.csv`
- `output/tables/table_recommended_honestdid_original_ci.csv`
- `output/tables/table_recommended_honestdid_smoothness.csv`
- `output/tables/table_recommended_honestdid_relative_magnitude.csv`
- `output/tables/table_research_move_focus_cno2_summary.csv`
- `output/tables/table_research_move_focus_cno2_occupation_profiles.csv`
- `output/tables/table_research_move_focus_cno2_event_summary.csv`
- `output/tables/table_research_move_denominator_inventory.csv`
- `output/tables/table_synthetic_did_control_results.csv`
- `output/tables/table_synthetic_donor_weights.csv`
- `output/tables/table_synthetic_paths.csv`
- `output/tables/table_continuous_event_study_summary.csv`
- `output/tables/table_continuous_event_study_all_samples.csv`
- `output/tables/table_binary_event_study_summary.csv`
- `output/tables/table_binary_event_study_all_samples.csv`
- `output/tables/table_exposure_variation_within_cno2.csv`
- `output/tables/table_memo_v1_indexed_inputs_total.csv`
- `output/tables/table_memo_v1_indexed_inputs_age.csv`
- `output/tables/table_memo_v2_indexed_inputs_total.csv`
- `output/tables/table_memo_v2_indexed_inputs_age3.csv`
- `output/tables/table_memo_v2_age3_twfe.csv`
- `output/tables/table_memo_v2_twfe_event_summary.csv`
- `output/tables/table_memo_v2_twfe_event_coefficients.csv`
- `output/tables/table_memo_v2_sdid_event_summary.csv`
- `output/tables/table_memo_v2_sdid_event_paths.csv`
- `output/tables/table_contdid_results_index.csv`
- `output/tables/table_contdid_eventstudy_*.csv`
- `output/tables/table_contdid_dose_response_*.csv`

Reports:

- `Summary of the exercises.md`
- `docs/research_memo.md`
- `docs/research_memo_v1.md`
- `docs/research_memo_v2.md`
- `docs/recommended_next_steps_memo.md`
- `docs/recommended_next_steps_memo_v1.md`
- `docs/recommended_steps_implementation_memo.md`
- `docs/next_research_moves_and_synthetic_memo.md`
- `correspondence/blindspot/2026-06-03_blindspot_report.md`
- `correspondence/referee2/2026-06-03_round1_report.md`

Stata empirical-analysis do-files:

- `code/stata_empirical_analysis_plan.do`
- `code/stata_produce_all_figures.do`
- `code/stata_sdid_event_focused.do`

Main Stata outputs after running `code/stata_empirical_analysis_plan.do` in Stata:

- `output/stata/tables/stata_spec_catalog.csv`
- `output/stata/tables/stata_twfe_reghdfe_results.csv`
- `output/stata/tables/stata_twfe_event_study_index.csv`
- `output/stata/tables/stata_sdid_sc_results.csv`
- `output/stata/tables/stata_sdid_event_index.csv`
- `output/stata/figures/twfe_event_*.png`
- `output/stata/figures/sdid_event_*.png`

Focused Stata `sdid_event` outputs:

- `output/stata/tables/stata_sdid_event_focused_index.csv`
- `output/stata/tables/sdid_event_focused_sdid_*.csv`
- `output/stata/figures/sdid_event_focused_sdid_*.png`

The focused Stata `sdid_event` run uses `vce(placebo) brep(50)` for total CNO4, grouped-age CNO4, and gender-CNO4 panels. Province-CNO4 remains `vce(off)` because placebo inference on that panel did not finish within 30 minutes.

## Main Finding

The simple two-way fixed-effect DiD estimate is positive: high-exposure occupations have about 9 percent higher registered unemployment counts after September 2022 relative to zero-exposure occupations.

However, the event-study pretrends reject parallel trends, and adding CNO4-specific linear trends moves the estimate close to zero. The conservative conclusion is therefore not that AI causally increased unemployment, but that high-exposure occupations show a post-2022 relative divergence that is not yet credibly identified as causal.

## Recommended Next Steps Implemented

The second-stage analysis adds province fixed effects, age/gender heterogeneity, contracts outcomes, continuous exposure models, alternative event dates, a CNO2 SEPE/EPA denominator proxy, and an exposure validation queue.

The updated conclusion is similar but sharper: the positive unemployment-count association survives province-month fixed effects, but unemployment estimates still collapse toward zero in trend-adjusted exposure/rate models. Contracts rise in exposed occupations, so the evidence points more toward reallocation or churn than clean AI-driven job destruction.

## Recommended Next Steps v1

The v1 extension adds CNO2-by-month fixed effects and merges the exposure repo's Anthropic/O*NET match diagnostics into the validation queue.

The CNO2-by-month event study improves the pretrend diagnostic substantially, with pretrend p-value around 0.110, but the post coefficient becomes approximately zero. The current best interpretation is that broad CNO2 occupation-family dynamics explain much of the original positive DiD estimate.

## Recommended Steps Implemented

The latest extension decomposes the baseline DiD by CNO2 family, estimates within-CNO2 family DiD models, checks age 30-39 with CNO2-by-month fixed effects, and applies HonestDiD sensitivity analysis to the original event-study coefficients.

Age 30-39 with CNO2-by-month fixed effects has a near-zero unemployment estimate and a much better pretrend diagnostic. HonestDiD intervals for the original first-year event-study average include zero under both smoothness and relative-magnitude relaxations.

## Synthetic Methods

The latest extension adds focused CNO2 family diagnostics and synthetic DiD / synthetic control estimates for total CNO4, province-CNO4, and age 30-39 CNO4 panels. The requested province-age 30-39-CNO4 panel is not estimable from the current CSV because province and age are separate dimensions rather than a joint cross-tab.

Synthetic estimates are positive in feasible panels, but they should be read as descriptive robustness checks because CNO2-by-month fixed effects still remove the unemployment effect.

## ContDID

The current memo extension adds continuous-treatment DiD using `contdid`. The default R run estimates event-time ACRT paths and dose-response curves for total CNO4 and the three grouped age panels: `<18 to 29`, `30-39`, and `40 to >44`. A province-CNO4 ContDID pass also produced figures and CSVs, but it is opt-in because of runtime.
