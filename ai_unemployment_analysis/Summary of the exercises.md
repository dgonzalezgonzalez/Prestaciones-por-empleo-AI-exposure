# Summary of the exercises

This file summarizes the empirical exercises run so far in the AI exposure and unemployment project and links each exercise to the saved outputs.

Project folder:

- `C:\Users\ngonzalezp\OneDrive - AIREF\Escritorio\Unempolyment_Benefits\ai_unemployment_analysis`

Main scripts:

- `code/run_analysis.py`
- `code/run_next_steps.py`
- `code/run_research_moves.py`
- `code/run_synthetic_methods.py`
- `code/run_continuous_event_studies.py`
- `code/run_binary_event_study_samples.py`
- `code/run_exposure_variation.py`
- `code/stata_empirical_analysis_plan.do`
- `code/stata_produce_all_figures.do`
- `run_all.ps1`

Current Stata-first outputs, once `code/stata_empirical_analysis_plan.do` is run in Stata, are organized under `output/stata`. The most important navigation files are `output/stata/tables/stata_spec_catalog.csv`, `output/stata/tables/stata_twfe_event_study_index.csv`, and `output/stata/tables/stata_sdid_event_index.csv`. Event-study graph filenames include the spec id, for example `twfe_event_total_cno4_ln_parados_spec1.png` and `sdid_event_sdid_total_cno4_ln_parados.png`.

## Data and treatment definitions

Main raw input:

- `data/raw/sepe_cno4_monthly_ai_exposure.csv`

Main processed panel:

- `data/processed/analysis_panel.csv`

Main exposure:

- `observed_exposure_cosine_nearest`

Main binary treatment:

- Treated occupations: CNO4 occupations with exposure above the 75th percentile.
- Zero-exposure controls: CNO4 occupations with exposure equal to zero.
- Middle-exposure occupations are excluded from binary high-versus-zero DiD models.
- Threshold in the main panel: 0.1169.
- Event month: September 2022; post starts after September 2022.

Main outcome variables:

- `ln_parados`: log registered unemployed, `log(parados)`.
- `ln_parados_p1`: `log(parados + 1)`, used when cells can be zero.
- `ln_contratos_p1`: `log(contratos + 1)`.
- `ln_rate_proxy`: CNO2-quarter SEPE/EPA denominator proxy.

Cluster convention:

- Standard errors are clustered at CNO4 unless otherwise noted.
- CNO2-quarter rate proxy clusters at CNO2.

## Specification inventory

| Exercise | Sample | Outcome | Treatment / regressor | Fixed effects | Cluster | Main result | Outputs |
|---|---|---|---|---|---|---|---|
| Descriptives by exposure group | Total-CNO4 monthly panel | `parados`, `ln_parados`, `contratos` | Exposure groups: zero, middle, high | None | None | High-exposure occupations have higher mean exposure and show a post-2022 relative divergence in log unemployment counts. | `output/tables/table_monthly_descriptives_by_exposure_group.csv`, `output/tables/table_exposure_group_summary.csv`, `output/figures/figure_01_mean_log_parados_by_exposure_group.png`, `output/figures/figure_02_indexed_total_log_parados_by_exposure_group.png`, `output/figures/figure_03_exposure_distribution.png` |
| Top-exposure occupation validation | Unique CNO4 occupations | Exposure and occupation title | Exposure rank | None | None | Top exposed occupations are largely expected: programmers, data-entry workers, commercial agents, financial analysts, software analysts, office/statistical/financial clerks. | `output/tables/table_top50_exposed_occupations.csv`, `output/tables/table_top25_exposed_occupations.tex`, `output/tables/table_first50_zero_exposure_occupations.csv` |
| Baseline binary DiD | Total-CNO4, high vs zero | `ln_parados` | `HighExposure_i * Post_t` | CNO4 FE and month FE | CNO4 | 0.089 log points, about +9.3%; statistically significant. | `output/tables/table_regression_specifications.csv`, `output/models/01_summary.txt` |
| Baseline with CNO4 trends | Total-CNO4, high vs zero | `ln_parados` | `HighExposure_i * Post_t` | CNO4 FE, month FE, CNO4 linear trends | CNO4 | -0.003 log points; effect disappears. | `output/tables/table_regression_specifications.csv`, `output/models/02_summary.txt` |
| `log(parados + 1)` robustness | Total-CNO4, high vs zero | `ln_parados_p1` | `HighExposure_i * Post_t` | CNO4 FE and month FE | CNO4 | 0.094 log points, similar to baseline. | `output/tables/table_regression_specifications.csv`, `output/models/03_summary.txt` |
| Weighted baseline | Total-CNO4, high vs zero | `ln_parados` | `HighExposure_i * Post_t` | CNO4 FE and month FE | CNO4 | 0.043 log points; smaller and marginal. | `output/tables/table_regression_specifications.csv`, `output/models/04_summary.txt` |
| Alternative event date | Total-CNO4, high vs zero | `ln_parados` | High exposure post Nov 2022 | CNO4 FE and month FE | CNO4 | 0.091 log points, similar to September timing. | `output/tables/table_regression_specifications.csv`, `output/models/05_summary.txt` |
| Top-decile treatment | Total-CNO4, top decile vs zero | `ln_parados` | `TopDecileExposure_i * Post_t` | CNO4 FE and month FE | CNO4 | 0.080 log points, positive but noisier. | `output/tables/table_regression_specifications.csv`, `output/models/06_summary.txt` |
| Continuous exposure DiD | Total-CNO4, all occupations | `ln_parados` | Exposure per 10pp times post | CNO4 FE and month FE | CNO4 | 0.027 log points per 10pp exposure. | `output/tables/table_regression_specifications.csv`, `output/models/07_summary.txt` |
| Alternative exposure measures | Total-CNO4 | `ln_parados` | Weighted cosine exposure and random-forest exposure | CNO4 FE and month FE | CNO4 | Weighted cosine remains positive; RF exposure positive but changes the control definition. | `output/tables/table_regression_specifications.csv`, `output/models/08_summary.txt`, `output/models/09_summary.txt` |
| Restricted 2022+ and placebo | Total-CNO4, high vs zero | `ln_parados` | Binary post interaction; placebo in pre-period | CNO4 FE and month FE | CNO4 | 2022+ remains positive; pre-period placebo is near zero. | `output/tables/table_regression_specifications.csv`, `output/models/10_summary.txt`, `output/models/11_summary.txt` |
| Contracts outcome | Total-CNO4, high vs zero | `ln_contratos_p1` | `HighExposure_i * Post_t` | CNO4 FE and month FE | CNO4 | 0.123 log points, about +13.1%; contracts rise rather than fall. | `output/tables/table_regression_specifications.csv`, `output/models/12_summary.txt` |
| Original event study | Total-CNO4, high vs zero | `ln_parados` | Treatment interacted with event-month dummies, omitting event month -1 | CNO4 FE and month FE | CNO4 | Pretrend Wald p-value is 2.1e-08; parallel trends fails. | `output/tables/table_event_study_coefficients.csv`, `output/figures/figure_04_event_study_high_vs_zero_sep2022.png`, `output/models/event_study_summary.txt` |
| Province panel | Province-CNO4 | `ln_parados_p1`, `ln_contratos_p1` | `HighExposure_i * Post_t` | CNO4-by-province FE and province-by-month FE | CNO4 | Parados +0.050; contratos +0.080. | `output/tables/table_next_step_province_fe.csv` |
| Gender heterogeneity | Gender-CNO4 panels | `ln_parados`, `ln_contratos_p1` | `HighExposure_i * Post_t` | CNO4 FE and month FE | CNO4 | Men +0.086 and women +0.069 for parados. | `output/tables/table_next_step_subgroup_heterogeneity.csv`, `output/figures/figure_05_next_step_subgroup_parados.png` |
| Age heterogeneity | Age-CNO4 panels | `ln_parados`, `ln_contratos_p1` | `HighExposure_i * Post_t` | CNO4 FE and month FE | CNO4 | Strongest count signal in ages 30-39: +0.181 before CNO2-month controls. | `output/tables/table_next_step_subgroup_heterogeneity.csv`, `output/figures/figure_05_next_step_subgroup_parados.png` |
| Timing, continuous, and contracts extensions | Total-CNO4 | `ln_parados`, `ln_contratos_p1` | Binary post dates and continuous exposure per 10pp | CNO4 FE and month FE; some with CNO4 trends | CNO4 | Binary timing is stable; continuous parados collapses with trends; continuous contracts stay positive with trends. | `output/tables/table_next_step_timing_continuous_contracts.csv`, `output/figures/figure_06_next_step_robustness_diagnostics.png` |
| CNO2-quarter rate proxy | CNO2-quarter panel | `ln_rate_proxy` | Exposure per 10pp times post | CNO2 FE and quarter FE; variant adds CNO2 trends | CNO2 | +0.028 without trends; -0.005 with CNO2 trends. | `output/tables/table_next_step_cno2_rate_proxy.csv`, `data/processed/cno2_sepe_epa_rate_proxy_panel.csv` |
| CNO2-by-month FE | Total-CNO4 | `ln_parados`, `ln_parados_p1`, `ln_contratos_p1` | Binary post or continuous exposure post | CNO4 FE and CNO2-by-month FE | CNO4 | Binary `ln_parados` becomes 0.002; pretrend p-value improves to 0.110. | `output/tables/table_v1_cno2_month_fe_specifications.csv`, `output/tables/table_v1_cno2_month_fe_event_study.csv`, `output/figures/figure_07_v1_cno2_month_fe_event_study.png` |
| CNO2 FWL decomposition | Total-CNO4, high vs zero | `ln_parados` | Baseline binary DiD residual contribution | CNO4 FE and month FE residualization | CNO4 | Original positive coefficient is heavily influenced by CNO2 family dynamics. | `output/tables/table_recommended_cno2_fwl_decomposition.csv`, `output/tables/table_recommended_cno2_family_summary.csv` |
| Within-CNO2 family DiD | CNO2 families with treated and zero occupations | `ln_parados` | `HighExposure_i * Post_t` | CNO4 FE and month FE within family | CNO4 | Mixed and often imprecise; CNO2 59 is positive but small-cluster and not obviously AI-related. | `output/tables/table_recommended_within_cno2_family_did.csv` |
| Age 30-39 with CNO2-by-month FE | Age 30-39 CNO4 panel | `ln_parados`, `ln_parados_p1`, `ln_contratos_p1` | `HighExposure_i * Post_t` | CNO4 FE and CNO2-by-month FE | CNO4 | `ln_parados` estimate is 0.024, p=0.718; event pretrend p-value 0.494. | `output/tables/table_recommended_age3039_cno2_month_fe.csv`, `output/tables/table_recommended_age3039_cno2_month_fe_event_study.csv`, `output/figures/figure_08_recommended_age3039_cno2_month_fe_event_study.png` |
| HonestDiD sensitivity | Original event-study coefficients, local window -12 to +12 | First-year average event effect, months 0 to 12 | Original event-study coefficients | CNO4 FE and month FE from original event study | CNO4 | Standard CI includes zero; HonestDiD smoothness intervals include zero and are wide. | `output/tables/table_recommended_honestdid_original_ci.csv`, `output/tables/table_recommended_honestdid_smoothness.csv`, `output/tables/table_recommended_honestdid_relative_magnitude.csv`, `output/figures/figure_09_recommended_honestdid_smoothness_original.png` |
| Focus CNO2 event studies | CNO2 24, 38, 59 | `ln_parados` | Event-time treatment dummies | CNO4 FE and month FE within family | CNO4 | Useful diagnostics but small-cluster; CNO2 27 skipped because no zero-exposure controls. | `output/tables/table_research_move_focus_cno2_event_coefficients.csv`, `output/tables/table_research_move_focus_cno2_event_summary.csv`, `output/figures/figure_research_move_cno2_24_event_study.png`, `output/figures/figure_research_move_cno2_38_event_study.png`, `output/figures/figure_research_move_cno2_59_event_study.png` |
| Synthetic DiD / synthetic control | Total-CNO4, province-CNO4, age 30-39-CNO4 | `ln_parados` or `ln_parados_p1` | High-exposure treated aggregate vs zero-exposure donor pool | Synthetic-control paths; SDID-style unit/time weights | Not clustered in current implementation | SDID-style ATT: total +0.072, province +0.113, age 30-39 +0.125. | `output/tables/table_synthetic_did_control_results.csv`, `output/tables/table_synthetic_donor_weights.csv`, `output/tables/table_synthetic_paths.csv`, `output/figures/figure_synthetic_total_ocup4d.png`, `output/figures/figure_synthetic_province_ocup4d.png`, `output/figures/figure_synthetic_age3039_ocup4d.png` |
| Continuous-DID event studies | Samples 1-4 | `ln_parados` or `ln_parados_p1` | Exposure per 10pp interacted with event-month dummies | Unit FE and month FE | CNO4 | TWFE continuous-treatment event-study analogue to `contdid`; all feasible samples still show pretrend concerns. | `output/tables/table_continuous_event_study_summary.csv`, `output/tables/table_continuous_event_study_all_samples.csv`, `output/figures/figure_continuous_event_study_sample1_total_ocup4d.png`, `output/figures/figure_continuous_event_study_sample2_province_ocup4d.png`, `output/figures/figure_continuous_event_study_sample3_gender_ocup4d.png`, `output/figures/figure_continuous_event_study_sample4_age_ocup4d.png` |
| Binary event studies for all feasible samples | Samples 1-4 plus age 30-39 | `ln_parados` or `ln_parados_p1` | High exposure vs zero exposure interacted with event-month dummies | Unit FE and month FE | CNO4 | Age 30-39 event study added; province-CNO4 event study added; sample 5 and 6 not available from current CSV. | `output/tables/table_binary_event_study_summary.csv`, `output/tables/table_binary_event_study_all_samples.csv`, `output/figures/figure_binary_event_study_sample1_total_ocup4d.png`, `output/figures/figure_binary_event_study_sample2_province_ocup4d.png`, `output/figures/figure_binary_event_study_sample3_gender_ocup4d.png`, `output/figures/figure_binary_event_study_sample4_age_ocup4d.png`, `output/figures/figure_binary_event_study_sample4_age3039_ocup4d.png` |
| Exposure variation within CNO2 | Unique CNO4 occupations grouped by first two CNO digits | Exposure measure only | `observed_exposure_cosine_nearest` / `exposure_nearest` | None | None | 62 CNO2 families; 22 have within-family exposure range above 0.25; 25 contain both zero and high-exposure occupations. | `output/tables/table_exposure_variation_within_cno2.csv`, `output/figures/figure_exposure_variation_within_cno2.png` |

## Sample-combination status

| Sample | Definition | Binary event study | Continuous-DID event study | Synthetic method |
|---|---|---|---|---|
| sample1 | Total-CNO4 | Estimated | Estimated | Estimated |
| sample2 | Province-CNO4 | Estimated | Estimated | Estimated |
| sample3 | Gender-CNO4 | Estimated | Estimated | Not run as synthetic method |
| sample4 | Age-CNO4 | Estimated | Estimated | Age 30-39 estimated |
| sample5 | Province-age-CNO4 | Not available | Not available | Not available |
| sample6 | Gender-age-CNO4 | Not available | Not available | Not available |

The current SEPE CSV has separate dimensions for `province`, `age`, and `gender`, but it does not include joint province-by-age-CNO4 or gender-by-age-CNO4 cells. I therefore did not impute sample5 or sample6.

## Figure inventory

| Figure | Location | What it shows |
|---|---|---|
| Mean log parados by exposure group | `output/figures/figure_01_mean_log_parados_by_exposure_group.png` | Descriptive evolution of mean `ln_parados` by zero/middle/high exposure. |
| Indexed total log parados | `output/figures/figure_02_indexed_total_log_parados_by_exposure_group.png` | Group totals indexed to the Jan-Aug 2022 pre-period mean. |
| Exposure distribution | `output/figures/figure_03_exposure_distribution.png` | Distribution of CNO4 AI exposure with p75 threshold. |
| Original event study | `output/figures/figure_04_event_study_high_vs_zero_sep2022.png` | Baseline high-vs-zero event study. |
| Subgroup parados | `output/figures/figure_05_next_step_subgroup_parados.png` | Gender and age subgroup DiD estimates. |
| Robustness diagnostics | `output/figures/figure_06_next_step_robustness_diagnostics.png` | Province FE, rate proxy, continuous exposure, and CNO2-month FE estimates. |
| CNO2-month FE event study | `output/figures/figure_07_v1_cno2_month_fe_event_study.png` | Event study with CNO4 FE and CNO2-by-month FE. |
| Age 30-39 CNO2-month FE event study | `output/figures/figure_08_recommended_age3039_cno2_month_fe_event_study.png` | Age 30-39 event study with CNO2-by-month FE. |
| HonestDiD sensitivity | `output/figures/figure_09_recommended_honestdid_smoothness_original.png` | Smoothness sensitivity intervals for the original event study. |
| Binary event study sample1 | `output/figures/figure_binary_event_study_sample1_total_ocup4d.png` | Total-CNO4 binary event study. |
| Binary event study sample2 | `output/figures/figure_binary_event_study_sample2_province_ocup4d.png` | Province-CNO4 binary event study. |
| Binary event study sample3 | `output/figures/figure_binary_event_study_sample3_gender_ocup4d.png` | Gender-CNO4 binary event study. |
| Binary event study sample4 | `output/figures/figure_binary_event_study_sample4_age_ocup4d.png` | Age-CNO4 binary event study. |
| Binary event study age 30-39 | `output/figures/figure_binary_event_study_sample4_age3039_ocup4d.png` | Age 30-39 CNO4 binary event study. |
| Continuous event study sample1 | `output/figures/figure_continuous_event_study_sample1_total_ocup4d.png` | Total-CNO4 continuous exposure event study. |
| Continuous event study sample2 | `output/figures/figure_continuous_event_study_sample2_province_ocup4d.png` | Province-CNO4 continuous exposure event study. |
| Continuous event study sample3 | `output/figures/figure_continuous_event_study_sample3_gender_ocup4d.png` | Gender-CNO4 continuous exposure event study. |
| Continuous event study sample4 | `output/figures/figure_continuous_event_study_sample4_age_ocup4d.png` | Age-CNO4 continuous exposure event study. |
| Focus CNO2 24 event study | `output/figures/figure_research_move_cno2_24_event_study.png` | Within-family event study for STEM professionals. |
| Focus CNO2 38 event study | `output/figures/figure_research_move_cno2_38_event_study.png` | Within-family event study for ICT technicians. |
| Focus CNO2 59 event study | `output/figures/figure_research_move_cno2_59_event_study.png` | Within-family event study for protection/security services. |
| Synthetic total CNO4 | `output/figures/figure_synthetic_total_ocup4d.png` | Treated path vs synthetic and SDID controls. |
| Synthetic province CNO4 | `output/figures/figure_synthetic_province_ocup4d.png` | Province-CNO4 treated path vs synthetic and SDID controls. |
| Synthetic age 30-39 CNO4 | `output/figures/figure_synthetic_age3039_ocup4d.png` | Age 30-39 treated path vs synthetic and SDID controls. |
| Exposure variation within CNO2 | `output/figures/figure_exposure_variation_within_cno2.png` | Within-CNO2 exposure range by two-digit occupation family. |

## Key table inventory

Long event-study coefficient tables are not copied into this memo. The key files are:

- `output/tables/table_regression_specifications.csv`
- `output/tables/table_event_study_coefficients.csv`
- `output/tables/table_next_step_subgroup_heterogeneity.csv`
- `output/tables/table_next_step_province_fe.csv`
- `output/tables/table_next_step_cno2_rate_proxy.csv`
- `output/tables/table_next_step_timing_continuous_contracts.csv`
- `output/tables/table_v1_cno2_month_fe_specifications.csv`
- `output/tables/table_v1_cno2_month_fe_event_study.csv`
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
- `output/tables/table_research_move_focus_cno2_event_coefficients.csv`
- `output/tables/table_synthetic_did_control_results.csv`
- `output/tables/table_synthetic_donor_weights.csv`
- `output/tables/table_synthetic_paths.csv`
- `output/tables/table_continuous_event_study_summary.csv`
- `output/tables/table_continuous_event_study_all_samples.csv`
- `output/tables/table_binary_event_study_summary.csv`
- `output/tables/table_binary_event_study_all_samples.csv`
- `output/tables/table_binary_event_study_unavailable_samples.csv`
- `output/tables/table_continuous_event_study_unavailable_samples.csv`
- `output/tables/table_exposure_variation_within_cno2.csv`

## Continuous-DID event studies

I used a continuous-treatment event-study specification inspired by `bcallaway11/contdid`:

- Source checked: `https://github.com/bcallaway11/contdid`
- Regressor: exposure per 10 percentage points interacted with event-month dummies.
- Omitted reference month: event month -1.
- Fixed effects: sample unit FE and month FE.
- Clustering: CNO4.

Because all units share the same event month, the implemented figure is the TWFE continuous-treatment event-study analogue rather than a staggered-adoption continuous-treatment estimator.

Summary:

| Sample | Outcome | Unit FE | Pretrend p-value | Output |
|---|---|---|---:|---|
| sample1 total-CNO4 | `ln_parados` | CNO4 | 8.34e-07 | `output/figures/figure_continuous_event_study_sample1_total_ocup4d.png` |
| sample2 province-CNO4 | `ln_parados_p1` | province-CNO4 | 2.40e-12 | `output/figures/figure_continuous_event_study_sample2_province_ocup4d.png` |
| sample3 gender-CNO4 | `ln_parados` | gender-CNO4 | 0.000279 | `output/figures/figure_continuous_event_study_sample3_gender_ocup4d.png` |
| sample4 age-CNO4 | `ln_parados` | age-CNO4 | 4.43e-05 | `output/figures/figure_continuous_event_study_sample4_age_ocup4d.png` |

## Binary event studies for all feasible samples

Summary:

| Sample | Outcome | Unit FE | Pretrend p-value | Output |
|---|---|---|---:|---|
| sample1 total-CNO4 | `ln_parados` | CNO4 | 2.10e-08 | `output/figures/figure_binary_event_study_sample1_total_ocup4d.png` |
| sample2 province-CNO4 | `ln_parados_p1` | province-CNO4 | 6.83e-11 | `output/figures/figure_binary_event_study_sample2_province_ocup4d.png` |
| sample3 gender-CNO4 | `ln_parados` | gender-CNO4 | 7.99e-05 | `output/figures/figure_binary_event_study_sample3_gender_ocup4d.png` |
| sample4 age-CNO4 | `ln_parados` | age-CNO4 | 6.25e-05 | `output/figures/figure_binary_event_study_sample4_age_ocup4d.png` |
| age 30-39 CNO4 | `ln_parados` | CNO4 | 0.0278 | `output/figures/figure_binary_event_study_sample4_age3039_ocup4d.png` |

This includes the requested replication of `figure_04_event_study_high_vs_zero_sep2022.png` for:

- age group 30-39 using the age-CNO4 data;
- province-CNO4 data.

## Variation of exposure within CNO2

Output:

- `output/tables/table_exposure_variation_within_cno2.csv`
- `output/figures/figure_exposure_variation_within_cno2.png`

The exposure measure varies substantially within many two-digit occupation families:

- Number of CNO2 families: 62.
- Families with at least two CNO4 occupations: 55.
- Mean within-CNO2 exposure range: 0.174.
- Median within-CNO2 exposure range: 0.084.
- Families with within-CNO2 exposure range above 0.25: 22.
- Families containing both zero-exposure and high-exposure CNO4 occupations: 25.

Largest within-CNO2 ranges:

| CNO2 | CNO4 count | Min | Max | Range | Highest-exposure CNO4 | Lowest-exposure CNO4 |
|---|---:|---:|---:|---:|---|---|
| 38 | 8 | 0.000 | 0.745 | 0.745 | 3820, Programadores informaticos | 3831, Tecnicos de grabacion audiovisual |
| 35 | 10 | 0.000 | 0.628 | 0.628 | 3510, Agentes y representantes comerciales | 3533, Agentes o intermediarios en contratacion de mano de obra |
| 26 | 13 | 0.000 | 0.572 | 0.572 | 2613, Analistas financieros | 2623, Especialistas de la administracion publica |
| 41 | 6 | 0.000 | 0.510 | 0.510 | 4113, Empleados de oficina de servicios estadisticos, financieros y bancarios | 4121, Empleados de control de abastecimientos e inventario |
| 36 | 10 | 0.000 | 0.453 | 0.453 | 3613, Asistentes de direccion y administrativos | 3612, Asistentes juridico-legales |

Interpretation:

- The CNO2-by-month FE design does not mechanically remove all treatment variation.
- It relies on meaningful within-CNO2 variation in exposure.
- However, in some substantively important CNO2 families, zero-exposure controls are sparse or absent, so within-family estimates should be treated as diagnostics rather than definitive causal estimates.

## Current bottom line

The project now has three layers of evidence:

1. Simple high-versus-zero DiD estimates are positive for registered unemployment counts.
2. Event-study pretrends and trend-adjusted models show that the simple positive estimate is not causally credible.
3. CNO2-by-month FE models improve pretrend diagnostics but remove the unemployment effect.

The best current interpretation remains:

> High-AI-exposure occupations show a positive post-2022 relative divergence in registered unemployment counts, but the evidence does not yet support a causal claim that AI penetration increased unemployment. The most plausible current explanation is broader CNO2 occupation-family dynamics, possibly related to reallocation/churn rather than direct displacement.
