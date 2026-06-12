# Memo: Recommended Next Steps v1

## Purpose

This memo documents the v1 extensions requested after the first round of next-step analysis. I implemented recommended next steps 1, 3, 4, and 5, and added a new specification with CNO2-by-year-month fixed effects to test whether broad two-digit occupation shocks explain the original event-study pretrend problem.

## Implemented Outputs

| Output | File |
|---|---|
| CNO2-by-month FE specifications | `output/tables/table_v1_cno2_month_fe_specifications.csv` |
| CNO2-by-month FE event study | `output/tables/table_v1_cno2_month_fe_event_study.csv` |
| CNO2-by-month FE event-study figure | `output/figures/figure_07_v1_cno2_month_fe_event_study.png` |
| Exposure match validation queue | `output/tables/table_v1_exposure_match_validation_queue.csv` |
| CNO2 SEPE/EPA denominator proxy | `output/tables/table_next_step_cno2_rate_proxy.csv` |
| Age/gender and contracts heterogeneity | `output/tables/table_next_step_subgroup_heterogeneity.csv` |
| Timing, continuous exposure, contracts | `output/tables/table_next_step_timing_continuous_contracts.csv` |
| Province FE count-based design | `output/tables/table_next_step_province_fe.csv` |

Main code:

- `code/run_next_steps.py`

## Step 1: Denominator-Adjusted Outcome and Stronger Count Design

The SEPE data still do not contain employment or labor-force denominators. The `personas` column is empty in the relevant total, gender, age, and province panels. Therefore, a true CNO4-month unemployment rate cannot be constructed from the attached SEPE file alone.

I implemented the closest feasible denominator-adjusted outcome using INE EPA table 65134. The constructed CNO2-quarter proxy is:

```text
rate_proxy_gq = SEPE parados_gq / (SEPE parados_gq + EPA ocupados_gq)
```

where `g` is CNO2 and `q` is quarter. This is not an official unemployment rate because the numerator is SEPE registered unemployment and the denominator uses EPA occupied workers, but it reduces the raw-count concern.

Results:

| Specification | Estimate | SE | Result |
|---|---:|---:|---|
| CNO2 rate proxy, continuous exposure | 0.028 | 0.011 | Positive |
| CNO2 rate proxy + CNO2 trends | -0.005 | 0.008 | Approximately zero |

I also kept the stronger count-based province design:

| Outcome | Estimate | SE | Approx. effect |
|---|---:|---:|---:|
| `log(parados + 1)` with CNO4-province FE and province-month FE | 0.050 | 0.016 | +5.2% |
| `log(contratos + 1)` with CNO4-province FE and province-month FE | 0.080 | 0.014 | +8.3% |

Interpretation: province-month controls preserve a positive count association, but denominator/trend-adjusted models remain much weaker.

## CNO2-by-Year-Month Fixed Effects

I estimated the key new specification:

```text
log(parados_it) = beta * HighExposure_i * Post_t
                + CNO4 FE
                + CNO2-by-month FE
                + error_it
```

This compares high-exposure and zero-exposure CNO4 occupations only against occupations in the same CNO2 group in the same month. It absorbs broad monthly shocks to two-digit occupation families.

Results:

| Specification | Estimate | SE | p-value |
|---|---:|---:|---:|
| Binary high-vs-zero, `log(parados)` | 0.002 | 0.044 | 0.966 |
| Binary high-vs-zero, `log(parados + 1)` | 0.010 | 0.044 | 0.823 |
| Binary high-vs-zero, `log(contratos + 1)` | 0.040 | 0.044 | 0.357 |
| Continuous exposure, `log(parados)` | -0.001 | 0.011 | 0.915 |
| Continuous exposure, `log(contratos + 1)` | 0.013 | 0.014 | 0.355 |

Event-study pretrend:

- Original event-study pretrend p-value: about 2.1e-08.
- CNO2-by-month FE event-study pretrend p-value: 0.110.

Interpretation: the CNO2-by-month fixed effects largely solve the pretrend problem statistically, but they also remove the estimated post effect. That is a strong signal that the original positive coefficient was driven by broad CNO2-level occupation-family dynamics rather than within-CNO2 AI exposure variation.

## Step 3: Unemployment and Contracts Jointly

I treated unemployment and contracts as paired outcomes rather than interpreting contracts as a side robustness check.

Patterns:

- Baseline contracts were positive, around +13.1%.
- Province FE contracts remained positive, around +8.3%.
- Continuous contracts remained positive with CNO4 trends, around +3.0% per 10pp exposure.
- CNO2-by-month FE contracts became smaller and statistically insignificant, around +4.0% in the binary design and +1.3% in the continuous design.

Interpretation: high-exposure occupations do not show a clean "unemployment up, hiring down" displacement pattern. When broader occupation-family shocks are absorbed, both unemployment and contracts effects become weak. The better mechanism framing is reallocation/churn or occupation-family labor-market dynamics, not direct AI job destruction.

## Step 4: Prioritize Age 30-39 and Contracts Heterogeneity

The age/gender heterogeneity remains informative:

| Group | `log(parados)` estimate | Approx. effect |
|---|---:|---:|
| Men | 0.086 | +9.0% |
| Women | 0.069 | +7.1% |
| 18-24 | 0.079 | +8.3% |
| 25-29 | 0.070 | +7.3% |
| Under 30 aggregate | 0.050 | +5.2%, not precise |
| 30-39 | 0.181 | +19.8% |
| 40-44 | 0.096 | +10.1% |
| >44 | 0.074 | +7.7% |

The strongest unemployment-count signal is ages 30-39, not younger workers. This matters because the Anthropic paper emphasizes young-worker hiring margins; the Spanish SEPE result appears to be on a different margin.

Contracts are positive across most subgroup models, including age 30-39. This again points away from a simple "AI reduces demand" mechanism.

## Step 5: Merge Exposure Match Diagnostics

I downloaded and merged the exposure repository's match diagnostics:

- `spanish_occupation_matches_cosine_nearest.csv`
- `spanish_occupation_matches_cosine_weighted.csv`

The v1 validation queue now includes:

- CNO4 occupation title,
- nearest Anthropic/O*NET occupation title,
- Anthropic occupation code,
- Anthropic observed exposure,
- cosine similarity,
- number of weighted-match rows,
- manual review flag.

Top examples:

| CNO4 | Spanish occupation | Nearest Anthropic title | Cosine |
|---|---|---|---:|
| 3820 | Programadores informáticos | Computer Programmers | 0.775 |
| 4301 | Grabadores de datos | Data Entry Keyers | 0.658 |
| 3510 | Agentes y representantes comerciales | Sales Representatives, Wholesale and Manufacturing | 0.697 |
| 2613 | Analistas financieros | Financial and Investment Analysts | 0.824 |
| 2719 | Analistas y diseñadores de software y multimedia | Software Quality Assurance Analysts and Testers | 0.701 |
| 3812 | Técnicos en asistencia al usuario de tecnologías de la información | Computer User Support Specialists | 0.773 |

Validation status:

- 123 high-exposure queue rows are flagged `ok`.
- 1 row is flagged `low_similarity`.

Interpretation: the top-exposure mappings are mostly plausible and now auditable. The next manual step is not to guess whether the exposure measure "looks right"; it is to review the v1 queue row by row and flag questionable matches.

## Updated Identification Assessment

The CNO2-by-month fixed-effect model is the most important v1 result.

What improved:

- Pretrend diagnostic improves substantially: p-value moves from near zero to 0.110.
- The specification absorbs broad occupation-family-by-month shocks.
- It provides a cleaner within-CNO2 comparison.

What changed:

- The unemployment effect becomes essentially zero.
- The continuous exposure effect becomes essentially zero.
- Contracts also become statistically weak.

This strongly suggests that the original post-2022 divergence is not robust to comparing occupations within the same two-digit occupation family and month.

## Bottom Line

The v1 answer is sharper than the earlier memo:

> Adding CNO2-by-month fixed effects largely solves the event-study pretrend problem, but it also eliminates the estimated unemployment effect. The evidence no longer supports a causal claim that AI penetration increased unemployment. The remaining positive associations are best interpreted as broad occupation-family dynamics, not within-family AI exposure effects.

## Recommended Next Move

The project should now pivot from "does AI exposure raise unemployment?" to a narrower diagnostic question:

> Which CNO2 occupation families generated the original divergence, and are those family-level shocks plausibly AI-related?

Concrete next tasks:

1. Decompose the baseline DiD by CNO2 family.
2. Compare high-exposure versus zero-exposure occupations within each large CNO2 family.
3. Use the exposure match queue to manually validate the top-exposed occupations before interpreting any family-specific effect.
4. Seek true CNO4 or CNO2 employment denominators from EPA microdata or administrative employment records.

Sources:

- Anthropic labor-market impact framing: https://www.anthropic.com/research/labor-market-impacts
- Exposure match diagnostics repo: https://github.com/dgonzalezgonzalez/Prestaciones-por-empleo-AI-exposure
- INE EPA table 65134: https://www.ine.es/jaxiT3/files/t/csv_bdsc/65134.csv
