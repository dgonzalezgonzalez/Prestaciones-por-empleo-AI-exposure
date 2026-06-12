# Referee 2 Report

**Mode:** code audit  
**Project:** `ai_unemployment_analysis`  
**Date:** 2026-06-03  
**Verdict:** Major revision for causal interpretation; replication package passes core implementation checks.

## Summary

The project builds a CNO4-month panel from the SEPE occupation CSV, defines high AI exposure as `observed_exposure_cosine_nearest > p75` and controls as zero exposure, then estimates two-way fixed-effect DiD and event-study specifications clustered at CNO4.

The core implementation replicates. A separate Frisch-Waugh-Lovell residualization reproduces the main DiD coefficient and clustered standard error exactly up to numerical precision. A pure Node.js parser independently verifies the processed panel dimensions and treatment threshold.

The main issue is econometric, not computational: the positive post coefficient is not robust to occupation-specific trends and the event-study pretrend test strongly rejects parallel pretrends.

## Audit 1: Code Audit

Files inspected or created:

- `code/run_analysis.py`
- `code/replication/referee2_replicate_core.py`
- `code/replication/referee2_replicate_node_counts.js`
- `data/processed/analysis_panel.csv`
- `output/tables/table_regression_specifications.csv`
- `output/tables/table_event_study_coefficients.csv`

Findings:

- The panel filter matches the preliminary Stata logic: CNO4 length equals 4, `dimension == total`, `category == Total`, and `gender == Total`.
- The main p75 threshold is computed on the filtered occupation-month panel, matching the preliminary do-file behavior. The resulting threshold is 0.1169.
- The main comparison is high exposure versus zero exposure. Middle-exposure occupations are excluded from the binary DiD.
- `log(parados)` drops zero and missing unemployment-count rows; `log(parados + 1)` is included as a robustness specification.

## Audit 2: Cross-Implementation Replication

Core FWL replication:

- Main table beta: 0.0890948456630706
- FWL beta: 0.0890948456630470
- Absolute difference: 2.36e-14
- Main table clustered SE: 0.0206712519368672
- Manual FWL clustered SE: 0.0206712519368673
- Absolute SE difference: 1.28e-16

Node data check:

- Rows: 31,626
- Occupations: 502
- Periods: 63
- p75 threshold: 0.1169
- High rows: 7,812
- Zero rows: 15,939

Limitation:

- R and Stata were not available on PATH in this environment. Cross-language replication is therefore partial: Node.js verifies the data construction, while the regression is independently replicated in Python/numpy without statsmodels formula estimation.

## Audit 3: Directory and Replication Package

The project is organized into:

- `data/raw/`: copied raw SEPE CSV
- `data/processed/`: constructed panel and occupation exposure table
- `code/`: main analysis driver
- `code/replication/`: referee replication scripts
- `output/figures/`: descriptives and event-study figures
- `output/tables/`: CSV and LaTeX tables
- `output/models/`: text model summaries
- `docs/`: memo and project notes
- `correspondence/blindspot/` and `correspondence/referee2/`: audit reports

The project is reproducible with the bundled Python runtime after installing `matplotlib`, `statsmodels`, `scipy`, and `jinja2`.

## Audit 4: Output Automation

The main driver regenerates:

- Processed panel
- Occupation exposure tables
- Exposure group descriptives
- Monthly descriptive series
- Four figures
- Regression specification table
- Event-study coefficient table
- Model summary text files

The replication scripts write their own outputs:

- `output/tables/referee2_core_replication.csv`
- `output/tables/referee2_node_data_check.json`

## Audit 5: Econometrics Audit

Major concerns:

- Parallel trends are not credible in the current main design. The event-study pretrend Wald p-value is about 2.1e-08.
- The main coefficient is +0.089 log points, but adding CNO4-specific linear trends changes it to -0.003 log points.
- The outcome is an unemployment count by prior occupation, not an unemployment rate. Without employment/labor-force denominators, differential occupation size changes remain a concern.
- The exposure variable is a time-invariant semantic exposure measure, not observed occupation-month AI adoption in Spain. The design estimates differential post-2022 changes by exposure, not literal measured penetration.
- The contracts outcome is positive, which does not fit a simple displacement-only mechanism.

Minor concerns:

- The treatment month is September 2022, following the preliminary do-file. Since ChatGPT was released on November 30, 2022, the paper should present September 2022 as a design choice and keep November/December 2022 as a robustness date.
- The RF exposure measure has no zero-exposure occupations, so RF robustness necessarily changes the control group to the bottom quartile.
- The event-study figure should be the central diagnostic figure, not an appendix-only robustness check.

## Questions for Authors

- Can the SEPE data be combined with occupation employment stocks to estimate unemployment rates rather than counts?
- Are age-specific panels reliable enough to test young-worker or entrant effects, closer to Anthropic's hiring margin?
- Can province-month fixed effects be used with province-by-occupation panels to absorb regional labor-market shocks?
- Should highly exposed occupations be manually validated against CNO descriptions and Anthropic nearest matches before causal estimation?

## Prioritized Recommendations

1. Do not claim that AI penetration causally increased unemployment from the main DiD alone.
2. Lead with descriptives and the event study; present the positive DiD as a fragile association.
3. Add occupation-specific trend results to the main table.
4. Develop denominators for unemployment rates or use contracts/job starts as a separate outcome family.
5. Add subgroup analyses by age and gender if the SEPE disaggregations are sufficiently complete.
