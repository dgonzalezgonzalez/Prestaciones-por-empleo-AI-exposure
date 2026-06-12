# Memo: Next Research Moves and Synthetic DiD / Synthetic Control

## Purpose

This memo implements the next research moves from the prior memo and adds synthetic estimators for the requested panels:

1. Total CNO4 panel.
2. Province-CNO4 panel.
3. Age 30-39 CNO4 panel.
4. Province-age 30-39-CNO4 panel.

The synthetic estimators follow the synthetic-control and synthetic-difference-in-differences setup described in the Python causality handbook, with simplex donor weights and SDID-style unit/time weights. I used a custom implementation instead of calling `pysynthdid` directly because the province-CNO4 donor pool is large and required explicit donor screening for tractability.

Sources:

- `pysynthdid`: https://github.com/MasaAsami/pysynthdid
- Synthetic DiD handbook chapter: https://matheusfacure.github.io/python-causality-handbook/25-Synthetic-Diff-in-Diff.html

## New Files

Scripts:

- `code/run_research_moves.py`
- `code/run_synthetic_methods.py`

Research-move outputs:

- `output/tables/table_research_move_focus_cno2_summary.csv`
- `output/tables/table_research_move_focus_cno2_occupation_profiles.csv`
- `output/tables/table_research_move_focus_cno2_event_summary.csv`
- `output/tables/table_research_move_focus_cno2_event_coefficients.csv`
- `output/tables/table_research_move_denominator_inventory.csv`

Synthetic outputs:

- `output/tables/table_synthetic_did_control_results.csv`
- `output/tables/table_synthetic_donor_weights.csv`
- `output/tables/table_synthetic_paths.csv`
- `output/figures/figure_synthetic_total_ocup4d.png`
- `output/figures/figure_synthetic_province_ocup4d.png`
- `output/figures/figure_synthetic_age3039_ocup4d.png`

## 1. CNO2 Family Investigation

I focused on CNO2 families 27, 38, 24, and 59, as recommended after the FWL decomposition.

| CNO2 | Family meaning | Occupations | Treated | Zero exposure | Main note |
|---|---|---:|---:|---:|---|
| 27 | IT professionals | 8 | 8 | 0 | Important contributor, but no zero-exposure within-family controls |
| 38 | ICT technicians | 8 | 5 | 2 | Estimable but very small control group |
| 24 | STEM professionals | 42 | 6 | 14 | Best within-family comparison among focus groups |
| 59 | Protection/security services | 12 | 3 | 8 | Positive within-family result, but not an obvious AI mechanism |

Within-family event studies:

- CNO2 27 is skipped because all observed occupations in the focus comparison are treated; there are no zero-exposure controls.
- CNO2 38, 24, and 59 are estimable, but the cluster counts are small.

Interpretation:

- The focus-family investigation confirms why the CNO2-by-month FE model is so revealing: some of the original baseline contribution comes from occupation families that do not contain comparable zero-exposure controls.
- CNO2 24 is the most credible family for within-family follow-up because it has both treated and zero-exposure occupations in meaningful numbers.
- CNO2 59 should be inspected manually before substantive interpretation because it contributes positively but is less obviously connected to AI exposure.

## 2. Match Validation Inside Focus Families

The occupation profile table merges the Anthropic/O*NET match diagnostics:

- nearest Anthropic title,
- nearest Anthropic observed exposure,
- cosine similarity,
- weighted-match counts,
- review flag.

Output:

- `output/tables/table_research_move_focus_cno2_occupation_profiles.csv`

Interpretation:

- The IT and ICT mappings remain substantively plausible.
- The next manual validation should prioritize CNO2 24 and 59 because those families are where within-family comparisons are possible and potentially interpretable.

## 3. Denominator Search

I wrote a denominator inventory:

- `output/tables/table_research_move_denominator_inventory.csv`

Findings:

- INE EPA table 65134 supports the existing CNO2-quarter national denominator proxy.
- Public table search did not produce a direct CNO4-month or province-age-CNO4 denominator.
- EPA microdata remains the most plausible route for age/region/occupation denominators, but would require a separate microdata extraction and likely aggregation to CNO2/CNO3 rather than CNO4.
- Administrative employment stocks would be the best monthly denominator if accessible internally.

## 4. Synthetic DiD and Synthetic Control

Treatment and donors:

- Treated aggregate: high-exposure occupations, `observed_exposure_cosine_nearest > p75`.
- Donor pool: zero-exposure occupations.
- Event month: September 2022.
- Pre-period: months through September 2022.
- Post-period: months after September 2022.

For province-CNO4, the donor pool is large. The script screens controls by pre-period distance to the treated aggregate, then solves the synthetic weights on the selected donor pool. The selected donor weights are saved in:

- `output/tables/table_synthetic_donor_weights.csv`

Results:

| Design | SDID-style ATT | Synthetic control ATT | Uniform DID | SDID approx. percent |
|---|---:|---:|---:|---:|
| Total CNO4 | 0.072 | 0.099 | 0.091 | +7.4% |
| Province-CNO4 | 0.113 | 0.110 | 0.052 | +11.9% |
| Age 30-39 CNO4 | 0.125 | 0.092 | 0.169 | +13.3% |
| Province-age 30-39-CNO4 | Not available | Not available | Not available | Not available |

Interpretation:

- The synthetic estimates are positive for all feasible panels.
- However, synthetic control achieves near-exact pre-period fit in several designs because the donor pool is rich relative to the number of pre-periods. This is useful descriptively but creates overfitting risk.
- These synthetic estimates therefore do not overturn the CNO2-by-month FE result. They show that one can construct zero-exposure synthetic donor aggregates with similar pre-period paths and positive post gaps, but those gaps may still reflect CNO2 family dynamics.

## 5. Province-Age 30-39-CNO4 Panel

This requested panel is not estimable from the current SEPE CSV.

Available dimensions are:

- `age`
- `gender`
- `geographic_mobility`
- `province`
- `total`

Age and province are separate dimensions, not a joint province-by-age cross-tab. Constructing a province-age 30-39-CNO4 panel would require a new SEPE extract with joint province and age categories. I did not impute or approximate those cells because that would create a synthetic data layer unrelated to the requested causal design.

## Bottom Line

The synthetic methods add a useful descriptive robustness check: the positive post-2022 gap appears under SDID-style and synthetic-control weighting in the feasible total, province, and age 30-39 CNO4 panels.

But the core identification conclusion remains:

> The positive association is not robust to absorbing CNO2-by-month shocks. Synthetic weighting can reproduce a positive post gap, but it does not prove the gap is caused by AI exposure rather than broader occupation-family dynamics.

## Recommended Next Step

The next high-value step is to obtain or construct true denominators:

1. Try EPA microdata for CNO2/CNO3 by age and region.
2. Search internal administrative employment stocks by CNO4 or CNO2.
3. Request the SEPE joint province-by-age-by-CNO4 extract if possible.
4. Re-run the CNO2-by-month FE and synthetic estimators on rate outcomes, not only registered-unemployment counts.
