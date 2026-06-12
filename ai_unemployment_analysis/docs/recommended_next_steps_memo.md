# Memo: Recommended Next Steps Implemented

## Purpose

The first memo concluded that the baseline DiD estimate was positive but not yet causally credible because event-study pretrends failed and occupation-specific trends erased the effect. This memo documents the recommended next steps that were implemented and what they imply for the research design.

## What Was Implemented

| Recommendation | Status | Output |
|---|---|---|
| Add denominator-adjusted outcome | Implemented as a CNO2-quarter proxy using INE EPA occupied workers | `output/tables/table_next_step_cno2_rate_proxy.csv` |
| Use province disaggregation | Implemented CNO4-province FE and province-month FE | `output/tables/table_next_step_province_fe.csv` |
| Run age and gender heterogeneity | Implemented for gender, age bands, and under-30 aggregate | `output/tables/table_next_step_subgroup_heterogeneity.csv` |
| Treat contracts as a separate outcome | Implemented in timing, subgroup, and province models | multiple `table_next_step_*` outputs |
| Validate high-exposure occupations | Implemented a manual validation queue | `output/tables/table_next_step_exposure_validation_queue.csv` |
| Add natural ChatGPT timing dates | Implemented September, November, December 2022 and January 2023 | `output/tables/table_next_step_timing_continuous_contracts.csv` |
| Add continuous exposure designs | Implemented with and without occupation trends | `output/tables/table_next_step_timing_continuous_contracts.csv` |

Figures:

- `output/figures/figure_05_next_step_subgroup_parados.png`
- `output/figures/figure_06_next_step_robustness_diagnostics.png`

Main script:

- `code/run_next_steps.py`

## 1. Denominator-Adjusted Outcome

The SEPE CSV does not contain employment or labor-force denominators. The `personas` column is empty for the total, age, gender, and province panels. Therefore, a true monthly CNO4 unemployment rate cannot be constructed from the attached file alone.

As a feasible next best step, I downloaded INE EPA table 65134, which reports occupied workers by sex and CNO-11 occupation. This supports a CNO2-quarter proxy:

```text
rate_proxy_gq = SEPE parados_gq / (SEPE parados_gq + EPA ocupados_gq)
```

where `g` is CNO2 and `q` is quarter.

This is not an official unemployment rate. It combines SEPE registered unemployment with EPA occupied employment, but it partially addresses the raw-count problem.

Results:

| Specification | Estimate | SE | Interpretation |
|---|---:|---:|---|
| CNO2 rate proxy, continuous exposure | 0.028 | 0.011 | About +2.9% per 10pp exposure |
| CNO2 rate proxy + CNO2 trends | -0.005 | 0.008 | Approximately zero |

Takeaway: the denominator-adjusted proxy repeats the core identification lesson. The positive association appears in a standard fixed-effect model, but trend adjustment removes it.

## 2. Province Fixed-Effect Design

Using the province panel, I estimated:

```text
log(parados + 1)_{i,p,t} = beta * HighExposure_i * Post_t
                         + CNO4-by-province FE
                         + province-by-month FE
                         + error_{i,p,t}
```

Standard errors are clustered at CNO4 because treatment varies at the occupation level.

Results:

| Outcome | Estimate | SE | Approx. percent effect |
|---|---:|---:|---:|
| `log(parados + 1)` | 0.050 | 0.016 | +5.2% |
| `log(contratos + 1)` | 0.080 | 0.014 | +8.3% |

Takeaway: after absorbing province-month shocks, the high-exposure occupations still show a positive relative post-period increase in registered unemployment counts. But contracts also rise, so the result does not look like simple displacement. It may reflect differential churn, demand, occupational reporting, or broader white-collar/service-sector dynamics.

## 3. Age and Gender Heterogeneity

Gender results:

| Group | Parados estimate | Approx. percent effect |
|---|---:|---:|
| Men | 0.086 | +9.0% |
| Women | 0.069 | +7.1% |

Age results:

| Group | Parados estimate | Approx. percent effect |
|---|---:|---:|
| <18 | 0.049 | +5.0%, not precise |
| 18-24 | 0.079 | +8.3% |
| 25-29 | 0.070 | +7.3% |
| Under 30 aggregate | 0.050 | +5.2%, not precise |
| 30-39 | 0.181 | +19.8% |
| 40-44 | 0.096 | +10.1% |
| >44 | 0.074 | +7.7% |

Takeaway: the strongest unemployment-count heterogeneity is not among the youngest workers; it is in ages 30-39. This differs from the Anthropic paper's emphasis on young-worker hiring margins and suggests the Spain/SEPE outcome is capturing a different margin.

## 4. Contracts as a Separate Outcome

Contracts are positive across many specifications:

- Baseline binary September 2022: +13.1%
- November 2022: +11.3%
- December 2022: +10.6%
- January 2023: +10.0%
- Continuous exposure: +3.3% per 10pp exposure
- Continuous exposure with CNO4 trends: +3.0% per 10pp exposure
- Province FE: +8.3%

Takeaway: contracts are not falling in high-exposure occupations. This is important. A story in which AI simply destroys jobs and raises unemployment is not consistent with the contracts evidence. The pattern looks more like higher turnover, reallocation, changing matching dynamics, or growth in occupations that are also AI-exposed.

## 5. Timing Checks

Changing the event month from September 2022 to dates closer to ChatGPT release does not materially change the binary unemployment-count estimate:

| Event month | Parados estimate |
|---|---:|
| September 2022 | 0.089 |
| November 2022 | 0.091 |
| December 2022 | 0.092 |
| January 2023 | 0.094 |

Takeaway: timing is not the main weakness. The main weakness remains differential trends.

## 6. Continuous Exposure

Continuous exposure uses all occupations and estimates effects per 10 percentage points of exposure.

| Outcome/specification | Estimate |
|---|---:|
| Parados, no occupation trends | 0.027 |
| Parados, CNO4 trends | 0.000 |
| Contracts, no occupation trends | 0.032 |
| Contracts, CNO4 trends | 0.030 |

Takeaway: for unemployment counts, continuous exposure has the same trend sensitivity as the binary design. For contracts, the positive association survives CNO4 trends.

## 7. Exposure Validation Queue

I created a 124-occupation validation queue. It prioritizes:

- top 25 occupations by exposure,
- large high-exposure occupations by mean registered unemployment,
- all remaining high-exposure occupations.

The queue is in:

- `output/tables/table_next_step_exposure_validation_queue.csv`

This is a manual-review artifact. The SEPE CSV does not contain Anthropic nearest-neighbor titles, so the next improvement is to merge in the exposure repository's match diagnostics if available.

## Updated Causal Assessment

The next-step analyses strengthen the descriptive result but do not solve identification.

What got stronger:

- The positive high-versus-zero association survives province-month fixed effects.
- The positive association appears in age/gender subgroups.
- The timing choice is not driving the baseline estimate.
- A denominator-adjusted CNO2 proxy also shows a positive simple fixed-effect estimate.

What still blocks causal interpretation:

- Trend-adjusted specifications continue to collapse toward zero for unemployment.
- The CNO2 rate proxy becomes approximately zero with CNO2-specific trends.
- Contracts rise in high-exposure occupations, which complicates displacement.
- The outcome is still mostly registered unemployment counts, not official unemployment rates.
- Exposure is potential/semantic exposure, not observed Spanish occupation-month AI adoption.

## Recommended Next Research Move

The next best research move is not another binary DiD table. It is a more explicit mechanism design:

1. Keep the province-month FE model as the stronger count-based design.
2. Develop official denominators, ideally EPA or administrative employment by occupation and region.
3. Treat unemployment and contracts jointly, as a reallocation/churn question rather than only job destruction.
4. Prioritize age 30-39 and contracts heterogeneity, since those are where the new evidence is most informative.
5. Merge the CNO4 exposure match diagnostics so the top exposed occupations can be manually validated.

## Bottom Line

After implementing the recommended next steps, the answer is still careful:

> High-AI-exposure occupations in Spain show a positive post-2022 relative divergence in registered unemployment counts, and this survives province-month fixed effects. But unemployment effects remain highly sensitive to occupation-specific trends, while contracts also rise. The evidence supports further investigation of reallocation or churn in exposed occupations, not a clean causal claim that AI penetration increased unemployment.

Sources:

- Anthropic, "Labor market impacts of AI: A new measure and early evidence": https://www.anthropic.com/research/labor-market-impacts
- INE EPA table 65134, occupied workers by sex and occupation: https://www.ine.es/jaxiT3/files/t/csv_bdsc/65134.csv
