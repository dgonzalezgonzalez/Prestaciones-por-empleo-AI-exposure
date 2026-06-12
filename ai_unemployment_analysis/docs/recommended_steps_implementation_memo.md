# Memo: Recommended Steps Implementation

## Purpose

This memo documents the latest implementation round:

1. Implement the recommended steps from `recommended_next_steps_memo_v1.md`.
2. Check the age 30-39 specification with occupation-family-by-year-month fixed effects.
3. Apply HonestDiD sensitivity analysis to the original event-study graph.

## New Outputs

| Output | File |
|---|---|
| CNO2 FWL decomposition of baseline DiD | `output/tables/table_recommended_cno2_fwl_decomposition.csv` |
| CNO2 family summary | `output/tables/table_recommended_cno2_family_summary.csv` |
| Within-CNO2 family DiD estimates | `output/tables/table_recommended_within_cno2_family_did.csv` |
| Age 30-39 CNO2-by-month FE estimates | `output/tables/table_recommended_age3039_cno2_month_fe.csv` |
| Age 30-39 CNO2-by-month FE event study | `output/tables/table_recommended_age3039_cno2_month_fe_event_study.csv` |
| Age 30-39 event-study figure | `output/figures/figure_08_recommended_age3039_cno2_month_fe_event_study.png` |
| HonestDiD original event coefficients | `output/tables/table_recommended_honestdid_original_event_coefficients.csv` |
| HonestDiD covariance matrix | `output/tables/table_recommended_honestdid_original_event_covariance.csv` |
| HonestDiD original CI | `output/tables/table_recommended_honestdid_original_ci.csv` |
| HonestDiD smoothness sensitivity | `output/tables/table_recommended_honestdid_smoothness.csv` |
| HonestDiD relative-magnitude sensitivity | `output/tables/table_recommended_honestdid_relative_magnitude.csv` |
| HonestDiD smoothness figure | `output/figures/figure_09_recommended_honestdid_smoothness_original.png` |

Main code:

- `code/run_next_steps.py`

## 1. CNO2 Decomposition of the Baseline Effect

I decomposed the original baseline DiD coefficient using Frisch-Waugh-Lovell residuals from the model with CNO4 and month fixed effects. This shows which CNO2 families contribute to the pooled positive baseline coefficient.

Top positive contributors:

| CNO2 | Family | Contribution to beta | Share of baseline beta |
|---|---|---:|---:|
| 27 | Profesionales de las tecnologías de la información | 0.024 | 27.3% |
| 38 | Técnicos de las tecnologías de la información y las comunicaciones | 0.013 | 14.2% |
| 81 | Operadores de instalaciones y maquinaria fijas | 0.010 | 11.4% |
| 24 | Profesionales de ciencias físicas, químicas, matemáticas e ingeniería | 0.007 | 7.5% |
| 59 | Trabajadores de servicios de protección y seguridad | 0.006 | 6.7% |

Interpretation:

- The original pooled coefficient is heavily influenced by CNO2 family-level dynamics.
- IT professional and ICT technician families are important contributors, which is substantively plausible.
- Some contributing families do not have both treated and zero-exposure occupations, so their FWL contribution is not the same as a clean within-family treatment comparison. It is a decomposition of the pooled residualized coefficient.

## 2. Within-CNO2 Family Comparisons

I estimated separate high-versus-zero DiD models within each CNO2 family with enough treated and control occupations.

Largest positive within-family estimates:

| CNO2 | Family | Estimate | SE | Note |
|---|---|---:|---:|---|
| 59 | Servicios de protección y seguridad | 0.239 | 0.066 | Positive and precise, but only 11 clusters |
| 38 | Técnicos TIC | 0.167 | 0.120 | Positive, imprecise |
| 24 | STEM professionals | 0.127 | 0.072 | Positive, marginal |
| 94 | Other elementary service occupations | 0.126 | 0.167 | Very imprecise |
| 37 | Legal/social/cultural/sports support professionals | 0.106 | 0.125 | Imprecise |

Interpretation:

- The within-family evidence is mixed and often imprecise.
- CNO2 59 is the clearest positive within-family estimate, but it is not an obvious AI-displacement family and has few clusters.
- The within-family results do not rescue a broad causal claim. They are better treated as a guide for qualitative inspection.

## 3. Exposure Validation

I kept and extended the exposure validation queue:

- `output/tables/table_v1_exposure_match_validation_queue.csv`

It now includes nearest Anthropic/O*NET titles, Anthropic occupation exposure, cosine similarity, and weighted-match diagnostics.

Top examples remain plausible:

| CNO4 | Spanish occupation | Nearest Anthropic title | Cosine |
|---|---|---|---:|
| 3820 | Programadores informáticos | Computer Programmers | 0.775 |
| 4301 | Grabadores de datos | Data Entry Keyers | 0.658 |
| 3510 | Agentes y representantes comerciales | Sales Representatives, Wholesale and Manufacturing | 0.697 |
| 2613 | Analistas financieros | Financial and Investment Analysts | 0.824 |

Interpretation:

- The top exposure mappings are mostly plausible.
- This validation supports using the measure descriptively, but it does not solve the identification problem.

## 4. Denominator Work

The project still cannot construct true CNO4-month unemployment rates from SEPE alone because the relevant `personas` denominator is empty.

Implemented denominator proxy:

- `output/tables/table_next_step_cno2_rate_proxy.csv`

Result:

| Specification | Estimate | SE | Interpretation |
|---|---:|---:|---|
| CNO2 SEPE/EPA rate proxy | 0.028 | 0.011 | Positive without trends |
| CNO2 SEPE/EPA rate proxy + CNO2 trends | -0.005 | 0.008 | Approximately zero |

Interpretation:

- The denominator proxy repeats the central pattern: positive simple association, approximately zero after trend adjustment.
- A true denominator should still be pursued through EPA microdata or administrative employment records.

## 5. Age 30-39 With Occupation-Family-by-Month FE

I checked the age 30-39 specification with CNO2-by-year-month fixed effects:

```text
log(parados_it) = beta * HighExposure_i * Post_t
                + CNO4 FE
                + CNO2-by-month FE
                + error_it
```

Note: literal CNO4-by-month fixed effects would absorb the treatment-by-month variation completely, so the estimable version is CNO2-by-month fixed effects.

Results:

| Outcome | Estimate | SE | p-value |
|---|---:|---:|---:|
| `log(parados)` | 0.024 | 0.067 | 0.718 |
| `log(parados + 1)` | 0.018 | 0.062 | 0.774 |
| `log(contratos + 1)` | 0.026 | 0.041 | 0.530 |

Event-study pretrend:

- Pretrend p-value: 0.494.

Interpretation:

- The earlier large age 30-39 estimate disappears when CNO2-by-month fixed effects are included.
- The pretrend diagnostic becomes much better.
- This supports the same conclusion as the full-sample CNO2-by-month FE model: broad occupation-family monthly shocks explain much of the apparent effect.

## 6. HonestDiD on the Original Event Study

I applied HonestDiD sensitivity analysis to the original event-study graph. The package requires the full covariance matrix, so I recomputed and exported it:

- `output/tables/table_recommended_honestdid_original_event_covariance.csv`

Because the full original graph has 43 post-treatment months and the default HonestDiD optimizer is computationally heavy, I used the original event-study estimates in a local event window:

- Pre-periods: event months -12 to -2
- Post-periods: event months 0 to 12
- Target: average effect over event months 0 to 12

This is still the original event-study specification: high exposure versus zero exposure, CNO4 FE, month FE, CNO4-clustered covariance.

Standard parallel-trends CI:

| Target | Lower | Upper |
|---|---:|---:|
| Average event 0 to 12 | -0.010 | 0.039 |

HonestDiD smoothness sensitivity:

| M | Lower | Upper |
|---:|---:|---:|
| 0.000 | -0.116 | 0.028 |
| 0.025 | -0.981 | 0.896 |
| 0.050 | -1.846 | 1.745 |

HonestDiD relative-magnitude sensitivity:

| Mbar | Lower | Upper |
|---:|---:|---:|
| 0.5 | -0.154 | 0.173 |
| 1.0 | -0.248 | 0.248 |
| 2.0 | -0.248 | 0.248 |

The HonestDiD routine warns that some CIs are open at grid endpoints. Therefore those intervals should be read as "at least this wide," not as exact final bounds.

Interpretation:

- Even under standard parallel trends, the first-year average from the original event study is not significantly positive.
- Under HonestDiD relaxations, the interval includes zero and becomes very wide.
- HonestDiD reinforces the earlier conclusion: the original event-study graph is not robust evidence that AI exposure increased unemployment.

## Bottom Line

The latest recommended-step implementation strengthens the conclusion:

> The apparent positive baseline DiD is largely explained by CNO2 occupation-family dynamics. When comparisons are made within occupation families and months, the effect disappears and pretrends improve. The age 30-39 signal also disappears under CNO2-by-month FE. HonestDiD sensitivity applied to the original event-study coefficients does not support a robust positive unemployment effect.

## Next Research Move

The best next step is no longer more global DiD variants. It is a focused diagnostic:

1. Investigate CNO2 27, 38, 24, and 59 individually.
2. Manually validate the high-exposure CNO4 matches inside those families.
3. Search for true employment denominators by occupation family.
4. If denominators remain unavailable, reframe the paper as descriptive evidence on occupation-family dynamics and reallocation rather than causal evidence of AI-driven unemployment.

Sources:

- HonestDiD repository: https://github.com/asheshrambachan/HonestDiD
- Exposure diagnostics repository: https://github.com/dgonzalezgonzalez/Prestaciones-por-empleo-AI-exposure
- INE EPA table 65134: https://www.ine.es/jaxiT3/files/t/csv_bdsc/65134.csv
