# Does AI Penetration Increase Unemployment?

## Short Answer

The simple DiD estimates show a positive post-2022 association between high AI exposure and registered unemployment counts. In the baseline two-way fixed-effect model, high-exposure CNO4 occupations have about 9.3 percent higher `parados` after September 2022 relative to zero-exposure occupations.

But the current evidence is not strong enough for a causal claim. The event study rejects parallel pretrends, and the estimate disappears when CNO4-specific linear trends are added. The conservative conclusion is:

> High-AI-exposure occupations show a relative post-2022 divergence in registered unemployment counts, but the current design does not yet credibly identify this as a causal effect of AI penetration.

## Context

The design follows the logic of Anthropic's labor-market impact work, which compares high-exposure occupations with low/no-exposure occupations before and after the diffusion of generative AI. Anthropic's measure of "observed exposure" combines theoretical LLM capability with observed work-related usage. Their own early evidence finds no systematic rise in unemployment for highly exposed workers, but some suggestive evidence of slower hiring among young workers.

For Spain, the exposure measure used here comes from the CNO4 adaptation in `dgonzalezgonzalez/Prestaciones-por-empleo-AI-exposure`. The repo maps Anthropic/O*NET occupational exposure to Spanish CNO-2011 occupations using CNO4 occupation descriptions and embedding-based matching. The main exposure in this analysis is `observed_exposure_cosine_nearest`.

Sources:

- Anthropic, "Labor market impacts of AI: A new measure and early evidence", March 5, 2026: https://www.anthropic.com/research/labor-market-impacts
- Spanish exposure repo: https://github.com/dgonzalezgonzalez/Prestaciones-por-empleo-AI-exposure

## Data

The raw input is:

- `data/raw/sepe_cno4_monthly_ai_exposure.csv`

The analysis keeps only the CNO4-month total rows:

- `dimension == total`
- `category == Total`
- `gender == Total`
- CNO4 code length equals 4

Final panel:

- Period: 2021-01 through 2026-03
- Occupations: 502 CNO4 occupations
- Rows: 31,626 occupation-month observations
- Outcome: `log(parados)`, with `log(parados + 1)` as robustness
- Main exposure: `observed_exposure_cosine_nearest`

## Treatment Definition

The baseline treatment follows the preliminary do-file:

- Treated: `observed_exposure_cosine_nearest > p75`
- Control: `observed_exposure_cosine_nearest == 0`
- Middle-exposure occupations are excluded

The 75th percentile is computed on the filtered occupation-month panel, matching the preliminary Stata code. The threshold is 0.1169.

Group counts:

| Exposure group | Occupations | Mean exposure | Mean pre-period parados |
|---|---:|---:|---:|
| High | 124 | 0.289 | 8,136 |
| Middle | 125 | 0.051 | 5,361 |
| Zero | 253 | 0.000 | 6,482 |

The event month is September 2022, so the baseline post period starts in October 2022. A November 2022 treatment date is included as robustness.

## Descriptives

The high-exposure occupations include many expected occupations:

- Programadores informaticos
- Grabadores de datos
- Agentes y representantes comerciales
- Analistas financieros
- Analistas y disenadores de software y multimedia
- Empleados de oficina de servicios estadisticos, financieros y bancarios
- Especialistas en bases de datos y redes informaticas
- Tecnicos de la web
- Tecnicos en asistencia al usuario de tecnologias de la informacion
- Profesionales de relaciones publicas
- Asistentes de direccion y administrativos
- Empleados administrativos
- Filologos, interpretes y traductores

The exposure table is useful for validation. It also surfaces occupations that deserve manual inspection, such as tourism-related occupations and some public-relations roles.

Main descriptive figures:

- `output/figures/figure_01_mean_log_parados_by_exposure_group.png`
- `output/figures/figure_02_indexed_total_log_parados_by_exposure_group.png`
- `output/figures/figure_03_exposure_distribution.png`

The descriptive series show that all groups experienced falling unemployment after the COVID recovery period, but high-exposure occupations declined less after 2022 than middle and zero-exposure occupations. They also start from different levels, so the figures should be interpreted as diagnostics rather than causal evidence.

## Baseline Model

The baseline DiD is:

```text
log(parados_it) = beta * HighExposure_i * Post_t + alpha_i + gamma_t + epsilon_it
```

where:

- `alpha_i` are CNO4 fixed effects
- `gamma_t` are year-month fixed effects
- standard errors are clustered by CNO4

## Estimates

Selected results from `output/tables/table_regression_specifications.csv`:

| Specification | Log-point beta | Clustered SE | Approx. percent effect | Interpretation |
|---|---:|---:|---:|---|
| Main DiD | 0.089 | 0.021 | 9.3% | Positive and significant |
| CNO4 trends | -0.003 | 0.012 | -0.3% | No effect after occupation trends |
| log(parados + 1) | 0.094 | 0.021 | 9.8% | Similar to baseline |
| Weighted by pre parados | 0.043 | 0.026 | 4.4% | Smaller, marginal |
| Event Nov 2022 | 0.091 | 0.021 | 9.5% | Similar timing robustness |
| Top decile | 0.080 | 0.031 | 8.3% | Similar but noisier |
| Continuous exposure | 0.027 | 0.007 | 2.7% per 10pp | Positive gradient |
| Cosine-weighted exposure | 0.100 | 0.021 | 10.5% | Similar |
| RF exposure | 0.150 | 0.021 | 16.2% | Positive, but control changes to RF bottom quartile |
| 2022 onward sample | 0.093 | 0.020 | 9.7% | Similar |
| Pre-period placebo | -0.005 | 0.011 | -0.5% | Null placebo |
| Contracts outcome | 0.123 | 0.030 | 13.1% | Positive contracts, not a simple hiring-collapse story |

## Event Study

The event-study specification interacts treatment with monthly event-time indicators and omits month -1.

Main output:

- `output/figures/figure_04_event_study_high_vs_zero_sep2022.png`
- `output/tables/table_event_study_coefficients.csv`

Key diagnostic:

- Joint pretrend Wald p-value: 2.1e-08

The pre-event coefficients are not flat. Several months before the event are significantly negative relative to month -1, and post-event coefficients rise gradually over time. This pattern is exactly why the trend-augmented DiD matters: the post rise may partly reflect pre-existing differential dynamics rather than AI-induced displacement.

## Identification Assessment

The design has a clear and transparent comparison: high-exposure CNO4 occupations versus zero-exposure CNO4 occupations, before and after the generative-AI diffusion period. But it currently relies on a strong parallel-trends assumption.

Threats to identification:

- High and zero-exposure occupations are compositionally different.
- The outcome is a count, not an unemployment rate.
- Occupation-specific trends matter empirically.
- AI exposure is time-invariant and potential/semantic, not observed Spanish AI adoption.
- Post-2022 shocks may differ across occupational families for reasons unrelated to AI.
- Contracts increase in the high-exposure group, which complicates a pure displacement interpretation.

The current best reading is descriptive and diagnostic:

- There is a positive relative post-2022 divergence in unemployment counts for high-exposure occupations.
- The simple DiD association is robust to several mechanical variants.
- The association is not robust to occupation-specific linear trends.
- The event study does not support clean parallel pretrends.

## Recommended Next Steps

1. Build unemployment rates rather than counts by merging occupation employment/labor-force denominators from EPA or another source.
2. Use the province-disaggregated SEPE panel to estimate CNO4-by-province fixed effects and province-by-month fixed effects.
3. Run age and gender heterogeneity, especially young workers/new entrants, to mirror Anthropic's hiring-margin result.
4. Treat `contratos` as a separate hiring/labor-demand outcome family rather than a robustness-only outcome.
5. Manually validate the high-exposure CNO4 occupations and nearest Anthropic matches for the top exposed groups.
6. Keep September 2022 as the preliminary design date, but present November/December 2022 timing as the more natural ChatGPT robustness.
7. Consider continuous exposure designs as the main specification if the binary top-versus-zero contrast is too compositionally sharp.

## Bottom Line

For the question "does AI penetration increase unemployment?", this dataset currently says:

> There is suggestive positive association in simple DiD estimates, but no credible causal answer yet. The evidence is too sensitive to differential occupation trends to conclude that AI penetration increased unemployment in Spain.
