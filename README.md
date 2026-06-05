# Spanish Employment AI Exposure Pipeline

This project builds Spanish labor-market AI exposure estimates from Anthropic's occupation-level `observed_exposure` data.

Current `main` branch scope: EPA only. The Census implementation lives on branch `census-ai-exposure`.

## Current Status

The current EPA pipeline no longer uses Ridge Regression or Ensemble outputs. Active exposure methods are:

- `rf`: Random Forest on embedding vectors.
- `cosine_weighted`: assignment-based cosine weighted average.
- `cosine_nearest`: nearest Anthropic occupation by cosine similarity.

The default occupation representation is now taxonomy-aware:

1. Anthropic occupations use O*NET title plus O*NET description.
2. Spanish occupations use parsed CNO-2011 4-digit occupation records from INE's CNO-2011 explanatory PDF.
3. Each CNO4 occupation is embedded as one structured semantic unit.
4. Exposure is predicted at CNO4.
5. CNO4 predictions are aggregated to EPA `OCUP1`.
6. EPA worker records are merged on `OCUP1`.
7. Industry-quarter exposure is aggregated by `ACT1` plus quarter.

The dated implementation log is in `docs/2026-05-29-cno4-cosine-update.md`.

## Data Sources

- Anthropic Economic Index `job_exposure.csv`
  - Local path: `data/raw/anthropic/job_exposure.csv`
  - URL: <https://huggingface.co/datasets/Anthropic/EconomicIndex/tree/main/labor_market_impacts>
  - Required columns: `occ_code`, `title`, `observed_exposure`

- O*NET 30.3 Occupation Data
  - Local path: `data/raw/anthropic/Occupation_Data_30_3.xlsx`
  - URL used by code: <https://www.onetcenter.org/dl_files/database/db_30_3_excel/Occupation%20Data.xlsx>
  - Used columns: `O*NET-SOC Code`, `Title`, `Description`
  - Join rule: strip trailing `.00` from O*NET codes before merging to Anthropic `occ_code`.

- INE CNO-2011 explanatory PDF
  - Local path: `data/raw/ine/cno11_notas.pdf`
  - URL used by code: <https://ine.es/daco/daco42/clasificaciones/cno11_notas.pdf>
  - Used to parse 4-digit CNO primary occupation groups.

- INE EPA table 65134 for CNO2 employment weights
  - Local path: `data/raw/ine/epa_65134_cno2_weights.csv`
  - URL used by code: <https://www.ine.es/jaxiT3/files/t/csv_bdsc/65134.csv>
  - Filter: `Sexo == Ambos sexos`, `Unidad == Valor absoluto`, latest period in the downloaded table.
  - Current latest parsed period during run: `2026T1`.

- INE EPA microdata
  - Manifest: `ine_manifest.csv`
  - Core variables: `OCUP1` for major occupation, `ACT1` for industry, `FACTOREL` when present for weights.
  - Metadata workbook: used to parse `OCUP1` labels and `ACT1` industry names.

## Exact Method

### 1. Anthropic side

1. Download or reuse `job_exposure.csv`.
2. Load rows with non-missing `occ_code`, `title`, and `observed_exposure`.
3. Download or reuse O*NET Occupation Data.
4. Normalize O*NET codes by removing a trailing `.00`.
5. Merge O*NET descriptions onto Anthropic rows by `occ_code`.
6. Build Anthropic embedding text:

```text
O*NET occupation: <Anthropic title>. Description: <O*NET description>
```

If O*NET description is missing, the code falls back to the cleaned Anthropic title only.

### 2. Spanish CNO side

1. Download or reuse INE `cno11_notas.pdf`.
2. Parse PDF text with `pypdf`.
3. Start after introductory pages.
4. Detect 4-digit headings such as `1111 Miembros del poder ejecutivo...`.
5. Treat each 4-digit CNO group as exactly one occupation record.
6. Extract structured fields where possible:
   - `CNO4`
   - `CNO2`
   - `OCUP1`
   - Spanish title
   - definition text
   - typical tasks after `Entre sus tareas se incluyen:`
   - included examples
   - related or excluded occupations
7. Build CNO embedding text:

```text
CNO occupation: <code> <title>. Definition: ... Typical tasks: ... Examples included: ... Related or excluded occupations: ...
```

No generic chunks are used. No arbitrary chunk averaging is used.

Spanish CNO text is not translated before embedding. Decision: `qwen3-embedding:4b` is multilingual, and translating long PDF descriptions would add another undocumented transformation layer. Anthropic text remains English because O*NET is English.

### 3. Embeddings

Embeddings are generated through local Ollama and cached in `data/cache/embeddings.sqlite`.

Cache key includes:

- embedding model name
- cleaned/normalized text

Current tracked outputs were generated with:

```text
qwen3-embedding:4b
```

Current embedding dimension from run metadata: `2560`.

### 4. Exposure model bundle

`src/model.py` builds an `ExposureModelBundle` containing:

- optional fitted Random Forest
- Anthropic embedding matrix
- Anthropic exposure vector
- Anthropic titles and occupation codes for diagnostics
- metrics dictionary

Valid methods:

```text
rf, cosine_weighted, cosine_nearest
```

`--methods cosine_weighted,cosine_nearest` runs without fitting or requiring RF. This was added so cosine-only runs do not pay the RF runtime cost.

### 5. Random Forest

RF method:

```text
observed_exposure_rf
```

Let Anthropic occupation $i = 1,\dots,N$ have embedding $x_i \in \mathbb{R}^d$ and observed exposure $y_i$. Let Spanish CNO4 occupation $j$ have embedding $z_j \in \mathbb{R}^d$. The Random Forest estimates:

$$
\widehat{y}^{\mathrm{RF}}_j = f_{\mathrm{RF}}(z_j)
$$

Implementation:

- `sklearn.ensemble.RandomForestRegressor`
- default trees: `AI_EXPOSURE_RF_TREES`, default `500`
- default seed: `AI_EXPOSURE_RANDOM_SEED`, default `20260527`
- `min_samples_leaf = 2`
- `n_jobs = 1`
- diagnostics: 80/20 holdout and 5-fold cross-validation
- final RF: fit on all 756 Anthropic rows

### 6. Cosine nearest

For Spanish CNO4 target vector $z_j$ and Anthropic vector $x_i$:

$$
c_{ji} =
\frac{z_j^\top x_i}{\lVert z_j \rVert \lVert x_i \rVert}
$$

Nearest method:

$$
i^\*(j) = \arg\max_i c_{ji}
$$

$$
\widehat{y}^{\mathrm{NN}}_j = y_{i^\*(j)}
$$

### 7. Cosine weighted

For every Anthropic occupation, find the nearest Spanish CNO4 target. Then for each Spanish CNO4, average all Anthropic exposures assigned to it, weighted by cosine similarity.

$$
j^\*(i) = \arg\max_j c_{ji}
$$

$$
A_j = \{i : j^\*(i) = j\}
$$

$$
\widehat{y}^{\mathrm{CW}}_j =
\frac{\sum_{i \in A_j} c_{ji} y_i}{\sum_{i \in A_j} c_{ji}}
$$

If no Anthropic occupation is assigned to a Spanish CNO4 target, the code falls back to cosine nearest for that target. This is explicit and prevents missing CNO4 predictions.

### 8. CNO4 to OCUP1 aggregation

EPA public table 65134 gives usable occupation employment counts down to CNO2, not CNO4.

Therefore EPA aggregation is:

```text
CNO4 -> CNO2: equal average within each CNO2
CNO2 -> OCUP1: weighted average using INE EPA table 65134 employment weights
```

For CNO2 group $g$, CNO4 occupations $J_g$, method $m$:

$$
\widehat{y}^{m}_{g} =
\frac{1}{|J_g|}
\sum_{j \in J_g} \widehat{y}^{m}_{j}
$$

For OCUP1 group $h$, CNO2 groups $G_h$, and EPA CNO2 employment weights $w_g$ from INE table 65134:

$$
\widehat{y}^{m}_{h} =
\frac{\sum_{g \in G_h} w_g \widehat{y}^{m}_{g}}
{\sum_{g \in G_h} w_g}
$$

If an `OCUP1` group has no matching CNO2 public weight, fallback is equal CNO4 weights for that group. The output records the source in `aggregation_weight_source`.

### 9. EPA industry-quarter aggregation

EPA microdata are merged on `OCUP1`.

For industry `ACT1`, quarter, and method:

Let person or record $r$ have industry $a(r)$, occupation $o(r)$, quarter $q(r)$, and survey weight $W_r$. For industry $k$, quarter $t$, and method $m$:

$$
\widehat{Y}^{m}_{kt} =
\frac{
\sum_{r: a(r)=k,\;q(r)=t} W_r \widehat{y}^{m}_{o(r)}
}{
\sum_{r: a(r)=k,\;q(r)=t,\;\widehat{y}^{m}_{o(r)}\;\mathrm{observed}} W_r
}
$$

Coverage is:

$$
\text{coverage share}_{kt} =
\frac{\text{covered weight}_{kt}}{\text{total weight}_{kt}}
$$

Weight source:

- use `FACTOREL` if present
- otherwise use record count `1.0`

Outputs include:

- `total_weight`
- `covered_weight`
- `coverage_share`
- `occupation_count`
- `industry_name`

## Match Diagnostics

Two diagnostic outputs show exactly which Anthropic occupations were matched to Spanish CNO4 occupations:

- `data/processed/spanish_occupation_matches_cosine_weighted.csv`
- `data/processed/spanish_occupation_matches_cosine_nearest.csv`

Columns include:

- `method`
- `spanish_code`
- `spanish_title`
- `spanish_embedding_text`
- `anthropic_occ_code`
- `anthropic_title`
- `anthropic_observed_exposure`
- `cosine_similarity`
- `CNO4`
- `CNO2`
- `OCUP1`

Weighted diagnostics have one row per Anthropic occupation assigned to a Spanish CNO4 target, plus fallback nearest rows for targets with no assignments. Nearest diagnostics have one row per CNO4 target.

## Commands Actually Run

Cosine-only EPA run:

```powershell
py -3 main.py --embedding-model qwen3-embedding:4b --ine-manifest ine_manifest.csv --methods cosine_weighted,cosine_nearest
```

RF-inclusive EPA run:

```powershell
py -3 main.py --embedding-model qwen3-embedding:4b --ine-manifest ine_manifest.csv --methods rf,cosine_weighted,cosine_nearest
```

Both command wrappers reported timeout after final output had already printed. Files were inspected afterward and committed only after expected outputs existed.

Tests:

```powershell
py -3 -m unittest discover -s tests -v
```

Result after cosine-only EPA implementation: 16 tests passed.

Result after RF-inclusive EPA run: 16 tests passed.

## RF Diagnostics From Current EPA RF Run

Model file:

```text
models/exposure_model_qwen3-embedding_4b_rf_cosine_weighted_cosine_nearest.joblib
```

Model SHA256:

```text
832de72489b2f2318a54b0b0b47b78fe6fad17a4a17e9c5ca8f419cf9cec0e1d
```

Holdout RF diagnostics:

- MAE: `0.0645711039940907`
- RMSE: `0.09334295237351001`
- R2: `0.42746290775471074`

5-fold cross-validation:

- global mean MAE: `0.09735047482001576`
- Random Forest MAE: `0.06917329313943038`
- cosine weighted MAE: `0.057537667569887326`
- cosine nearest MAE: `0.06683253968253969`

## Current EPA Outputs

Generated tracked outputs:

- `data/processed/spanish_occupation_exposure.csv`
  - rows: 10
  - columns:
    - `OCUP1`
    - `occupation_title`
    - `cno4_count`
    - `cno2_count`
    - `aggregation_weight_source`
    - `observed_exposure_rf`
    - `observed_exposure_cosine_weighted`
    - `observed_exposure_cosine_nearest`

- `data/processed/spanish_industry_quarter_exposure.csv`
  - rows: 610
  - columns:
    - `cnae`
    - `industry_name`
    - `quarter`
    - `observed_exposure_cnae_rf`
    - `observed_exposure_cnae_cosine_weighted`
    - `observed_exposure_cnae_cosine_nearest`
    - `total_weight`
    - `covered_weight`
    - `coverage_share`
    - `occupation_count`

- `data/processed/spanish_occupation_matches_cosine_weighted.csv`
  - rows after cosine-only run: 913

- `data/processed/spanish_occupation_matches_cosine_nearest.csv`
  - rows after cosine-only run: 502

- `data/processed/run_metadata.json`

- `data/processed/spanish_ai_exposure.sqlite`

Ridge and Ensemble columns are intentionally absent.

## SEPE CNO4 Monthly Dashboard Scrape

SEPE's occupation page exposes monthly CNO4 dashboard reports through HTML report pages rather than a documented bulk
download. This repo now includes a resumable scraper/parser for those reports:

```powershell
py -3 scripts/build_sepe_occupation_dataset.py --embedding-model qwen3-embedding:4b
```

If `data/raw/sepe/reports/` already contains the cached report HTML files, rebuild the processed dataset without
touching the SEPE website:

```powershell
py -3 scripts/build_sepe_occupation_dataset.py --embedding-model qwen3-embedding:4b --from-cache --workers 8
```

Smoke-test options:

```powershell
py -3 scripts/build_sepe_occupation_dataset.py --embedding-model qwen3-embedding:4b --max-occupations 1 --max-reports 1
```

Outputs:

- `data/processed/sepe_cno4_monthly_ai_exposure.csv`

This generated CSV is intentionally ignored by git because the full processed file is larger than GitHub's normal
single-file limit.

The output is compact long-by-disaggregation format with:

- `period`: monthly period as `YYYY-MM`
- `cno4`
- `occupation_title`
- `dimension`: `total`, `gender`, `age`, `province`, or `geographic_mobility`
- `category`: e.g. `Total`, `Hombre`, `Mujer`, age band, province, `Permanecen`, `Se mueven`
- `gender`: `Total` unless the row is gender-disaggregated
- `contratos`, `parados`, `personas`: level columns; monthly and annual variation columns are intentionally ignored
- `exposure_occupation_title`: CNO4 title from the exposure model source, kept separate from the SEPE title
- `observed_exposure_rf`, `observed_exposure_cosine_weighted`, `observed_exposure_cosine_nearest`

`Total` is written once per CNO4-month as `dimension == total` and `category == Total`; duplicate totals from gender,
age, province, or mobility subtables are dropped.

Raw report HTML is cached under `data/raw/sepe/reports/`, so interrupted runs can resume without re-fetching completed
report pages. The script reads the existing model bundle and embedding cache to reconstruct CNO4 exposure measures; it
does not retrain the exposure model.

## SEPE CNO4 Econometric Analysis

The SEPE econometric analysis is run from one script:

```powershell
py -3 scripts/run_ai_exposure_econometrics.py
```

It can also be launched from the main pipeline entry point:

```powershell
py -3 main.py --analysis-only --run-sepe-econometrics
```

Run all SEPE analysis modules from `main.py`:

```powershell
py -3 main.py --analysis-only --run-all-analyses --rscript "C:\Users\dgonzalez\AppData\Local\Programs\R\R-4.5.2\bin\Rscript.exe"
```

Available analysis flags:

- `--run-sepe-econometrics`: OLS and TWFE event studies.
- `--run-sdid`: synthetic difference-in-differences smoke-test estimates and diagnostic figures.
- `--run-contdid`: continuous-treatment `contdid` event-study and dose-aggregation estimates.
- `--run-all-analyses`: run all three analysis modules.
- `--analysis-only`: skip the exposure build and run only requested analysis modules. With no specific analysis flag, this runs all analysis modules.

Without `--analysis-only`, these flags run after the regular exposure pipeline completes.

Input:

- `data/processed/sepe_cno4_monthly_ai_exposure.csv`

The script uses only aggregate CNO4-month rows:

```text
dimension == total
category == Total
gender == Total
```

OLS output:

- Unit of analysis: CNO4 occupation.
- Outcome: average monthly log growth in registered unemployed (`parados`) from `2021-01` to `2026-01`, in percentage points.
- Regressors: `observed_exposure_rf`, `observed_exposure_cosine_weighted`, and `observed_exposure_cosine_nearest`.
- Outputs:
  - `analysis/econometrics_outputs/tables/ols_growth_regressions.tex`
  - `analysis/econometrics_outputs/tables/ols_growth_regressions_document.pdf`
  - `analysis/econometrics_outputs/tables/ols_growth_regressions.csv`

Event-study output:

- Intervention period: `2022-09`, preserving the requested dating in the analysis.
- Estimator: TWFE OLS with CNO4 and period fixed effects.
- Standard errors: clustered by CNO4 occupation.
- Baseline event month: `-1`.
- Outcomes:
  - unemployment: `log1p(parados)`
  - contracts: `contratos` in levels, with no log transform because many observations are zero
- Specifications for each outcome:
  - continuous treatment using each of the three AI exposure measures
  - top exposure quartile vs zero exposure, for cosine weighted and cosine nearest
  - top exposure quartile vs bottom exposure quartile, for all three AI exposure measures

Event-study outputs are split by outcome:

- `analysis/econometrics_outputs/event_studies/unemployment/`
- `analysis/econometrics_outputs/event_studies/contracts/`
- combined outcome file: `analysis/econometrics_outputs/event_studies/event_study_coefficients_all_outcomes.csv`

AIReF-style event-study figures are split by outcome:

- `analysis/econometrics_outputs/Graficos/unemployment/`
- `analysis/econometrics_outputs/Graficos/contracts/`

Each figure folder contains SVG, PDF, PNG, and XLSX source-data exports for each event-study specification.

Additional DiD outputs:

- `analysis/econometrics_outputs/sdid/`: synthetic DiD estimates, SDID gap/event-style figures, and treated-vs-synthetic level figures.
- `analysis/econometrics_outputs/contdid/`: continuous-treatment `contdid` estimates using RF, cosine weighted, and cosine nearest AI exposure as dose variables. This folder contains event-study ACRT plots plus dose-aggregation `ATT(d)` and `ACRT(d)` plots following Case 1 of the `contdid` README.

`contdid` is run through R. If `Rscript` is not on `PATH`, pass `--rscript` or set `R_SCRIPT`.

Current committed SDID and `contdid` outputs use 100 bootstrap/placebo replications.

## Install

```powershell
py -3 -m pip install -r requirements.txt
```

Required Python packages:

- `pandas`
- `scikit-learn`
- `requests`
- `joblib`
- `openpyxl`
- `pypdf`
- `beautifulsoup4`
- `matplotlib`
- `scipy`

Install and start Ollama separately:

```powershell
ollama pull qwen3-embedding:4b
```

## Run Options

Important options:

- `--methods cosine_weighted,cosine_nearest`: cosine-only, no RF fit.
- `--methods rf,cosine_weighted,cosine_nearest`: all active methods.
- `--occupation-detail cno4`: default taxonomy-aware CNO4 pipeline.
- `--occupation-detail metadata`: legacy metadata-title path.
- `--refresh`: re-download source files.
- `--max-quarters 1`: quick manifest check.
- `--allow-code-labels`: allow fallback labels if metadata labels are missing. Not recommended for final outputs.

## SQLite Tables

### `occupation_exposure`

- `ocup1`
- `occupation_title`
- `occupation_title_en`
- `embedding_model`
- `translation_model`
- `model_sha256`
- `observed_exposure_rf`
- `observed_exposure_cosine_weighted`
- `observed_exposure_cosine_nearest`
- `generated_at`

The RF column allows nulls so cosine-only runs can be stored.

### `industry_quarter_exposure`

- `cnae`
- `industry_name`
- `quarter`
- `total_weight`
- `covered_weight`
- `coverage_share`
- `occupation_count`
- `embedding_model`
- `translation_model`
- `model_sha256`
- `observed_exposure_cnae_rf`
- `observed_exposure_cnae_cosine_weighted`
- `observed_exposure_cnae_cosine_nearest`
- `generated_at`

## What Was Removed

Removed from active code/output:

- `observed_exposure_ridge`
- `observed_exposure_ensemble`
- `observed_exposure_cnae_ridge`
- `observed_exposure_cnae_ensemble`

Older SQLite files may still contain historical dropped columns only if opened before rebuild logic runs. Current active writes do not populate them.

## Known Caveats

- Anthropic labels are US occupation semantics, so outputs are semantic-transfer estimates, not validated Spanish causal measures.
- EPA still merges final predictions at `OCUP1`, so final EPA panel has 10 occupation groups.
- CNO4 aggregation uses equal CNO4 weights within CNO2 because public EPA table 65134 does not expose CNO4 counts.
- CNO PDF parsing is rule-based. It uses 4-digit headings and section markers; it does not use an LLM to interpret the PDF.
- Some PDF text includes extraction artifacts from line breaks or hyphenation. The parser applies light cleanup only.
- CNO descriptions remain Spanish by design.
- Final results depend on `qwen3-embedding:4b`.

## Commit History For This Update

- `4a697fe`: CNO4 cosine pipeline, cosine-only EPA outputs, docs note.
- `05595af`: RF-inclusive EPA outputs and RF documentation.
