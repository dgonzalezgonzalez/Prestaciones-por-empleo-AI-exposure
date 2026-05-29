# Spanish Employment AI Exposure Pipeline

This branch adds Census 2021 support to the Spanish labor-market AI exposure pipeline.

Current branch: `census-ai-exposure`.

The EPA-only implementation lives on `main`.

## Current Status

The current pipeline no longer uses Ridge Regression or Ensemble outputs. Active exposure methods are:

- `rf`: Random Forest on embedding vectors.
- `cosine_weighted`: assignment-based cosine weighted average.
- `cosine_nearest`: nearest Anthropic occupation by cosine similarity.

The default occupation representation is taxonomy-aware:

1. Anthropic occupations use O*NET title plus O*NET description.
2. Spanish occupations use parsed CNO-2011 4-digit occupation records from INE's CNO-2011 explanatory PDF.
3. Each CNO4 occupation is embedded as one structured semantic unit.
4. Exposure is predicted at CNO4.
5. Census branch aggregates CNO4 predictions to CNO2 with equal CNO4 weights.
6. Census person records are merged on `OCU63`, normalized internally to the pipeline key `OCUP1`.
7. Industry-period exposure is aggregated by `ACT89` and period `2021`.

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

- INE EPA table 65134 for EPA CNO2 weights
  - Local path: `data/raw/ine/epa_65134_cno2_weights.csv`
  - Used by EPA source only.
  - Not used for Census outputs.

- INE Census 2021 persons microdata
  - Manifest: `census_manifest.csv`
  - Core variables: `OCU63` for 2-digit CNO occupation and `ACT89` for 2-digit CNAE industry.
  - Metadata zip: used to parse `T_CNO` occupation labels and industry labels.

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
   - `OCUP1`, meaning the 1-digit major group in the parsed CNO4 table
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

Implementation:

- `sklearn.ensemble.RandomForestRegressor`
- default trees: `AI_EXPOSURE_RF_TREES`, default `500`
- default seed: `AI_EXPOSURE_RANDOM_SEED`, default `20260527`
- `min_samples_leaf = 2`
- `n_jobs = 1`
- diagnostics: 80/20 holdout and 5-fold cross-validation
- final RF: fit on all 756 Anthropic rows

### 6. Cosine nearest

For Spanish CNO4 target vector `z_j` and Anthropic vector `x_i`:

```text
cosine(j, i) = dot(z_j, x_i) / (norm(z_j) * norm(x_i))
```

Nearest method:

```text
observed_exposure_cosine_nearest = exposure of Anthropic occupation with max cosine similarity
```

### 7. Cosine weighted

For every Anthropic occupation, find the nearest Spanish CNO4 target. Then for each Spanish CNO4, average all Anthropic exposures assigned to it, weighted by cosine similarity.

If no Anthropic occupation is assigned to a Spanish CNO4 target, the code falls back to cosine nearest for that target. This is explicit and prevents missing CNO4 predictions.

### 8. CNO4 to CNO2 aggregation for Census

Census `OCU63` is already a 2-digit CNO variable. Therefore Census aggregation is:

```text
CNO4 -> CNO2: equal average within each CNO2
```

No EPA table 65134 employment weighting is used for Census. Reason: table 65134 is an EPA employment table, while Census branch already uses two-digit CNO categories from Census person records. The Census microdata itself supplies the actual person counts by `OCU63` and `ACT89` during final industry aggregation.

The CNO2 output keeps:

- `CNO2`
- `OCUP1`, intentionally set to the same 2-digit CNO2 code for merge compatibility with the shared aggregation code
- `CNO1`, the 1-digit major group
- `cno4_count`
- `aggregation_weight_source`

### 9. Census industry-period aggregation

Census microdata are merged on normalized `OCUP1`, which contains the Census `OCU63` value.

For industry `ACT89`, period, and method:

```text
observed_exposure_cnae_<method> =
  sum(weight * observed_exposure_<method>) / covered_weight
```

Census run has no `FACTOREL`; the pipeline uses record count `1.0` per person/record.

Outputs include:

- `total_weight`
- `covered_weight`
- `coverage_share`
- `occupation_count`
- `industry_name`

## Match Diagnostics

Two diagnostic outputs show exactly which Anthropic occupations were matched to Spanish CNO4 occupations:

- `data/processed/spanish_census_occupation_matches_cosine_weighted.csv`
- `data/processed/spanish_census_occupation_matches_cosine_nearest.csv`

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

Cosine-only Census run:

```powershell
py -3 main.py --source census --embedding-model qwen3-embedding:4b --ine-manifest census_manifest.csv --methods cosine_weighted,cosine_nearest
```

RF-inclusive Census run:

```powershell
py -3 main.py --source census --embedding-model qwen3-embedding:4b --ine-manifest census_manifest.csv --methods rf,cosine_weighted,cosine_nearest
```

The RF-inclusive Census command wrapper timed out while Python was still running. The process was allowed to finish, then outputs were inspected and only then committed.

Tests:

```powershell
py -3 -m unittest discover -s tests -v
```

Result after cosine-only Census implementation: 18 tests passed.

Result after RF-inclusive Census run: 18 tests passed.

## RF Diagnostics

The Census RF run uses the same fitted model bundle as EPA because the Anthropic training data and embedding model are the same.

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

## Current Census Outputs

Generated tracked outputs:

- `data/processed/spanish_census_occupation_exposure.csv`
  - rows: 62
  - columns:
    - `CNO2`
    - `observed_exposure_rf`
    - `observed_exposure_cosine_weighted`
    - `observed_exposure_cosine_nearest`
    - `cno4_count`
    - `OCUP1`
    - `CNO1`
    - `occupation_title`
    - `aggregation_weight_source`

- `data/processed/spanish_census_industry_period_exposure.csv`
  - rows: 88
  - columns:
    - `cnae`
    - `industry_name`
    - `quarter`, holding period `2021`
    - `observed_exposure_cnae_rf`
    - `observed_exposure_cnae_cosine_weighted`
    - `observed_exposure_cnae_cosine_nearest`
    - `total_weight`
    - `covered_weight`
    - `coverage_share`
    - `occupation_count`

- `data/processed/spanish_census_occupation_matches_cosine_weighted.csv`
  - rows after cosine-only run: 913

- `data/processed/spanish_census_occupation_matches_cosine_nearest.csv`
  - rows after cosine-only run: 502

- `data/processed/spanish_census_run_metadata.json`

- `data/processed/spanish_census_ai_exposure.sqlite`

Ridge and Ensemble columns are intentionally absent.

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

Install and start Ollama separately:

```powershell
ollama pull qwen3-embedding:4b
```

## Run Options

Important options:

- `--source census`: process Census 2021 with `OCU63` and `ACT89`.
- `--source epa`: process EPA in this branch if desired.
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
- Census 2021 is one cross-section, not a quarterly panel.
- Census occupation detail is CNO2 after aggregation, not CNO4 in final industry output.
- CNO4 aggregation uses equal CNO4 weights within each CNO2 for Census.
- CNO PDF parsing is rule-based. It uses 4-digit headings and section markers; it does not use an LLM to interpret the PDF.
- Some PDF text includes extraction artifacts from line breaks or hyphenation. The parser applies light cleanup only.
- CNO descriptions remain Spanish by design.
- Final results depend on `qwen3-embedding:4b`.

## Commit History For This Update

- `159f8d3`: CNO4 cosine pipeline, cosine-only Census outputs, docs note.
- `2532e92`: RF-inclusive Census outputs and RF documentation.
