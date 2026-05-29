# 2026-05-29 CNO4 Cosine Pipeline Update

## Scope

This update changes the EPA pipeline from a coarse `OCUP1` title-only transfer to a taxonomy-aware transfer:

1. Anthropic occupations are embedded as O*NET title plus O*NET description.
2. Spanish CNO-2011 occupations are parsed from INE's CNO-2011 explanatory PDF at the 4-digit primary group level.
3. Each CNO4 record is embedded as one structured unit.
4. Exposure is predicted per CNO4 with cosine-weighted and cosine-nearest methods.
5. CNO4 predictions are aggregated to `OCUP1`, then merged into EPA worker records and aggregated to CNAE-quarter.

Ridge Regression and Ensemble outputs were removed. Random Forest remains implemented but is optional and was not run in the cosine-only pass described below.

## Source Files

- Anthropic exposure data: `data/raw/anthropic/job_exposure.csv`.
- O*NET descriptions: `data/raw/anthropic/Occupation_Data_30_3.xlsx`, downloaded from O*NET 30.3 Occupation Data.
- Spanish CNO descriptions: `data/raw/ine/cno11_notas.pdf`, downloaded from INE CNO-2011 notes.
- CNO2 employment weights: `data/raw/ine/epa_65134_cno2_weights.csv`, downloaded from INEbase table 65134.
- EPA microdata manifest: `ine_manifest.csv`.

## Code Changes

- `src/model.py`
  - Removed Ridge Regression and Ensemble from `EXPOSURE_COLUMNS`.
  - Added method selection with valid methods: `rf`, `cosine_weighted`, `cosine_nearest`.
  - Allows cosine-only model bundles with no fitted Random Forest.
  - Adds `cosine_match_details`, which records the Anthropic occupations assigned to each Spanish occupation under both cosine methods.

- `src/taxonomy.py`
  - Downloads and loads O*NET occupation descriptions.
  - Joins O*NET descriptions to Anthropic `occ_code`.
  - Downloads and parses the INE CNO-2011 PDF.
  - Extracts each 4-digit CNO group as a structured record with code, title, definition, tasks, examples, and exclusions where present.
  - Uses CNO4 as the embedding unit.
  - Aggregates CNO4 predictions to `OCUP1` with CNO2 employment weights from INE table 65134.
  - Falls back to equal CNO4 weights only if a CNO2 group has no public employment weight.

- `src/aggregate.py`
  - Removes Ridge/Ensemble exposure ordering.
  - Adds `industry_name` to industry-level processed output when metadata labels are available.

- `src/ine_metadata.py`
  - Adds ACT1/CNAE industry label parsing from INE metadata workbooks.

- `src/database.py`
  - Removes Ridge/Ensemble storage columns from active writes.
  - Allows `observed_exposure_rf` to be null so cosine-only runs can be stored.
  - Adds `industry_name` to industry-quarter records.
  - Rebuilds older SQLite tables if their RF column was created as `NOT NULL`.

- `main.py`
  - Adds `--methods`, e.g. `--methods cosine_weighted,cosine_nearest`.
  - Adds `--occupation-detail cno4`, now default.
  - Writes cosine match diagnostics.

- `requirements.txt`
  - Adds `pypdf` for PDF parsing.

## Key Decisions

- No generic PDF chunking was used. The parser treats each 4-digit CNO primary occupation group as the semantic unit.
- No averaging of CNO4 embeddings was used. The pipeline predicts exposure at CNO4 and averages predictions upward.
- EPA aggregation uses public 2-digit CNO employment weights from INE table 65134, because public EPA table output does not expose CNO4 employment counts.
- Within each CNO2, CNO4 groups are equally weighted before applying the CNO2 employment weight to `OCUP1`.
- Census branch should use equal CNO4 weights to its 2-digit occupation categories, because the Census branch already works at 2-digit occupation detail and does not have matching CNO4 employment weights in the current inputs.
- CNO structured text remains Spanish. This is explicit: `qwen3-embedding:4b` is multilingual, and keeping Spanish avoids introducing undocumented machine-translation changes to long INE descriptions. Anthropic-side text is English because O*NET descriptions are English.

## Cosine-Only EPA Run

Command run:

```powershell
py -3 main.py --embedding-model qwen3-embedding:4b --ine-manifest ine_manifest.csv --methods cosine_weighted,cosine_nearest
```

The command wrapper reported a timeout after final output had already printed, but all expected files were written and verified.

Outputs:

- `data/processed/spanish_occupation_exposure.csv`
  - 10 `OCUP1` rows.
  - Columns: `OCUP1`, `occupation_title`, `cno4_count`, `cno2_count`, `aggregation_weight_source`, `observed_exposure_cosine_weighted`, `observed_exposure_cosine_nearest`.

- `data/processed/spanish_industry_quarter_exposure.csv`
  - 610 CNAE-quarter rows.
  - Includes `industry_name`.
  - Includes only cosine exposure columns, not RF/Ridge/Ensemble.

- `data/processed/spanish_occupation_matches_cosine_weighted.csv`
  - 913 match rows.
  - One row per Anthropic occupation assigned to a CNO4 target under the cosine-weighted assignment rule, with fallback nearest rows where no Anthropic occupation assigned to a target.

- `data/processed/spanish_occupation_matches_cosine_nearest.csv`
  - 502 match rows.
  - One nearest Anthropic occupation per parsed CNO4 record.

- `data/processed/run_metadata.json`
  - Records embedding model, methods, occupation detail mode, source artifacts, model hash, and output paths.

## Verification

```powershell
py -3 -m unittest discover -s tests -v
```

Result: 16 tests passed.

Additional checks:

- Parsed CNO4 records: 502.
- Parsed CNO2 EPA weights: 62, using latest downloaded period `2026T1`.
- EPA cosine output row counts:
  - occupation exposure: 10
  - industry-quarter exposure: 610
  - cosine-weighted match diagnostics: 913
  - cosine-nearest match diagnostics: 502

## RF-Inclusive EPA Run

Command run after the cosine-only commit was pushed:

```powershell
py -3 main.py --embedding-model qwen3-embedding:4b --ine-manifest ine_manifest.csv --methods rf,cosine_weighted,cosine_nearest
```

The command wrapper again reported a timeout after final output had already printed, but all expected files were written and verified.

Model diagnostics:

- Holdout RF MAE: 0.0645711039940907.
- Holdout RF RMSE: 0.09334295237351001.
- Holdout RF R2: 0.42746290775471074.
- 5-fold cross-validation:
  - Random Forest MAE: 0.06917329313943038.
  - Cosine-weighted MAE: 0.057537667569887326.
  - Cosine-nearest MAE: 0.06683253968253969.

Output changes from cosine-only run:

- `data/processed/spanish_occupation_exposure.csv` now includes `observed_exposure_rf`.
- `data/processed/spanish_industry_quarter_exposure.csv` now includes `observed_exposure_cnae_rf`.
- Ridge and Ensemble columns remain absent.
- Cosine match diagnostic files were regenerated unchanged in schema.

Verification after RF run:

```powershell
py -3 -m unittest discover -s tests -v
```

Result: 16 tests passed.
