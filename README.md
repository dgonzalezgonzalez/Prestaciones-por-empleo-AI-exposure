# Spanish Employment AI Exposure Pipeline

This project builds a Spanish labor-market version of Anthropic's occupation AI exposure analysis.

It downloads Anthropic's `job_exposure.csv`, embeds Anthropic occupation titles with a local Ollama embedding model, trains a Random Forest model to predict `observed_exposure`, translates Spanish occupation labels from INE EPA metadata into English, embeds the English translations, predicts Spanish occupation exposure, stores occupation-level predictions in SQLite, and aggregates them to industry-quarter exposure using EPA occupation frequencies within each `ACT1` industry cell.

## Data Sources

- Anthropic Economic Index dataset: `labor_market_impacts/job_exposure.csv`
  - URL: <https://huggingface.co/datasets/Anthropic/EconomicIndex/tree/main/labor_market_impacts>
  - Required columns: `occ_code`, `title`, `observed_exposure`
- INE Encuesta de Poblacion Activa microdata
  - URL: <https://ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736176918&menu=resultados&secc=1254736030639&idp=1254735976595>
  - Core variables: `OCUP1` for occupation and `ACT1` for industry
  - Metadata workbook: "Diseño de registro y valores validos" from the same INE page

Raw INE microdata, embedding cache, model files, and generated outputs are excluded from git. They are reproducible local artifacts and may be large.

## Method

1. Download Anthropic occupation exposure data.
2. Clean occupation titles.
3. Embed Anthropic titles with Ollama.
4. Train a `RandomForestRegressor` with `observed_exposure` as the target and embedding dimensions as predictors.
5. Download INE EPA microdata from a local manifest.
6. Parse Spanish `OCUP1` occupation labels from INE metadata.
7. Clean Spanish occupation labels before translation. This removes non-semantic annotations such as `(codigos CNO-2011)` or `(códigos CNO-2011)`.
8. Translate cleaned Spanish occupation labels to English using an Ollama text-generation model.
9. Embed the English translations with the same Ollama embedding model used for Anthropic titles.
10. Predict `observed_exposure` for each Spanish occupation.
11. Store predictions in SQLite.
12. Aggregate to `ACT1` industry by quarter:

```text
observed_exposure_cnae =
  sum_occupations(weight_occupation_in_industry_quarter * observed_exposure_occupation)
```

If an EPA weight column such as `FACTOREL` exists, the pipeline uses it. Otherwise it falls back to record counts.

## Install

No virtual environment is required.

```powershell
py -3 -m pip install -r requirements.txt
```

Install and start Ollama separately:

```powershell
ollama list
ollama pull nomic-embed-text
```

The default embedding model is `nomic-embed-text`. Override it with `--embedding-model` or `OLLAMA_EMBED_MODEL`.

The default translation provider is `auto`. It uses DeepL if `DEEPL_API_KEY` is set, Google Cloud if `GOOGLE_TRANSLATE_API_KEY` is set, and otherwise falls back to the no-key Google Translate web endpoint. Ollama translation remains available only when explicitly requested with `--translation-provider ollama`.

## INE Manifest

INE microdata links are selected on the INE web page and can change over time, so the pipeline uses a small manifest CSV for the exact quarters you want to process.

Create a file such as `ine_manifest.csv`:

```csv
quarter,microdata_url,metadata_url
2025Q4,https://example.ine.es/path/to/epa_2025q4.zip,https://example.ine.es/path/to/diseno_registro.xlsx
2025Q3,https://example.ine.es/path/to/epa_2025q3.zip,https://example.ine.es/path/to/diseno_registro.xlsx
```

Required columns:

- `quarter`: label used in final panel, for example `2025Q4`
- `microdata_url`: direct URL to the INE EPA microdata zip/csv/txt

Optional columns:

- `metadata_url`: direct URL to INE record/value metadata workbook
- `microdata_filename`: local filename override
- `metadata_filename`: local filename override

You can use one metadata workbook for all quarters by passing `--metadata-xlsx`; otherwise the manifest can download metadata files for archival use.

The repository includes `ine_manifest.csv` for all CNO-2011-consistent quarters currently exposed on the INE page, from `2011Q1` through `2026Q1`.

## Run

Train only on Anthropic data:

```powershell
py -3 main.py --skip-ine --embedding-model nomic-embed-text
```

Full run:

```powershell
py -3 main.py `
  --embedding-model nomic-embed-text `
  --translation-provider auto `
  --ine-manifest ine_manifest.csv `
  --metadata-xlsx path\to\diseno_registro_y_valores_validos.xlsx
```

Useful options:

- `--refresh`: re-download source files
- `--max-quarters 1`: process only first manifest row for a quick check
- `--ollama-host http://127.0.0.1:11434`: override Ollama host
- `--translation-provider deepl`: use DeepL API; requires `DEEPL_API_KEY`
- `--translation-provider google_cloud`: use official Google Cloud Translation; requires `GOOGLE_TRANSLATE_API_KEY`
- `--translation-provider google_unofficial`: use the no-key Google Translate web endpoint
- `--translation-provider ollama --translation-model gpt-oss:120b-cloud`: use the previous local LLM translator fallback
- `--allow-code-labels`: allow fallback labels like `OCUP1 1` if metadata labels are missing. This is not recommended for final analysis.

## Outputs

Generated files:

```text
data/cache/embeddings.sqlite
data/cache/translations.sqlite
models/random_forest_<embedding-model>.joblib
models/random_forest_<embedding-model>_metrics.json
data/processed/spanish_ai_exposure.sqlite
data/processed/spanish_occupation_exposure.csv
data/processed/spanish_industry_quarter_exposure.csv
data/processed/run_metadata.json
```

SQLite tables:

### `occupation_exposure`

- `ocup1`: Spanish occupation code from `OCUP1`
- `occupation_title`: cleaned Spanish occupation label
- `occupation_title_en`: English translation used for embedding
- `embedding_model`: Ollama embedding model
- `translation_model`: translation provider identifier
- `model_sha256`: hash of trained Random Forest artifact
- `observed_exposure`: predicted occupation exposure
- `generated_at`: write timestamp

### `industry_quarter_exposure`

- `cnae`: industry code from `ACT1`
- `quarter`: quarter label from manifest
- `observed_exposure_cnae`: weighted average predicted exposure
- `total_weight`: total records/weights in industry-quarter cell
- `covered_weight`: weight with occupation exposure available
- `coverage_share`: `covered_weight / total_weight`
- `occupation_count`: distinct `OCUP1` count in cell
- `embedding_model`: Ollama embedding model
- `translation_model`: translation provider identifier
- `model_sha256`: hash of trained Random Forest artifact
- `generated_at`: write timestamp

## Translation And Embedding Cache

The cache is SQLite-backed at `data/cache/embeddings.sqlite`.

Translations are SQLite-backed at `data/cache/translations.sqlite`.

Cache keys include:

- embedding model name
- cleaned/normalized text

Spanish labels are cleaned before translation, cache lookup, and embedding. Parentheticals containing terms such as `codigo`, `códigos`, `CNO`, or `CNAE` are removed. Example:

```text
Directores comerciales (códigos CNO-2011)
```

becomes:

```text
Directores comerciales
```

becomes an English translation before embedding. Changing the embedding model or translation provider creates separate cache entries.

## Tests

Run:

```powershell
py -3 -m unittest discover -s tests
```

The tests use fixtures/mocks and do not require network or Ollama.

## Limitations

- Anthropic's occupation exposure labels are based on US occupation semantics, so Spanish predictions are semantic-transfer estimates.
- `OCUP1` may be coarser than Anthropic's occupation codes.
- INE file formats and metadata workbooks can vary by period; update the manifest/parser if INE changes file layout.
- The Random Forest baseline is predictive, not causal.
- Final results depend on the chosen embedding model.
- Final results also depend on the chosen machine-translation provider and its translation choices.

## Repository Hygiene

Commit source code, tests, README, and lightweight manifests only. Do not commit:

- raw INE microdata
- embedding cache
- trained model artifacts
- generated SQLite/CSV outputs
