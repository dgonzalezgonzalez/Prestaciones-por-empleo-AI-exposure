# Spanish Employment AI Exposure Pipeline

This project builds a Spanish labor-market version of Anthropic's occupation AI exposure analysis.

It downloads Anthropic's `job_exposure.csv`, embeds Anthropic occupation titles with a local Ollama embedding model, trains exposure models from Anthropic `observed_exposure`, translates Spanish occupation labels from INE metadata into English, embeds the English translations, predicts Spanish occupation exposure with several approaches, stores occupation-level predictions in SQLite, and aggregates them to industry-period exposure using INE occupation frequencies within each industry cell.

## Data Sources

- Anthropic Economic Index dataset: `labor_market_impacts/job_exposure.csv`
  - URL: <https://huggingface.co/datasets/Anthropic/EconomicIndex/tree/main/labor_market_impacts>
  - Required columns: `occ_code`, `title`, `observed_exposure`
- INE Encuesta de Poblacion Activa microdata
  - URL: <https://ine.es/dyngs/INEbase/es/operacion.htm?c=Estadistica_C&cid=1254736176918&menu=resultados&secc=1254736030639&idp=1254735976595>
  - Core variables: `OCUP1` for occupation and `ACT1` for industry
  - Metadata workbook: "Diseno de registro y valores validos" from the same INE page
- INE Censo de Poblacion y Viviendas 2021 microdata
  - URL: <https://ine.es/dyngs/INEbase/operacion.htm?c=Estadistica_C&cid=1254736177108&idp=1254735576757&menu=resultados>
  - Core variables: `OCU63` for CNO-11 occupation at two digits and `ACT89` for CNAE-09 industry at two digits
  - Metadata zip: `dr_CensoPersonas_2021.zip`

Raw INE microdata, embedding cache, model files, and generated outputs are excluded from git. They are reproducible local artifacts and may be large.

## Method

1. Download Anthropic occupation exposure data.
2. Clean occupation titles.
3. Embed Anthropic titles with Ollama.
4. Train and validate exposure models with `observed_exposure` as the target and embedding dimensions as predictors. The Random Forest is evaluated on a holdout split, compared with cross-validated baselines, then refit on all Anthropic rows for final prediction.
5. Download INE EPA or Census microdata from a local manifest.
6. Parse Spanish occupation labels from INE metadata (`OCUP1` for EPA, `OCU63`/`T_CNO` for Census).
7. Clean Spanish occupation labels before translation. This removes non-semantic annotations such as `(codigos CNO-2011)`.
8. Translate cleaned Spanish occupation labels to English using the configured translation provider. Domain overrides are applied before external translation for known labor-market terms whose literal translation is misleading. In particular, Spanish `restauración` is translated as food service/catering/hospitality, not art/building restoration.
9. Embed the English translations with the same Ollama embedding model used for Anthropic titles.
10. Predict exposure for each Spanish occupation with Random Forest, ridge regression, cosine-similarity weighted average, cosine nearest-neighbor, and a simple ensemble average.
11. Store predictions in SQLite.
12. Aggregate to industry by period:

```text
observed_exposure_cnae_<method> =
  sum_occupations(weight_occupation_in_industry_period * observed_exposure_occupation_<method>)
```

If a weight column such as `FACTOREL` exists, the pipeline uses it. Otherwise it falls back to record counts. Census microdata has no quarter, so its period is `2021`.

## Exposure Estimation Details

Let Anthropic occupation \(i = 1,\dots,N\) have cleaned English title \(a_i\), embedding vector \(x_i \in \mathbb{R}^d\), and observed Anthropic exposure \(y_i\). Let Spanish occupation \(j = 1,\dots,M\) have cleaned Spanish title \(s_j\), English translation \(t_j\), and embedding vector \(z_j \in \mathbb{R}^d\). The same Ollama embedding model is used for \(x_i\) and \(z_j\), so all approaches work in one shared semantic vector space.

The training set keeps Anthropic rows with non-missing exposure and a cached/generated embedding. Current full runs use \(N = 756\) and \(d = 768\) with `nomic-embed-text`. Diagnostics are computed first; final models used for Spanish prediction are then fit on all \(N\) Anthropic rows. This matters because diagnostic splits should measure performance, not permanently discard labeled Anthropic occupations from final estimation.

### Random Forest: `observed_exposure_rf`

The Random Forest estimates a nonlinear function \(f_{\mathrm{RF}}\) from embedding dimensions to exposure:

```text
observed_exposure_rf_j = f_RF(z_j)
```

Implementation details:

- estimator: `sklearn.ensemble.RandomForestRegressor`
- trees: `AI_EXPOSURE_RF_TREES`, default `500`
- random seed: `AI_EXPOSURE_RANDOM_SEED`, default `20260527`
- `min_samples_leaf = 2`
- `n_jobs = 1`
- diagnostics: one 80/20 holdout split plus 5-fold cross-validation when at least five Anthropic rows exist
- final prediction model: refit on all Anthropic rows after diagnostics

### Ridge Regression: `observed_exposure_ridge`

Ridge regression estimates a linear map from standardized embeddings to exposure:

```text
observed_exposure_ridge_j = alpha + z_j' beta
```

The model is a `StandardScaler()` followed by `RidgeCV`. The penalty grid is:

```text
lambda in {10^-3, 10^-2.5, ..., 10^3}
```

This is included because dense semantic embeddings often encode information in joint vector geometry; a regularized linear model can sometimes be more stable than trees that split on individual embedding coordinates.

### Cosine-Weighted Anthropic Imputation: `observed_exposure_cosine_weighted`

For every Spanish occupation \(j\), cosine similarity to every Anthropic occupation \(i\) is:

```text
c_ji = (z_j dot x_i) / (||z_j|| ||x_i||)
```

Negative similarities are set to zero:

```text
w_ji = max(c_ji, 0)
```

The imputed exposure is the similarity-weighted average of all Anthropic exposure values:

```text
observed_exposure_cosine_weighted_j =
  sum_i(w_ji * y_i) / sum_i(w_ji)
```

If all weights are zero, the fallback is the Anthropic global mean. This is the explicit semantic-linking approach: each Spanish occupation inherits exposure from all Anthropic occupations, with closer occupations receiving more weight.

### Cosine Nearest Neighbor: `observed_exposure_cosine_nearest`

This approach assigns the exposure of the single nearest Anthropic occupation:

```text
i*(j) = argmax_i c_ji
observed_exposure_cosine_nearest_j = y_i*(j)
```

It is intentionally simple and useful as a transparent benchmark: every Spanish occupation can be audited by looking at its closest Anthropic occupation in embedding space.

### Ensemble: `observed_exposure_ensemble`

The ensemble is the unweighted arithmetic mean of the four method-specific estimates:

```text
observed_exposure_ensemble_j =
  (observed_exposure_rf_j
   + observed_exposure_ridge_j
   + observed_exposure_cosine_weighted_j
   + observed_exposure_cosine_nearest_j) / 4
```

It is not optimized or trained; it is a simple robustness summary.

### Industry-Period Aggregation

For industry \(k\), period \(q\), method \(m\), person/record \(r\), occupation code \(o(r)\), industry code \(a(r)\), and survey weight \(W_r\):

```text
observed_exposure_cnae_m(k,q) =
  sum_{r: a(r)=k, period(r)=q} W_r * exposure_m(o(r))
  /
  sum_{r: a(r)=k, period(r)=q, exposure_m(o(r)) observed} W_r
```

Coverage is:

```text
coverage_share(k,q) =
  covered_weight(k,q) / total_weight(k,q)
```

EPA `OCUP1` has only 10 broad groups. Census `OCU63` has 61 two-digit CNO-11 occupations in the processed 2021 file, so it gives more occupational detail but only one cross-section.

### Current Occupation-Level Correlations

Correlations are calculated across the 61 Spanish Census occupation rows in `data/processed/spanish_census_occupation_exposure.csv`.

| measure | rf | ridge | cosine_weighted | cosine_nearest | ensemble |
|---|---:|---:|---:|---:|---:|
| rf | 1.000 | 0.234 | -0.405 | 0.311 | 0.577 |
| ridge | 0.234 | 1.000 | 0.153 | -0.044 | 0.244 |
| cosine_weighted | -0.405 | 0.153 | 1.000 | -0.246 | -0.279 |
| cosine_nearest | 0.311 | -0.044 | -0.246 | 1.000 | 0.929 |
| ensemble | 0.577 | 0.244 | -0.279 | 0.929 | 1.000 |

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

## INE Manifests

INE microdata links are selected on the INE web page and can change over time, so the pipeline uses a small manifest CSV for the exact files you want to process.

EPA example:

```csv
quarter,microdata_url,metadata_url
2025Q4,https://example.ine.es/path/to/epa_2025q4.zip,https://example.ine.es/path/to/diseno_registro.xlsx
```

Census example:

```csv
period,microdata_url,metadata_url
2021,https://ine.es/ftp/microdatos/censopv/cen21/CensoPersonas_2021.zip,https://ine.es/ftp/microdatos/censopv/cen21/dr_CensoPersonas_2021.zip
```

Required columns:

- `microdata_url`: direct URL to the INE microdata zip/csv/txt

Optional columns:

- `period`, `quarter`, or `year`: label used in the final panel, for example `2025Q4` or `2021`
- `metadata_url`: direct URL to INE record/value metadata workbook or zip
- `microdata_filename`: local filename override
- `metadata_filename`: local filename override

You can use one metadata workbook for all EPA quarters by passing `--metadata-xlsx`; otherwise the manifest can download metadata files for archival use. The repository includes a minimal EPA `ine_manifest.csv` and a Census `census_manifest.csv` for 2021 persons microdata.

## Run

Train only on Anthropic data:

```powershell
py -3 main.py --skip-ine --embedding-model nomic-embed-text
```

EPA run:

```powershell
py -3 main.py `
  --source epa `
  --embedding-model nomic-embed-text `
  --translation-provider auto `
  --ine-manifest ine_manifest.csv `
  --metadata-xlsx path\to\diseno_registro_y_valores_validos.xlsx
```

Census run:

```powershell
py -3 main.py `
  --source census `
  --embedding-model nomic-embed-text `
  --translation-provider auto `
  --ine-manifest census_manifest.csv
```

Useful options:

- `--source census`: use Census 2021 `OCU63` and `ACT89` instead of EPA `OCUP1` and `ACT1`
- `--refresh`: re-download source files
- `--max-quarters 1`: process only first manifest row for a quick check
- `--ollama-host http://127.0.0.1:11434`: override Ollama host
- `--translation-provider deepl`: use DeepL API; requires `DEEPL_API_KEY`
- `--translation-provider google_cloud`: use official Google Cloud Translation; requires `GOOGLE_TRANSLATE_API_KEY`
- `--translation-provider google_unofficial`: use the no-key Google Translate web endpoint
- `--translation-provider ollama --translation-model gpt-oss:120b-cloud`: use the previous local LLM translator fallback
- `--allow-code-labels`: allow fallback labels like `OCUP1 1`. This is not recommended for final analysis.

## Outputs

Generated files:

```text
data/cache/embeddings.sqlite
data/cache/translations.sqlite
models/random_forest_<embedding-model>.joblib
models/random_forest_<embedding-model>_metrics.json
data/processed/spanish_ai_exposure.sqlite
data/processed/spanish_census_ai_exposure.sqlite
data/processed/spanish_occupation_exposure.csv
data/processed/spanish_industry_quarter_exposure.csv
data/processed/spanish_census_occupation_exposure.csv
data/processed/spanish_census_industry_period_exposure.csv
data/processed/run_metadata.json
data/processed/spanish_census_run_metadata.json
```

SQLite tables:

### `occupation_exposure`

- `ocup1`: Spanish occupation code normalized to the pipeline occupation key (`OCUP1` for EPA, `OCU63` for Census)
- `occupation_title`: cleaned Spanish occupation label
- `occupation_title_en`: English translation used for embedding
- `embedding_model`: Ollama embedding model
- `translation_model`: translation provider identifier
- `model_sha256`: hash of trained model bundle artifact
- `observed_exposure_rf`: Random Forest prediction
- `observed_exposure_ridge`: ridge regression prediction
- `observed_exposure_cosine_weighted`: weighted average of Anthropic exposure values using nonnegative cosine similarities as weights
- `observed_exposure_cosine_nearest`: exposure from the closest Anthropic occupation by cosine similarity
- `observed_exposure_ensemble`: average of RF, ridge, cosine-weighted, and cosine-nearest predictions
- `generated_at`: write timestamp

### `industry_quarter_exposure`

- `cnae`: industry code normalized to the pipeline industry key (`ACT1` for EPA, `ACT89` for Census)
- `quarter`: quarter or period label from manifest
- `observed_exposure_cnae_rf`, `observed_exposure_cnae_ridge`, `observed_exposure_cnae_cosine_weighted`, `observed_exposure_cnae_cosine_nearest`, `observed_exposure_cnae_ensemble`: method-specific weighted averages
- `total_weight`: total records/weights in industry-period cell
- `covered_weight`: weight with occupation exposure available
- `coverage_share`: `covered_weight / total_weight`
- `occupation_count`: distinct occupation count in cell
- `embedding_model`: Ollama embedding model
- `translation_model`: translation provider identifier
- `model_sha256`: hash of trained model bundle artifact
- `generated_at`: write timestamp

## Translation And Embedding Cache

The cache is SQLite-backed at `data/cache/embeddings.sqlite`.

Translations are SQLite-backed at `data/cache/translations.sqlite`.

Cache keys include:

- embedding model name
- cleaned/normalized text

Spanish labels are cleaned before translation, cache lookup, and embedding. Parentheticals containing terms such as `codigo`, `codigos`, `CNO`, or `CNAE` are removed.

Changing the embedding model or translation provider creates separate cache entries.

## Tests

Run:

```powershell
py -3 -m unittest discover -s tests
```

The tests use fixtures/mocks and do not require network or Ollama.

## Limitations

- Anthropic's occupation exposure labels are based on US occupation semantics, so Spanish predictions are semantic-transfer estimates.
- EPA `OCUP1` and `ACT1` are broad; Census `OCU63` and `ACT89` provide two-digit CNO-11/CNAE-09 detail but only for 2021.
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
