from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.aggregate import aggregate_industry_quarter_exposure
from src.config import PipelineConfig
from src.database import connect, upsert_industry_quarter_exposure, upsert_occupation_exposure
from src.download_anthropic import download_anthropic_job_exposure, load_anthropic_job_exposure
from src.download_ine import download_ine_from_manifest, read_ine_microdata
from src.embeddings import EmbeddingCache, OllamaEmbeddingClient, embed_texts
from src.ine_metadata import build_occupation_table, parse_occupation_mapping_from_excel
from src.model import EXPOSURE_COLUMNS, predict_occupation_exposure, train_exposure_model
from src.translation import TranslationCache, TranslationClient, translate_texts_to_english
from src.utils import clean_occupation_title, file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Spanish industry-quarter AI exposure from Anthropic exposure data and INE EPA microdata."
    )
    parser.add_argument("--embedding-model", default=None, help="Ollama embedding model name.")
    parser.add_argument(
        "--translation-provider",
        default=None,
        choices=["auto", "deepl", "google_cloud", "google_unofficial", "ollama"],
        help="Translation provider for Spanish occupation titles.",
    )
    parser.add_argument("--translation-model", default=None, help="Ollama model used only with --translation-provider ollama.")
    parser.add_argument("--ollama-host", default=None, help="Ollama host URL.")
    parser.add_argument("--ine-manifest", type=Path, default=None, help="CSV manifest with quarter,microdata_url,metadata_url.")
    parser.add_argument("--metadata-xlsx", type=Path, default=None, help="INE record/value metadata workbook for OCUP1 labels.")
    parser.add_argument("--refresh", action="store_true", help="Re-download source files.")
    parser.add_argument("--max-quarters", type=int, default=None, help="Limit manifest rows for test runs.")
    parser.add_argument("--allow-code-labels", action="store_true", help="Allow fallback labels like 'OCUP1 1'. Not recommended.")
    parser.add_argument("--skip-ine", action="store_true", help="Only download/train Anthropic model; skip Spanish aggregation.")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    base = PipelineConfig()
    return PipelineConfig(
        ollama_host=args.ollama_host or base.ollama_host,
        embedding_model=args.embedding_model or base.embedding_model,
        translation_provider=args.translation_provider or base.translation_provider,
        translation_model=args.translation_model or base.translation_model,
    )


def _load_ine_files(config: PipelineConfig, args: argparse.Namespace):
    if not args.ine_manifest:
        raise ValueError(
            "INE manifest required for full run. Create a CSV with quarter,microdata_url,metadata_url "
            "from the INE EPA microdata page, then pass --ine-manifest."
        )
    return download_ine_from_manifest(config, args.ine_manifest, args.refresh, args.max_quarters)


def _load_ine_data(config: PipelineConfig, args: argparse.Namespace) -> tuple[pd.DataFrame, Path | None]:
    files = _load_ine_files(config, args)
    frames = [read_ine_microdata(item.microdata_path, item.quarter) for item in files]
    if not frames:
        raise ValueError("INE manifest produced no downloaded quarters.")
    metadata_path = args.metadata_xlsx or next((item.metadata_path for item in files if item.metadata_path), None)
    return pd.concat(frames, ignore_index=True), metadata_path


def main() -> None:
    args = parse_args()
    config = _config_from_args(args)
    config.ensure_dirs()

    anthropic_path = download_anthropic_job_exposure(config, refresh=args.refresh)
    anthropic = load_anthropic_job_exposure(anthropic_path)
    anthropic["embedding_text"] = anthropic["title"].map(clean_occupation_title)

    cache = EmbeddingCache(config.cache_dir / "embeddings.sqlite")
    client = OllamaEmbeddingClient(config.ollama_host, config.embedding_model)
    print(f"Embedding {anthropic['embedding_text'].nunique()} Anthropic occupation titles with {config.embedding_model}...")
    anthropic_embeddings = embed_texts(
        anthropic["embedding_text"].dropna().unique(),
        cache,
        client,
        progress=lambda idx, text: print(f"  embedded Anthropic #{idx}: {text}"),
    )

    model_path = config.models_dir / f"random_forest_{config.embedding_model.replace(':', '_')}.joblib"
    metrics_path = config.models_dir / f"random_forest_{config.embedding_model.replace(':', '_')}_metrics.json"
    model = train_exposure_model(
        anthropic,
        anthropic_embeddings,
        model_path,
        metrics_path,
        config.random_seed,
        config.n_estimators,
    )
    model_sha = file_sha256(model_path)
    print(f"Trained model: {model_path}")
    print(metrics_path.read_text(encoding="utf-8"))

    if args.skip_ine:
        print("Skipped INE processing by request.")
        return

    ine, metadata_path = _load_ine_data(config, args)
    mapping = None
    if metadata_path:
        mapping = parse_occupation_mapping_from_excel(metadata_path)

    occupations = build_occupation_table(ine, mapping, allow_code_labels=args.allow_code_labels)
    occupations["occupation_title_clean"] = occupations["occupation_title"].map(clean_occupation_title)
    translation_cache = TranslationCache(config.cache_dir / "translations.sqlite")
    translation_client = TranslationClient(
        provider=config.translation_provider,
        model=config.translation_model,
        host=config.ollama_host,
    )
    print(
        f"Translating {occupations['occupation_title_clean'].nunique()} Spanish occupation titles "
        f"to English with {translation_client.provider_id}..."
    )
    translations = translate_texts_to_english(
        occupations["occupation_title_clean"].dropna().unique(),
        translation_cache,
        translation_client,
        progress=lambda idx, source, target: print(f"  translated Spain #{idx}: {source} -> {target}"),
    )
    occupations["occupation_title_en"] = occupations["occupation_title_clean"].map(translations)
    occupations["embedding_text"] = occupations["occupation_title_en"].map(clean_occupation_title)
    print(f"Embedding {occupations['embedding_text'].nunique()} translated Spanish occupation titles...")
    spanish_embeddings = embed_texts(
        occupations["embedding_text"].dropna().unique(),
        cache,
        client,
        progress=lambda idx, text: print(f"  embedded Spain #{idx}: {text}"),
    )

    predictions = predict_occupation_exposure(occupations, spanish_embeddings, model)
    panel = aggregate_industry_quarter_exposure(ine, predictions)

    conn = connect(config.db_path)
    upsert_occupation_exposure(conn, predictions, config.embedding_model, translation_client.provider_id, model_sha)
    upsert_industry_quarter_exposure(conn, panel, config.embedding_model, translation_client.provider_id, model_sha)

    occupation_out = config.processed_dir / "spanish_occupation_exposure.csv"
    panel_out = config.processed_dir / "spanish_industry_quarter_exposure.csv"
    predictions.to_csv(occupation_out, index=False)
    panel.to_csv(panel_out, index=False)

    run_meta = {
        "embedding_model": config.embedding_model,
        "translation_provider": translation_client.provider_id,
        "ollama_host": config.ollama_host,
        "model_path": str(model_path),
        "model_metrics_path": str(metrics_path),
        "model_sha256": model_sha,
        "exposure_columns": [column for column in EXPOSURE_COLUMNS if column in predictions.columns],
        "database": str(config.db_path),
        "occupation_output": str(occupation_out),
        "industry_quarter_output": str(panel_out),
    }
    (config.processed_dir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"Wrote {occupation_out}")
    print(f"Wrote {panel_out}")
    print(f"Wrote {config.db_path}")


if __name__ == "__main__":
    main()
