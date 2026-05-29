from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.aggregate import aggregate_industry_quarter_exposure
from src.config import PipelineConfig
from src.database import connect, upsert_industry_quarter_exposure, upsert_occupation_exposure
from src.download_anthropic import download_anthropic_job_exposure, load_anthropic_job_exposure
from src.download_ine import download_ine_from_manifest, get_ine_source_spec, read_ine_microdata
from src.embeddings import EmbeddingCache, OllamaEmbeddingClient, embed_texts
from src.ine_metadata import build_occupation_table, parse_industry_mapping_from_excel, parse_occupation_mapping_from_excel
from src.model import EXPOSURE_COLUMNS, cosine_match_details, normalize_methods, predict_occupation_exposure, train_exposure_model
from src.taxonomy import add_onet_descriptions, aggregate_cno4_predictions, load_cno4_records, load_epa_cno2_weights
from src.translation import TranslationCache, TranslationClient, translate_texts_to_english
from src.utils import clean_occupation_title, file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Spanish industry-period AI exposure from Anthropic exposure data and INE microdata."
    )
    parser.add_argument(
        "--source",
        default="epa",
        choices=["epa", "census"],
        help="INE source to process. EPA uses OCUP1/ACT1; census uses OCU63/ACT89.",
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
    parser.add_argument(
        "--methods",
        default="rf,cosine_weighted,cosine_nearest",
        help="Comma-separated methods to run: rf,cosine_weighted,cosine_nearest.",
    )
    parser.add_argument(
        "--occupation-detail",
        default="cno4",
        choices=["cno4", "metadata"],
        help="Use parsed 4-digit CNO records or legacy metadata occupation labels.",
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    base = PipelineConfig()
    db_path = base.db_path
    if args.source != "epa":
        db_path = base.processed_dir / f"spanish_{args.source}_ai_exposure.sqlite"
    return PipelineConfig(
        ollama_host=args.ollama_host or base.ollama_host,
        embedding_model=args.embedding_model or base.embedding_model,
        translation_provider=args.translation_provider or base.translation_provider,
        translation_model=args.translation_model or base.translation_model,
        db_path=db_path,
    )


def _load_ine_files(config: PipelineConfig, args: argparse.Namespace):
    if not args.ine_manifest:
        raise ValueError(
            "INE manifest required for full run. Create a CSV with period,microdata_url,metadata_url "
            "from the INE microdata page, then pass --ine-manifest."
        )
    source_spec = get_ine_source_spec(args.source)
    return download_ine_from_manifest(config, args.ine_manifest, source_spec, args.refresh, args.max_quarters)


def _load_ine_data(config: PipelineConfig, args: argparse.Namespace) -> tuple[pd.DataFrame, Path | None]:
    source_spec = get_ine_source_spec(args.source)
    files = _load_ine_files(config, args)
    frames = [read_ine_microdata(item.microdata_path, item.period, source_spec) for item in files]
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
    anthropic = add_onet_descriptions(anthropic, config, refresh=args.refresh)
    methods = normalize_methods(args.methods)

    cache = EmbeddingCache(config.cache_dir / "embeddings.sqlite")
    client = OllamaEmbeddingClient(config.ollama_host, config.embedding_model)
    print(f"Embedding {anthropic['embedding_text'].nunique()} Anthropic occupation titles with {config.embedding_model}...")
    anthropic_embeddings = embed_texts(
        anthropic["embedding_text"].dropna().unique(),
        cache,
        client,
        progress=lambda idx, text: print(f"  embedded Anthropic #{idx}: {text}"),
    )

    method_slug = "_".join(methods)
    model_path = config.models_dir / f"exposure_model_{config.embedding_model.replace(':', '_')}_{method_slug}.joblib"
    metrics_path = config.models_dir / f"exposure_model_{config.embedding_model.replace(':', '_')}_{method_slug}_metrics.json"
    model = train_exposure_model(
        anthropic,
        anthropic_embeddings,
        model_path,
        metrics_path,
        config.random_seed,
        config.n_estimators,
        methods,
    )
    model_sha = file_sha256(model_path)
    print(f"Built exposure model bundle: {model_path}")
    print(metrics_path.read_text(encoding="utf-8"))

    if args.skip_ine:
        print("Skipped INE processing by request.")
        return

    source_spec = get_ine_source_spec(args.source)
    ine, metadata_path = _load_ine_data(config, args)
    mapping = None
    industry_mapping = None
    if metadata_path:
        mapping = parse_occupation_mapping_from_excel(metadata_path, source_spec)
        industry_mapping = parse_industry_mapping_from_excel(metadata_path, source_spec)

    translation_cache = TranslationCache(config.cache_dir / "translations.sqlite")
    translation_client = TranslationClient(
        provider=config.translation_provider,
        model=config.translation_model,
        host=config.ollama_host,
    )
    if args.occupation_detail == "cno4":
        cno4 = load_cno4_records(config, refresh=args.refresh)
        print(f"Embedding {cno4['embedding_text'].nunique()} parsed CNO4 occupation descriptions...")
        spanish_embeddings = embed_texts(
            cno4["embedding_text"].dropna().unique(),
            cache,
            client,
            progress=lambda idx, text: print(f"  embedded CNO4 #{idx}: {str(text)[:90]}"),
        )
        cno4_predictions = predict_occupation_exposure(cno4, spanish_embeddings, model, methods)
        if args.source == "epa":
            cno2_weights = load_epa_cno2_weights(config, refresh=args.refresh)
            predictions = aggregate_cno4_predictions(cno4_predictions, "ocup1", cno2_weights)
        else:
            predictions = aggregate_cno4_predictions(cno4_predictions, "cno2")
        if mapping is not None:
            labels = mapping.rename(columns={"occupation_title": "metadata_occupation_title"})
            predictions = predictions.merge(labels, on="OCUP1", how="left")
            predictions["occupation_title"] = predictions["metadata_occupation_title"].fillna(predictions["occupation_title"])
            predictions = predictions.drop(columns=["metadata_occupation_title"])
        weighted_matches, nearest_matches = cosine_match_details(cno4, spanish_embeddings, model)
    else:
        occupations = build_occupation_table(ine, mapping, allow_code_labels=args.allow_code_labels)
        occupations["occupation_title_clean"] = occupations["occupation_title"].map(clean_occupation_title)
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
        predictions = predict_occupation_exposure(occupations, spanish_embeddings, model, methods)
        weighted_matches, nearest_matches = cosine_match_details(occupations, spanish_embeddings, model)

    panel = aggregate_industry_quarter_exposure(ine, predictions, industry_mapping=industry_mapping)

    conn = connect(config.db_path)
    upsert_occupation_exposure(conn, predictions, config.embedding_model, translation_client.provider_id, model_sha)
    upsert_industry_quarter_exposure(conn, panel, config.embedding_model, translation_client.provider_id, model_sha)

    if args.source == "epa":
        occupation_out = config.processed_dir / "spanish_occupation_exposure.csv"
        panel_out = config.processed_dir / "spanish_industry_quarter_exposure.csv"
        metadata_out = config.processed_dir / "run_metadata.json"
        weighted_matches_out = config.processed_dir / "spanish_occupation_matches_cosine_weighted.csv"
        nearest_matches_out = config.processed_dir / "spanish_occupation_matches_cosine_nearest.csv"
    else:
        source_prefix = f"spanish_{args.source}"
        occupation_out = config.processed_dir / f"{source_prefix}_occupation_exposure.csv"
        panel_out = config.processed_dir / f"{source_prefix}_industry_period_exposure.csv"
        metadata_out = config.processed_dir / f"{source_prefix}_run_metadata.json"
        weighted_matches_out = config.processed_dir / f"{source_prefix}_occupation_matches_cosine_weighted.csv"
        nearest_matches_out = config.processed_dir / f"{source_prefix}_occupation_matches_cosine_nearest.csv"
    predictions.to_csv(occupation_out, index=False)
    panel.to_csv(panel_out, index=False)
    weighted_matches.to_csv(weighted_matches_out, index=False)
    nearest_matches.to_csv(nearest_matches_out, index=False)

    run_meta = {
        "embedding_model": config.embedding_model,
        "source": args.source,
        "occupation_column": source_spec.occupation_column,
        "industry_column": source_spec.industry_column,
        "methods": list(methods),
        "occupation_detail": args.occupation_detail,
        "translation_provider": translation_client.provider_id,
        "ollama_host": config.ollama_host,
        "model_path": str(model_path),
        "model_metrics_path": str(metrics_path),
        "model_sha256": model_sha,
        "exposure_columns": [column for column in EXPOSURE_COLUMNS if column in predictions.columns],
        "database": str(config.db_path),
        "occupation_output": str(occupation_out),
        "industry_quarter_output": str(panel_out),
        "cosine_weighted_matches_output": str(weighted_matches_out),
        "cosine_nearest_matches_output": str(nearest_matches_out),
    }
    metadata_out.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(f"Wrote {occupation_out}")
    print(f"Wrote {panel_out}")
    print(f"Wrote {weighted_matches_out}")
    print(f"Wrote {nearest_matches_out}")
    print(f"Wrote {config.db_path}")


if __name__ == "__main__":
    main()
