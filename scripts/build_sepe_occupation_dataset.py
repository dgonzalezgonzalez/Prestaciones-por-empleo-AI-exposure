from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig
from src.sepe import build_sepe_dataset_from_cached_reports, scrape_sepe_monthly_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape SEPE CNO4 monthly occupation dashboards and merge them with CNO4 AI exposure measures."
    )
    parser.add_argument("--output", type=Path, default=None, help="Compact SEPE + AI exposure output CSV.")
    parser.add_argument("--merged-output", type=Path, default=None, help="Deprecated alias for --output.")
    parser.add_argument("--exposure-path", type=Path, default=None, help="Optional existing CNO4 exposure CSV.")
    parser.add_argument("--model-path", type=Path, default=None, help="Existing exposure model bundle for CNO4 merge.")
    parser.add_argument("--embedding-model", default=None, help="Embedding model used by the existing bundle/cache.")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch cached SEPE report HTML.")
    parser.add_argument("--delay-seconds", type=float, default=0.25, help="Sleep between SEPE requests.")
    parser.add_argument("--max-occupations", type=int, default=None, help="Limit CNO4 occupations for smoke tests.")
    parser.add_argument("--max-reports", type=int, default=None, help="Limit monthly report pages for smoke tests.")
    parser.add_argument("--resume", action="store_true", help="Append only missing reports to existing output CSVs.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel report fetch workers per occupation.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N reports.")
    parser.add_argument("--from-cache", action="store_true", help="Build output only from cached raw SEPE report HTML.")
    parser.add_argument("--batch-size", type=int, default=500, help="Cached-report batches per CSV append.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(embedding_model=args.embedding_model or PipelineConfig().embedding_model)
    if args.from_cache:
        summary = build_sepe_dataset_from_cached_reports(
            config,
            output_path=args.output or args.merged_output,
            exposure_path=args.exposure_path,
            model_path=args.model_path,
            embedding_model=args.embedding_model,
            workers=args.workers,
            progress=lambda message: print(message, flush=True),
            progress_every=args.progress_every,
            batch_size=args.batch_size,
        )
        print(
            f"Summary: {int(summary.loc[0, 'rows']):,} rows, {int(summary.loc[0, 'reports']):,} reports.",
            flush=True,
        )
        return
    dataset, merged = scrape_sepe_monthly_dataset(
        config,
        exposure_path=args.exposure_path,
        output_path=args.output,
        merged_output_path=args.merged_output,
        model_path=args.model_path,
        embedding_model=args.embedding_model,
        refresh=args.refresh,
        delay_seconds=args.delay_seconds,
        max_occupations=args.max_occupations,
        max_reports=args.max_reports,
        progress=lambda message: print(message, flush=True),
        resume=args.resume,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    print(f"Summary: {int(dataset.loc[0, 'rows']):,} rows, {int(merged.loc[0, 'reports']):,} reports.", flush=True)


if __name__ == "__main__":
    main()
