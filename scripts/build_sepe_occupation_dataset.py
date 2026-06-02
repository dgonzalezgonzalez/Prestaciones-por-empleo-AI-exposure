from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PipelineConfig
from src.sepe import scrape_sepe_monthly_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape SEPE CNO4 monthly occupation dashboards and merge them with CNO4 AI exposure measures."
    )
    parser.add_argument("--output", type=Path, default=None, help="Long-format SEPE output CSV.")
    parser.add_argument("--merged-output", type=Path, default=None, help="Long-format SEPE + AI exposure output CSV.")
    parser.add_argument("--exposure-path", type=Path, default=None, help="Optional existing CNO4 exposure CSV.")
    parser.add_argument("--model-path", type=Path, default=None, help="Existing exposure model bundle for CNO4 merge.")
    parser.add_argument("--embedding-model", default=None, help="Embedding model used by the existing bundle/cache.")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch cached SEPE report HTML.")
    parser.add_argument("--delay-seconds", type=float, default=0.25, help="Sleep between SEPE requests.")
    parser.add_argument("--max-occupations", type=int, default=None, help="Limit CNO4 occupations for smoke tests.")
    parser.add_argument("--max-reports", type=int, default=None, help="Limit monthly report pages for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(embedding_model=args.embedding_model or PipelineConfig().embedding_model)
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
    )
    print(f"Wrote {len(dataset):,} SEPE rows.")
    print(f"Wrote {len(merged):,} merged rows.")


if __name__ == "__main__":
    main()
