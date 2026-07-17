"""Build the live Spain/US Anthropic occupation-use figure.

The source data are loaded through the shared Anthropic parser, which orders Spain
occupation groups by descending Spanish usage with a stable code tie-break.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PipelineConfig
from src.download_anthropic import download_anthropic_economic_index_release, load_country_job_usage

DEFAULT_OUTPUT = ROOT / "analysis" / "econometrics_outputs" / "Graficos" / "figure_anthropic_country_soc_major_group_spain_us_may2026.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Spain/US Anthropic occupation-use figure.")
    parser.add_argument("--refresh", action="store_true", help="Re-download the Anthropic Economic Index release zip.")
    parser.add_argument("--date-start", default=None, help="Release period start date; defaults to the latest available month.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="PNG output path.")
    return parser.parse_args()


def build_figure(table: pd.DataFrame, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    plot = table.copy()
    y = range(len(plot))

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.barh(list(y), plot["spain_usage_pct"], color="#83082A", alpha=0.9, label="Spain")
    ax.barh(list(y), -plot["us_usage_pct"], color="#666666", alpha=0.85, label="United States")
    ax.axvline(0, color="#404040", linewidth=0.8)
    ax.set_yticks(list(y), labels=plot["job_group"])
    ax.invert_yaxis()
    ax.set_xlabel("Share of Claude use (%)")
    ax.set_ylabel("Occupation group")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.grid(axis="x", color="#CCCCCC", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    args = parse_args()
    release_path = download_anthropic_economic_index_release(PipelineConfig(), refresh=args.refresh)
    table = load_country_job_usage(release_path, date_start=args.date_start)
    output = build_figure(table, args.output)
    print(f"Figure: {output}")
    print(f"Period: {table['date_start'].iloc[0]} to {table['date_end'].iloc[0]}")
    print(f"Top Spain occupation group: {table['job_group'].iloc[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
