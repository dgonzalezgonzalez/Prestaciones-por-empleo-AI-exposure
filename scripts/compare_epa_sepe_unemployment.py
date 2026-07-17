"""Compare EPA and scraped SEPE unemployment totals at quarterly frequency."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PipelineConfig
from src.download_ine import download_ine_from_manifest, find_first_tabular_file, load_ine_manifest


SEPE_PATH = ROOT / "data" / "processed" / "sepe_cno4_monthly_ai_exposure.csv"
OUTPUT_DIR = ROOT / "analysis" / "econometrics_outputs" / "Graficos" / "data_quality"
EPA_UNEMPLOYED_AOI = {"05", "06"}
AIREF_COLORS = {
    "burgundy": "#83082A",
    "teal": "#007C89",
    "text": "#404040",
    "axis_label": "#4D4D4D",
    "grid": "#CCCCCC",
}
AIREF_FIGSIZE = (14.5 / 2.54, 7.25 / 2.54)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot quarterly EPA unemployment against scraped SEPE registered unemployment."
    )
    parser.add_argument("--ine-manifest", type=Path, required=True, help="CSV manifest with quarter and EPA microdata URL.")
    parser.add_argument("--sepe-data", type=Path, default=SEPE_PATH, help="Processed SEPE CNO4 monthly CSV.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for figure and source data.")
    parser.add_argument("--refresh", action="store_true", help="Re-download EPA microdata listed in the manifest.")
    return parser.parse_args()


def read_sepe_quarterly_unemployment(path: Path) -> pd.DataFrame:
    usecols = ["period", "dimension", "category", "gender", "parados"]
    df = pd.read_csv(path, usecols=usecols)
    total = df[
        (df["dimension"] == "total")
        & (df["category"] == "Total")
        & (df["gender"] == "Total")
    ].copy()
    if total.empty:
        raise ValueError(f"No total SEPE rows found in {path}")
    total["parados"] = pd.to_numeric(total["parados"], errors="coerce")
    total["period_dt"] = pd.to_datetime(total["period"].astype(str) + "-01", errors="coerce")
    total = total.dropna(subset=["period_dt", "parados"])
    monthly = (
        total.groupby("period_dt", as_index=False)
        .agg(sepe_registered_unemployed=("parados", "sum"))
        .sort_values("period_dt")
    )
    monthly["quarter"] = monthly["period_dt"].dt.to_period("Q").astype(str)
    return monthly.groupby("quarter", as_index=False).agg(
        sepe_registered_unemployed=("sepe_registered_unemployed", "mean"),
        sepe_months=("period_dt", "nunique"),
    )


def read_epa_quarterly_unemployment(
    config: PipelineConfig,
    ine_manifest: Path,
    refresh: bool = False,
    quarters: set[str] | None = None,
) -> pd.DataFrame:
    rows = []
    with TemporaryDirectory() as tmp:
        manifest_for_run = _write_filtered_manifest(Path(tmp), ine_manifest, quarters)
        for item in download_ine_from_manifest(config, manifest_for_run, refresh=refresh):
            print(f"Reading EPA {item.quarter}...", flush=True)
            tabular = find_first_tabular_file(item.microdata_path)
            frame = pd.read_csv(
                tabular,
                sep="\t",
                usecols=["AOI", "FACTOREL"],
                dtype="string",
                encoding_errors="replace",
            )
            frame["AOI"] = frame["AOI"].astype("string").str.zfill(2)
            frame["weight"] = pd.to_numeric(frame["FACTOREL"], errors="coerce").fillna(0.0)
            unemployed = frame[frame["AOI"].isin(EPA_UNEMPLOYED_AOI)]
            rows.append(
                {
                    "quarter": item.quarter,
                    "epa_unemployed": unemployed["weight"].sum(),
                    "epa_records": len(unemployed),
                }
            )
    if not rows:
        raise ValueError(f"INE manifest produced no EPA quarters: {ine_manifest}")
    return pd.DataFrame(rows).sort_values("quarter").reset_index(drop=True)


def _write_filtered_manifest(tmp_dir: Path, path: Path, quarters: set[str] | None) -> Path:
    rows = load_ine_manifest(path)
    if quarters:
        rows = [row for row in rows if row["quarter"].strip() in quarters]
    if not rows:
        raise ValueError(f"INE manifest has no quarters overlapping SEPE data: {sorted(quarters or [])}")
    out = tmp_dir / "ine_manifest_filtered.csv"
    columns = list(rows[0].keys())
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return out


def build_comparison(epa: pd.DataFrame, sepe: pd.DataFrame) -> pd.DataFrame:
    comparison = epa.merge(sepe, on="quarter", how="inner").sort_values("quarter").reset_index(drop=True)
    if comparison.empty:
        raise ValueError("EPA and SEPE data have no overlapping quarters.")
    return comparison


def save_outputs(comparison: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_base = output_dir / "epa_vs_sepe_unemployment_quarterly"
    comparison.to_csv(out_base.with_suffix(".csv"), index=False)
    comparison.to_excel(out_base.with_suffix(".xlsx"), index=False)
    fig, ax = plt.subplots(figsize=AIREF_FIGSIZE)
    x = pd.PeriodIndex(comparison["quarter"], freq="Q").to_timestamp()
    ax.plot(
        x,
        comparison["epa_unemployed"] / 1_000_000,
        color=AIREF_COLORS["burgundy"],
        linewidth=2.0,
        label="EPA unemployed",
    )
    ax.plot(
        x,
        comparison["sepe_registered_unemployed"] / 1_000_000,
        color=AIREF_COLORS["teal"],
        linewidth=2.0,
        label="SEPE registered unemployed",
    )
    ax.set_ylabel("Millions of persons", color=AIREF_COLORS["axis_label"])
    ax.set_xlabel("")
    ax.grid(True, axis="y", color=AIREF_COLORS["grid"], linewidth=0.7, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AIREF_COLORS["grid"])
    ax.spines["bottom"].set_color(AIREF_COLORS["grid"])
    ax.tick_params(colors=AIREF_COLORS["axis_label"])
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".svg"), format="svg", facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), facecolor="white")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(fig)
    return {
        "csv": out_base.with_suffix(".csv"),
        "xlsx": out_base.with_suffix(".xlsx"),
        "svg": out_base.with_suffix(".svg"),
        "pdf": out_base.with_suffix(".pdf"),
        "png": out_base.with_suffix(".png"),
    }


def main() -> None:
    args = parse_args()
    config = PipelineConfig()
    config.ensure_dirs()
    sepe = read_sepe_quarterly_unemployment(args.sepe_data)
    epa = read_epa_quarterly_unemployment(
        config,
        args.ine_manifest,
        refresh=args.refresh,
        quarters=set(sepe["quarter"]),
    )
    comparison = build_comparison(epa, sepe)
    outputs = save_outputs(comparison, args.output_dir)
    print(f"Wrote {outputs['csv']}")
    print(f"Wrote {outputs['svg']}")
    print(f"Wrote {outputs['pdf']}")
    print(f"Wrote {outputs['png']}")


if __name__ == "__main__":
    main()
