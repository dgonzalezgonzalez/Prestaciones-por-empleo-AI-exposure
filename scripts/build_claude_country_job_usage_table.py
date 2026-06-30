from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PipelineConfig
from src.download_anthropic import download_anthropic_economic_index_release, load_country_job_usage


OUTPUT_DIR = Path("analysis") / "econometrics_outputs" / "tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Spain-US Claude usage by SOC major job group table.")
    parser.add_argument("--refresh", action="store_true", help="Re-download the Anthropic Economic Index release zip.")
    parser.add_argument(
        "--date-start",
        default=None,
        help="Collection period start date to use. Defaults to latest available month in the release.",
    )
    return parser.parse_args()


def latex_escape(value: str) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def write_latex_table(table, output_path: Path) -> Path:
    period_start = table["date_start"].iloc[0]
    period_end = table["date_end"].iloc[0]
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Job group & Spain (\%) & United States (\%) & Difference (p.p.) \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{latex_escape(row['job_group'])} & "
            f"{row['spain_usage_pct']:.2f} & "
            f"{row['us_usage_pct']:.2f} & "
            rf"{row['spain_minus_us_pct']:.2f} \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"% Source: Anthropic Economic Index, Claude AI release 2026-06-26, country SOC major-group pct rows. Period: {period_start} to {period_end}.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()
    config = PipelineConfig()
    release_path = download_anthropic_economic_index_release(config, refresh=args.refresh)
    table = load_country_job_usage(release_path, date_start=args.date_start)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "claude_country_job_usage_spain_us.csv"
    tex_path = OUTPUT_DIR / "claude_country_job_usage_spain_us.tex"
    table.to_csv(csv_path, index=False)
    write_latex_table(table, tex_path)

    print(f"CSV: {csv_path}")
    print(f"LaTeX table: {tex_path}")
    print(f"Period: {table['date_start'].iloc[0]} to {table['date_end'].iloc[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
