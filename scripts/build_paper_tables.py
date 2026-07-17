"""Render frozen paper tables without re-estimating any model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs" / "paper_table_values.json"
DEFAULT_OUTPUT = ROOT / "analysis" / "econometrics_outputs" / "tables"


def format_number(value: float | int, digits: int = 3) -> str:
    return f"{value:,.{digits}f}"


def format_summary_value(value: float | int, kind: str) -> str:
    if kind == "level":
        return f"{value:,.0f}"
    return format_number(value)


def latex_escape(value: str) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def wrap_table(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"


def render_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Summary statistics for SEPE occupation-level variables and AI exposure indices}",
        r"\label{tab:summary_statistics}",
        r"\footnotesize",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r" & Mean & SD & Min & Max & Obs. \\",
        r"\midrule",
        r"\textbf{Outcomes} & & & & & \\",
    ]
    for row in rows[:4]:
        values = [format_summary_value(row[key], row["kind"]) for key in ("mean", "sd", "min", "max")]
        lines.append(rf"\hspace{{0.3cm}} {latex_escape(row['label'])} & " + " & ".join(values) + rf" & {row['obs']:,} \\")
    lines.append(r"\textbf{AI exposure index} & & & & & \\")
    for row in rows[4:]:
        values = [format_summary_value(row[key], row["kind"]) for key in ("mean", "sd", "min", "max")]
        lines.append(rf"\hspace{{0.3cm}} {latex_escape(row['label'])} & " + " & ".join(values) + rf" & {row['obs']:,} \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.96\linewidth}",
            r"\footnotesize Notes: Labor-market data come from SEPE. The occupation-level AI exposure measure is based on Anthropic/O*NET occupations mapped to four-digit Spanish CNO occupations.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    return wrap_table(lines)


def render_contdid(rows: list[dict[str, Any]]) -> str:
    columns = [f"({i}) {row['panel']}" for i, row in enumerate(rows, 1)]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Continuous DiD results}",
        r"\label{tab:contdid}",
        r"\footnotesize",
        r"\begin{tabular}{l" + "c" * len(rows) + r"}",
        r"\toprule",
        " & " + " & ".join(columns) + r" \\",
        "Outcome & " + " & ".join(rf"\({latex_escape(row['outcome'])}\)" for row in rows) + r" \\",
        r"\midrule",
        "Overall ATT & " + " & ".join(format_number(row["att"]) for row in rows) + r" \\",
        " & " + " & ".join(rf"({format_number(row['se'])})" for row in rows) + r" \\",
        "Overall ACRT & " + " & ".join(format_number(row["acrt"]) for row in rows) + r" \\",
        r"\midrule",
        "Bootstrap replications & " + " & ".join(str(row["biters"]) for row in rows) + r" \\",
        r"\bottomrule",
        r"\multicolumn{" + str(len(rows) + 1) + r"}{l}{\footnotesize Notes: Standard errors in parentheses. No p-values were present in the current paper output, so no stars are added.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return wrap_table(lines)


def render_sdid(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Synthetic DiD and synthetic-control estimates}",
        r"\label{tab:sdid}",
        r"\footnotesize",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        " & " + " & ".join(f"({i})" for i in range(1, len(rows) + 1)) + r" \\",
        "Outcome & " + " & ".join(latex_escape(f"{row['design']} ({row['outcome']})") for row in rows) + r" \\",
        r"\midrule",
        "SDID ATT & " + " & ".join(format_number(row["sdid_att"]) for row in rows) + r" \\",
        "Synthetic-control ATT & " + " & ".join(format_number(row["sc_att"]) for row in rows) + r" \\",
        "Uniform DiD & " + " & ".join(format_number(row["uniform_did"]) for row in rows) + r" \\",
        r"\bottomrule",
        r"\multicolumn{4}{l}{\footnotesize Notes: The current paper output reports point estimates only for these three estimands; no SEs or stars are added.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return wrap_table(lines)

def render_twfe(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{TWFE benchmark estimates}",
        r"\label{tab:twfe}",
        r"\footnotesize",
        r"\begin{tabular}{l" + "c" * len(rows) + r"}",
        r"\toprule",
        " & " + " & ".join(f"({i})" for i in range(1, len(rows) + 1)) + r" \\",
        " & " + " & ".join(latex_escape(row["specification"]) for row in rows) + r" \\",
        "Outcome & " + " & ".join(latex_escape(row["outcome"]) for row in rows) + r" \\",
        r"\midrule",
        "Estimate & " + " & ".join(format_number(row["estimate"]) for row in rows) + r" \\",
        " & " + " & ".join(rf"({format_number(row['se'])})" for row in rows) + r" \\",
        r"\bottomrule",
        r"\multicolumn{" + str(len(rows) + 1) + r"}{l}{\footnotesize Notes: Standard errors in parentheses. The current paper output did not include p-values, so no stars are added. Original interpretations are retained in the input snapshot.} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return wrap_table(lines)


def build_tables(input_path: Path, output_dir: Path) -> dict[str, Path]:
    data = json.loads(input_path.read_text(encoding="utf-8-sig"))
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary_statistics.tex": render_summary(data["summary_statistics"]),
        "contdid_results.tex": render_contdid(data["continuous_did"]),
        "sdid_results.tex": render_sdid(data["sdid"]),
        "twfe_results.tex": render_twfe(data["twfe"]),
    }
    paths = {}
    for filename, content in outputs.items():
        path = output_dir / filename
        path.write_text(content, encoding="utf-8")
        paths[filename] = path
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for name, path in build_tables(args.input, args.output_dir).items():
        print(f"Wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

