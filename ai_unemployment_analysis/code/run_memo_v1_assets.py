from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw" / "sepe_cno4_monthly_ai_exposure.csv"
PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES = PROJECT_ROOT / "output" / "tables"
FIGURES = PROJECT_ROOT / "output" / "figures"
EXPOSURE = "observed_exposure_cosine_nearest"
EVENT_MONTH = "2022-09"


def ensure_dirs() -> None:
    for path in [PROCESSED, TABLES, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def month_id(period: pd.Series) -> pd.Series:
    return period.str.slice(0, 4).astype(int) * 12 + period.str.slice(5, 7).astype(int)


def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["parados"] = pd.to_numeric(out["parados"], errors="coerce")
    out["contratos"] = pd.to_numeric(out["contratos"], errors="coerce")
    out["ln_parados"] = np.nan
    out.loc[out["parados"] > 0, "ln_parados"] = np.log(out.loc[out["parados"] > 0, "parados"])
    out["ln_contratos"] = np.nan
    out.loc[out["contratos"] > 0, "ln_contratos"] = np.log(out.loc[out["contratos"] > 0, "contratos"])
    ratio = out["parados"] / out["contratos"].replace(0, np.nan)
    out["ln_parados_contratos"] = np.nan
    out.loc[ratio > 0, "ln_parados_contratos"] = np.log(ratio.loc[ratio > 0])
    out["period_date"] = pd.to_datetime(out["period"] + "-01")
    out["month_id"] = month_id(out["period"])
    return out


def exposure_groups(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["exposure_group"] = "middle"
    out.loc[out[EXPOSURE].eq(0), "exposure_group"] = "zero"
    out.loc[out[EXPOSURE].gt(threshold), "exposure_group"] = "high"
    return out


def load_dimension(dimension: str) -> pd.DataFrame:
    usecols = ["period", "cno4", "occupation_title", "dimension", "category", "gender", "contratos", "parados", EXPOSURE]
    chunks = []
    for chunk in pd.read_csv(RAW, usecols=usecols, dtype={"cno4": "string"}, chunksize=500_000):
        chunk["cno4"] = chunk["cno4"].str.zfill(4)
        keep = chunk["cno4"].str.len().eq(4) & chunk["dimension"].eq(dimension)
        chunks.append(chunk.loc[keep].copy())
    return add_outcomes(pd.concat(chunks, ignore_index=True))


def plot_indexed(df: pd.DataFrame, group_cols: list[str], tag: str, title_prefix: str) -> list[str]:
    colors = {"zero": "#5d7186", "middle": "#8c7a3f", "high": "#8f2445"}
    labels = {"zero": "Zero exposure", "middle": "Middle exposure", "high": "High exposure"}
    outcomes = {
        "ln_parados": "log(parados)",
        "ln_contratos": "log(contratos)",
        "ln_parados_contratos": "log(parados / contratos)",
    }
    outputs = []
    plot_group_cols = list(group_cols)
    if not plot_group_cols:
        df = df.copy()
        df["panel"] = "all"
        plot_group_cols = ["panel"]

    collapsed = (
        df.groupby([*plot_group_cols, "period", "period_date", "exposure_group"], as_index=False)
        .agg(
            ln_parados=("ln_parados", "mean"),
            ln_contratos=("ln_contratos", "mean"),
            ln_parados_contratos=("ln_parados_contratos", "mean"),
        )
    )
    collapsed.to_csv(TABLES / f"table_memo_v1_indexed_inputs_{tag}.csv", index=False)

    for outcome, ytitle in outcomes.items():
        plot = collapsed.copy()
        base = (
            plot.loc[plot["period"].between("2022-01", "2022-08")]
            .groupby([*plot_group_cols, "exposure_group"])[outcome]
            .mean()
            .rename("base")
            .reset_index()
        )
        plot = plot.merge(base, on=[*plot_group_cols, "exposure_group"], how="left")
        plot["indexed"] = plot[outcome] - plot["base"]
        levels = list(plot[plot_group_cols[0]].dropna().unique())
        for level in levels:
            sub_panel = plot.loc[plot[plot_group_cols[0]].eq(level)].copy()
            fig, ax = plt.subplots(figsize=(8.8, 4.8))
            for group in ["zero", "middle", "high"]:
                sub = sub_panel.loc[sub_panel["exposure_group"].eq(group)].sort_values("period_date")
                if sub.empty:
                    continue
                ax.plot(sub["period_date"], sub["indexed"], lw=2.0, color=colors[group], label=labels[group])
            ax.axhline(0, color="#777777", lw=0.8)
            ax.axvline(pd.to_datetime(EVENT_MONTH + "-01"), color="#444444", ls="--", lw=1)
            ax.set_title(f"{title_prefix}: {level}, {ytitle}")
            ax.set_xlabel("")
            ax.set_ylabel("Log points relative to Jan-Aug 2022")
            ax.legend(frameon=False, ncols=3, loc="upper right")
            fig.tight_layout()
            safe_level = str(level).replace("<", "lt").replace(">", "gt").replace("-", "_").replace(" ", "_")
            out = FIGURES / f"figure_memo_v1_indexed_{tag}_{safe_level}_{outcome}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            outputs.append(str(out.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    return outputs


def main() -> None:
    ensure_dirs()
    total = pd.read_csv(PROCESSED / "analysis_panel.csv", dtype={"cno4": "string"})
    total = add_outcomes(total)
    threshold = float(total[EXPOSURE].quantile(0.75))
    total = exposure_groups(total, threshold)

    age = load_dimension("age")
    age = exposure_groups(age, threshold)

    outputs = []
    outputs.extend(plot_indexed(total, [], "total", "Total CNO4 panel"))
    outputs.extend(plot_indexed(age, ["category"], "age", "Age-CNO4 panel"))

    metadata = {
        "figures": outputs,
        "tables": [
            "output/tables/table_memo_v1_indexed_inputs_total.csv",
            "output/tables/table_memo_v1_indexed_inputs_age.csv",
        ],
        "threshold_p75": threshold,
        "note": "Indexed outcomes are relative to each exposure group's Jan-Aug 2022 mean.",
    }
    (PROCESSED / "memo_v1_assets_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
