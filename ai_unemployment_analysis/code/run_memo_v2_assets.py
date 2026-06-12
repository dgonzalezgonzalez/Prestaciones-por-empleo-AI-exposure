from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw" / "sepe_cno4_monthly_ai_exposure.csv"
PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES = PROJECT_ROOT / "output" / "tables"
FIGURES = PROJECT_ROOT / "output" / "figures"
EXPOSURE = "observed_exposure_cosine_nearest"
EVENT_MONTH = "2022-09"
EVENT_MONTH_ID = 2022 * 12 + 9


AGE_BUCKETS = {
    "<18": "<18 to 29",
    "18-24": "<18 to 29",
    "25-29": "<18 to 29",
    "30-39": "30-39",
    "40-44": "40 to >44",
    ">44": "40 to >44",
}


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


def add_binary_design(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["treat"] = out[EXPOSURE] > threshold
    out["control"] = out[EXPOSURE] == 0
    out = out.loc[out["treat"] | out["control"]].copy()
    out["post"] = out["month_id"] > EVENT_MONTH_ID
    out["did_post"] = out["treat"].astype(int) * out["post"].astype(int)
    return out


def load_dimension(dimension: str) -> pd.DataFrame:
    usecols = ["period", "cno4", "occupation_title", "dimension", "category", "gender", "contratos", "parados", EXPOSURE]
    chunks = []
    for chunk in pd.read_csv(RAW, usecols=usecols, dtype={"cno4": "string"}, chunksize=500_000):
        chunk["cno4"] = chunk["cno4"].str.zfill(4)
        keep = chunk["cno4"].str.len().eq(4) & chunk["dimension"].eq(dimension)
        chunks.append(chunk.loc[keep].copy())
    return pd.concat(chunks, ignore_index=True)


def aggregate_age_buckets(age: pd.DataFrame) -> pd.DataFrame:
    out = age.copy()
    out["age_bucket"] = out["category"].map(AGE_BUCKETS)
    out = out.loc[out["age_bucket"].notna()].copy()
    grouped = (
        out.groupby(["period", "cno4", "occupation_title", "age_bucket"], as_index=False)
        .agg(
            parados=("parados", "sum"),
            contratos=("contratos", "sum"),
            observed_exposure_cosine_nearest=(EXPOSURE, "first"),
        )
    )
    return add_outcomes(grouped)


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
    collapsed.to_csv(TABLES / f"table_memo_v2_indexed_inputs_{tag}.csv", index=False)

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
            safe_level = (
                str(level)
                .replace("<", "lt")
                .replace(">", "gt")
                .replace("-", "_")
                .replace(" ", "_")
            )
            out = FIGURES / f"figure_memo_v2_indexed_{tag}_{safe_level}_{outcome}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            outputs.append(str(out.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    return outputs


def fit_clustered_twfe(df: pd.DataFrame, bucket: str, outcome: str) -> dict:
    data = df.loc[df["age_bucket"].eq(bucket)].dropna(subset=[outcome, "did_post", "cno4", "period"]).copy()
    result = smf.ols(f"{outcome} ~ did_post + C(cno4) + C(period)", data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["cno4"], "use_correction": True},
    )
    beta = float(result.params["did_post"])
    se = float(result.bse["did_post"])
    return {
        "age_bucket": bucket,
        "outcome": outcome,
        "beta_log_points": beta,
        "se": se,
        "p_value": float(result.pvalues["did_post"]),
        "ci95_low": beta - 1.96 * se,
        "ci95_high": beta + 1.96 * se,
        "effect_pct": 100 * (math.exp(beta) - 1),
        "nobs": int(result.nobs),
        "clusters": int(data["cno4"].nunique()),
        "periods": int(data["period"].nunique()),
        "notes": "Age rows are summed into the v2 three-bucket construction before logs are computed.",
    }


def run_age3_twfe(age3: pd.DataFrame, threshold: float) -> str:
    design = add_binary_design(age3, threshold)
    rows = []
    for bucket in ["<18 to 29", "30-39", "40 to >44"]:
        rows.append(fit_clustered_twfe(design, bucket, "ln_parados"))
        rows.append(fit_clustered_twfe(design, bucket, "ln_contratos"))
    out = pd.DataFrame(rows)
    path = TABLES / "table_memo_v2_age3_twfe.csv"
    out.to_csv(path, index=False)
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def main() -> None:
    ensure_dirs()
    total = pd.read_csv(PROCESSED / "analysis_panel.csv", dtype={"cno4": "string"})
    total = add_outcomes(total)
    threshold = float(total[EXPOSURE].quantile(0.75))
    total = exposure_groups(total, threshold)

    age = load_dimension("age")
    age3 = aggregate_age_buckets(age)
    age3 = exposure_groups(age3, threshold)

    outputs = []
    outputs.extend(plot_indexed(total, [], "total", "Total CNO4 panel"))
    outputs.extend(plot_indexed(age3, ["age_bucket"], "age3", "Age-bucket CNO4 panel"))
    age3_twfe = run_age3_twfe(age3, threshold)

    metadata = {
        "figures": outputs,
        "tables": [
            "output/tables/table_memo_v2_indexed_inputs_total.csv",
            "output/tables/table_memo_v2_indexed_inputs_age3.csv",
            age3_twfe,
        ],
        "age_buckets": AGE_BUCKETS,
        "threshold_p75": threshold,
        "note": "Age rows are summed into three buckets before logs are computed. Indexed outcomes are relative to each exposure group's Jan-Aug 2022 mean.",
    }
    (PROCESSED / "memo_v2_assets_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
