from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

from run_continuous_event_studies import (
    EVENT_MONTH_ID,
    EXPOSURE,
    PROCESSED,
    RAW_CSV,
    TABLES,
    FIGURES,
    demean_multiway,
    event_var_name,
    month_id,
    multi_regressor_absorbed,
)


def ensure_dirs() -> None:
    for path in [PROCESSED, TABLES, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def read_dimension(dimension: str) -> pd.DataFrame:
    usecols = ["period", "cno4", "dimension", "category", "gender", "parados", EXPOSURE]
    chunks = []
    for chunk in pd.read_csv(RAW_CSV, usecols=usecols, dtype={"cno4": "string"}, chunksize=500_000):
        chunk["cno4"] = chunk["cno4"].str.zfill(4)
        keep = chunk["cno4"].str.len().eq(4) & chunk["dimension"].eq(dimension)
        chunks.append(chunk.loc[keep].copy())
    out = pd.concat(chunks, ignore_index=True)
    out["month_id"] = month_id(out["period"])
    out["parados"] = pd.to_numeric(out["parados"], errors="coerce")
    out["ln_parados"] = np.nan
    positive = out["parados"] > 0
    out.loc[positive, "ln_parados"] = np.log(out.loc[positive, "parados"])
    out["ln_parados_p1"] = np.log(out["parados"].fillna(0) + 1)
    return out


def add_treatment(df: pd.DataFrame) -> pd.DataFrame:
    total = pd.read_csv(PROCESSED / "analysis_panel.csv", dtype={"cno4": "string"})
    threshold = float(total[EXPOSURE].quantile(0.75))
    out = df.copy()
    out["treat"] = out[EXPOSURE] > threshold
    out["control"] = out[EXPOSURE] == 0
    out = out.loc[out["treat"] | out["control"]].copy()
    out["event_time"] = out["month_id"] - EVENT_MONTH_ID
    return out


def samples() -> list[dict]:
    total = pd.read_csv(PROCESSED / "analysis_panel.csv", dtype={"cno4": "string"})
    total["unit"] = total["cno4"]

    province = read_dimension("province")
    province["unit"] = province["category"].astype(str) + "::" + province["cno4"].astype(str)

    gender = read_dimension("gender")
    gender["unit"] = gender["category"].astype(str) + "::" + gender["cno4"].astype(str)

    age = read_dimension("age")
    age["unit"] = age["category"].astype(str) + "::" + age["cno4"].astype(str)

    age3039 = age.loc[age["category"].eq("30-39")].copy()
    age3039["unit"] = age3039["cno4"]

    unavailable = pd.DataFrame(
        [
            {
                "sample": "sample5_province_age_ocup4d",
                "status": "not_available",
                "reason": "No joint province-by-age CNO4 cells in current CSV.",
            },
            {
                "sample": "sample6_gender_age_ocup4d",
                "status": "not_available",
                "reason": "No joint gender-by-age CNO4 cells in current CSV.",
            },
        ]
    )
    unavailable.to_csv(TABLES / "table_binary_event_study_unavailable_samples.csv", index=False)

    return [
        {"sample": "sample1_total_ocup4d", "data": total, "outcome": "ln_parados", "unit": "unit", "cluster": "cno4"},
        {"sample": "sample2_province_ocup4d", "data": province, "outcome": "ln_parados_p1", "unit": "unit", "cluster": "cno4"},
        {"sample": "sample3_gender_ocup4d", "data": gender, "outcome": "ln_parados", "unit": "unit", "cluster": "cno4"},
        {"sample": "sample4_age_ocup4d", "data": age, "outcome": "ln_parados", "unit": "unit", "cluster": "cno4"},
        {"sample": "sample4_age3039_ocup4d", "data": age3039, "outcome": "ln_parados", "unit": "unit", "cluster": "cno4"},
    ]


def run_event_study(sample: dict) -> tuple[pd.DataFrame, dict]:
    df = add_treatment(sample["data"]).dropna(subset=[sample["outcome"], sample["unit"], "period", "cno4"]).copy()
    event_times = sorted(int(x) for x in df["event_time"].unique())
    terms = []
    for t in event_times:
        if t == -1:
            continue
        name = event_var_name(t).replace("cevent", "bevent")
        df[name] = df["treat"].astype(int) * df["event_time"].eq(t).astype(int)
        terms.append(name)

    estimates, absorb_meta = multi_regressor_absorbed(
        df,
        outcome=sample["outcome"],
        regressors=terms,
        fe_cols=[sample["unit"], "period"],
        cluster_col=sample["cluster"],
    )
    term_index = estimates.set_index("term")
    rows = []
    for t in event_times:
        if t == -1:
            rows.append({"event_time": -1, "term": "reference", "estimate": 0.0, "se": np.nan, "p_value": np.nan})
        else:
            term = event_var_name(t).replace("cevent", "bevent")
            rows.append(
                {
                    "event_time": t,
                    "term": term,
                    "estimate": float(term_index.loc[term, "estimate"]),
                    "se": float(term_index.loc[term, "se"]),
                    "p_value": float(term_index.loc[term, "p_value"]),
                }
            )
    out = pd.DataFrame(rows)
    out["ci95_low"] = out["estimate"] - 1.96 * out["se"]
    out["ci95_high"] = out["estimate"] + 1.96 * out["se"]
    out["sample"] = sample["sample"]
    out["outcome"] = sample["outcome"]
    pre_terms = [event_var_name(t).replace("cevent", "bevent") for t in event_times if t <= -2]
    pre_idx = [terms.index(term) for term in pre_terms]
    b = term_index.loc[pre_terms, "estimate"].to_numpy()
    cov = absorb_meta["cov"][np.ix_(pre_idx, pre_idx)]
    wald = float(b.T @ np.linalg.pinv(cov) @ b)
    pretrend_p = float(1 - st.chi2.cdf(wald, len(pre_terms)))
    meta = {
        "sample": sample["sample"],
        "outcome": sample["outcome"],
        "nobs": absorb_meta["nobs"],
        "clusters": absorb_meta["clusters"],
        "units": absorb_meta["units"],
        "periods": absorb_meta["periods"],
        "pretrend_wald_chi2": wald,
        "pretrend_df": len(pre_terms),
        "pretrend_p_value": pretrend_p,
    }
    return out, meta


def plot_event(coefs: pd.DataFrame, meta: dict) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    plot = coefs.sort_values("event_time")
    ax.axhline(0, color="#777777", lw=0.9)
    ax.axvline(0, color="#444444", ls="--", lw=1)
    ax.errorbar(
        plot["event_time"],
        plot["estimate"],
        yerr=[plot["estimate"] - plot["ci95_low"], plot["ci95_high"] - plot["estimate"]],
        fmt="o",
        ms=3.5,
        color="#8f2445",
        ecolor="#bd8da0",
        capsize=2,
    )
    ax.set_title(f"Binary event study: {meta['sample']}")
    ax.set_xlabel("Months relative to September 2022")
    ax.set_ylabel("Log-point effect on parados")
    ax.text(
        0.01,
        0.03,
        f"Reference month: -1; pretrend p={meta['pretrend_p_value']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / f"figure_binary_event_study_{meta['sample']}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    coefs_all = []
    meta_rows = []
    for sample in samples():
        coefs, meta = run_event_study(sample)
        coefs_all.append(coefs)
        meta_rows.append(meta)
        coefs.to_csv(TABLES / f"table_binary_event_study_{sample['sample']}.csv", index=False)
        plot_event(coefs, meta)
    pd.concat(coefs_all, ignore_index=True).to_csv(TABLES / "table_binary_event_study_all_samples.csv", index=False)
    pd.DataFrame(meta_rows).to_csv(TABLES / "table_binary_event_study_summary.csv", index=False)
    metadata = {
        "summary": "output/tables/table_binary_event_study_summary.csv",
        "coefficients": "output/tables/table_binary_event_study_all_samples.csv",
        "unavailable": "output/tables/table_binary_event_study_unavailable_samples.csv",
    }
    (PROCESSED / "binary_event_study_samples_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
