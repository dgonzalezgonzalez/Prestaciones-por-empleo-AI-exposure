from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

from run_continuous_event_studies import (
    EVENT_MONTH_ID,
    EXPOSURE,
    FIGURES,
    PROCESSED,
    RAW_CSV,
    TABLES,
    event_var_name,
    month_id,
    multi_regressor_absorbed,
)
from run_synthetic_methods import solve_simplex, solve_time_weights


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AGE_BUCKETS = {
    "<18": "<18 to 29",
    "18-24": "<18 to 29",
    "25-29": "<18 to 29",
    "30-39": "30-39",
    "40-44": "40 to >44",
    ">44": "40 to >44",
}


@dataclass
class EventSample:
    sample: str
    label: str
    data: pd.DataFrame
    outcome: str
    unit: str
    cluster: str
    max_donors: int = 600


def ensure_dirs() -> None:
    for path in [PROCESSED, TABLES, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def safe_name(text: str) -> str:
    return (
        str(text)
        .lower()
        .replace("<", "lt")
        .replace(">", "gt")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("::", "_")
    )


def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "month_id" not in out.columns:
        out["month_id"] = month_id(out["period"])
    out["parados"] = pd.to_numeric(out["parados"], errors="coerce")
    out["ln_parados"] = np.nan
    out.loc[out["parados"] > 0, "ln_parados"] = np.log(out.loc[out["parados"] > 0, "parados"])
    out["ln_parados_p1"] = np.log(out["parados"].fillna(0) + 1)
    return out


def read_dimension(dimension: str) -> pd.DataFrame:
    usecols = ["period", "cno4", "occupation_title", "dimension", "category", "gender", "parados", EXPOSURE]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(RAW_CSV, usecols=usecols, dtype={"cno4": "string"}, chunksize=500_000):
        chunk["cno4"] = chunk["cno4"].str.zfill(4)
        keep = chunk["cno4"].str.len().eq(4) & chunk["dimension"].eq(dimension)
        chunks.append(chunk.loc[keep].copy())
    return add_outcomes(pd.concat(chunks, ignore_index=True))


def exposure_threshold() -> float:
    total = pd.read_csv(PROCESSED / "analysis_panel.csv", dtype={"cno4": "string"})
    return float(total[EXPOSURE].quantile(0.75))


def add_treatment(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["treat"] = out[EXPOSURE] > threshold
    out["control"] = out[EXPOSURE] == 0
    out = out.loc[out["treat"] | out["control"]].copy()
    out["event_time"] = out["month_id"] - EVENT_MONTH_ID
    return out


def aggregate_age_buckets(age: pd.DataFrame) -> pd.DataFrame:
    out = age.copy()
    out["age_bucket"] = out["category"].map(AGE_BUCKETS)
    out = out.loc[out["age_bucket"].notna()].copy()
    grouped = (
        out.groupby(["period", "cno4", "occupation_title", "age_bucket"], as_index=False)
        .agg(
            parados=("parados", "sum"),
            observed_exposure_cosine_nearest=(EXPOSURE, "first"),
        )
    )
    return add_outcomes(grouped)


def build_event_samples(threshold: float) -> list[EventSample]:
    total = pd.read_csv(PROCESSED / "analysis_panel.csv", dtype={"cno4": "string"})
    total["cno4"] = total["cno4"].str.zfill(4)
    total = add_outcomes(total)
    total["unit"] = total["cno4"]

    province = read_dimension("province")
    province["unit"] = province["category"].astype(str) + "::" + province["cno4"].astype(str)

    age3 = aggregate_age_buckets(read_dimension("age"))
    gender = read_dimension("gender")

    samples = [
        EventSample("total_cno4", "Total CNO4", total, "ln_parados", "unit", "cno4"),
        EventSample("province_cno4", "Province-CNO4", province, "ln_parados_p1", "unit", "cno4"),
    ]

    for bucket in ["<18 to 29", "30-39", "40 to >44"]:
        sub = age3.loc[age3["age_bucket"].eq(bucket)].copy()
        sub["unit"] = sub["cno4"]
        samples.append(
            EventSample(
                f"age3_{safe_name(bucket)}",
                f"Age {bucket}",
                sub,
                "ln_parados",
                "unit",
                "cno4",
            )
        )

    for category, label in [("Hombre", "Men"), ("Mujer", "Women")]:
        sub = gender.loc[gender["category"].eq(category)].copy()
        sub["unit"] = sub["cno4"]
        samples.append(
            EventSample(
                f"gender_{safe_name(category)}",
                label,
                sub,
                "ln_parados",
                "unit",
                "cno4",
            )
        )

    return [EventSample(s.sample, s.label, add_treatment(s.data, threshold), s.outcome, s.unit, s.cluster, s.max_donors) for s in samples]


def run_twfe_event_study(sample: EventSample) -> tuple[pd.DataFrame, dict]:
    df = sample.data.dropna(subset=[sample.outcome, sample.unit, "period", sample.cluster]).copy()
    event_times = sorted(int(x) for x in df["event_time"].unique())
    terms = []
    for t in event_times:
        if t == -1:
            continue
        name = event_var_name(t).replace("cevent", "memo_bevent")
        df[name] = df["treat"].astype(int) * df["event_time"].eq(t).astype(int)
        terms.append(name)

    estimates, absorb_meta = multi_regressor_absorbed(
        df,
        outcome=sample.outcome,
        regressors=terms,
        fe_cols=[sample.unit, "period"],
        cluster_col=sample.cluster,
    )
    term_index = estimates.set_index("term")
    rows = []
    for t in event_times:
        if t == -1:
            rows.append({"event_time": -1, "term": "reference", "estimate": 0.0, "se": np.nan, "p_value": np.nan})
        else:
            term = event_var_name(t).replace("cevent", "memo_bevent")
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
    out["sample"] = sample.sample
    out["label"] = sample.label
    out["outcome"] = sample.outcome

    pre_terms = [event_var_name(t).replace("cevent", "memo_bevent") for t in event_times if t <= -2]
    pre_idx = [terms.index(term) for term in pre_terms]
    b = term_index.loc[pre_terms, "estimate"].to_numpy()
    cov = absorb_meta["cov"][np.ix_(pre_idx, pre_idx)]
    wald = float(b.T @ np.linalg.pinv(cov) @ b)
    pretrend_p = float(1 - st.chi2.cdf(wald, len(pre_terms)))
    meta = {
        "sample": sample.sample,
        "label": sample.label,
        "outcome": sample.outcome,
        "nobs": absorb_meta["nobs"],
        "clusters": absorb_meta["clusters"],
        "units": absorb_meta["units"],
        "periods": absorb_meta["periods"],
        "pretrend_wald_chi2": wald,
        "pretrend_df": len(pre_terms),
        "pretrend_p_value": pretrend_p,
    }
    return out, meta


def plot_twfe_event(coefs: pd.DataFrame, meta: dict) -> str:
    plot = coefs.sort_values("event_time")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
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
    ax.set_title(f"TWFE event study: {meta['label']}")
    ax.set_xlabel("Months relative to September 2022")
    ax.set_ylabel("Log-point effect")
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
    path = FIGURES / f"figure_memo_v2_twfe_event_{meta['sample']}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def run_sdid_event(sample: EventSample) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = sample.data.dropna(subset=[sample.outcome, sample.unit, "period", "month_id"]).copy()
    unit_treatment = df.groupby(sample.unit)["treat"].first()
    wide = df.pivot_table(index=sample.unit, columns="month_id", values=sample.outcome, aggfunc="mean").dropna(axis=0)
    unit_treatment = unit_treatment.loc[wide.index]
    periods = np.array(wide.columns.tolist(), dtype=int)
    pre_mask = periods <= EVENT_MONTH_ID
    post_mask = periods > EVENT_MONTH_ID

    treated_units = wide.index[unit_treatment.to_numpy(dtype=bool)].tolist()
    control_units = wide.index[(~unit_treatment.to_numpy(dtype=bool))].tolist()
    if not treated_units or not control_units:
        raise ValueError(f"{sample.sample} has no treated or control units after balancing.")

    y_treat = wide.loc[treated_units].to_numpy(dtype=float)
    y_control_all = wide.loc[control_units].to_numpy(dtype=float)
    treated_pre = y_treat[:, pre_mask].mean(axis=0)
    distances = np.sqrt(((y_control_all[:, pre_mask] - treated_pre) ** 2).mean(axis=1))
    keep_idx = np.argsort(distances)[: min(sample.max_donors, len(control_units))]
    selected_controls = [control_units[i] for i in keep_idx]
    y_control = y_control_all[keep_idx]

    unit_weights, unit_alpha, unit_solver = solve_simplex(
        y_control[:, pre_mask],
        treated_pre,
        ridge=1e-6,
        intercept=True,
    )
    time_weights, time_alpha, time_solver = solve_time_weights(
        y_control[:, pre_mask],
        y_control[:, post_mask].mean(axis=1),
        ridge=1e-6,
    )

    treated_path = y_treat.mean(axis=0)
    control_path = unit_alpha + unit_weights @ y_control
    raw_gap = treated_path - control_path
    weighted_pre_gap = float(time_weights @ raw_gap[pre_mask])
    centered_gap = raw_gap - weighted_pre_gap
    sdid_att = float(centered_gap[post_mask].mean())
    pre_rmspe = float(np.sqrt(np.mean(centered_gap[pre_mask] ** 2)))

    path = pd.DataFrame(
        {
            "sample": sample.sample,
            "label": sample.label,
            "outcome": sample.outcome,
            "month_id": periods,
            "event_time": periods - EVENT_MONTH_ID,
            "treated_mean": treated_path,
            "sdid_control_mean": control_path,
            "raw_gap": raw_gap,
            "weighted_pre_gap": weighted_pre_gap,
            "sdid_event_gap": centered_gap,
            "period_type": np.where(pre_mask, "pre", "post"),
        }
    )
    summary = pd.DataFrame(
        [
            {
                "sample": sample.sample,
                "label": sample.label,
                "outcome": sample.outcome,
                "treated_units": len(treated_units),
                "control_units_available": len(control_units),
                "control_units_selected": len(selected_controls),
                "periods": len(periods),
                "pre_periods": int(pre_mask.sum()),
                "post_periods": int(post_mask.sum()),
                "sdid_att_log_points": sdid_att,
                "sdid_effect_pct": 100 * (math.exp(sdid_att) - 1),
                "sdid_pre_rmspe_centered": pre_rmspe,
                "largest_unit_weight": float(unit_weights.max()),
                "effective_donors_inverse_hhi": float(1 / np.sum(unit_weights**2)),
                "unit_intercept": float(unit_alpha),
                "time_intercept": float(time_alpha),
                "unit_solver": unit_solver,
                "time_solver": time_solver,
                "note": "SDID-style event gap equals treated path minus intercept-adjusted SDID unit-weighted control path, centered by the SDID time-weighted pre-period gap.",
            }
        ]
    )
    return path, summary


def plot_sdid_event(path: pd.DataFrame, summary: pd.Series) -> str:
    plot = path.sort_values("event_time")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.axhline(0, color="#777777", lw=0.9)
    ax.axvline(0, color="#444444", ls="--", lw=1)
    ax.plot(plot["event_time"], plot["sdid_event_gap"], color="#325d79", lw=1.8)
    ax.scatter(plot["event_time"], plot["sdid_event_gap"], color="#325d79", s=14)
    ax.set_title(f"SDID-style event gap: {summary['label']}")
    ax.set_xlabel("Months relative to September 2022")
    ax.set_ylabel("Centered treated-synthetic gap")
    ax.text(
        0.01,
        0.03,
        f"Post ATT={summary['sdid_att_log_points']:.3f}; pre RMSPE={summary['sdid_pre_rmspe_centered']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout()
    path_out = FIGURES / f"figure_memo_v2_sdid_event_{summary['sample']}.png"
    fig.savefig(path_out, bbox_inches="tight")
    plt.close(fig)
    return str(path_out.relative_to(PROJECT_ROOT)).replace("\\", "/")


def main() -> None:
    ensure_dirs()
    threshold = exposure_threshold()
    samples = build_event_samples(threshold)

    twfe_coefs = []
    twfe_meta = []
    sdid_paths = []
    sdid_summary = []
    figures = []

    for sample in samples:
        if sample.sample != "province_cno4":
            coefs, meta = run_twfe_event_study(sample)
            fig_path = plot_twfe_event(coefs, meta)
            meta["figure"] = fig_path
            twfe_coefs.append(coefs)
            twfe_meta.append(meta)
            figures.append(fig_path)

        path, summary = run_sdid_event(sample)
        summary_row = summary.iloc[0].copy()
        fig_path = plot_sdid_event(path, summary_row)
        summary_row["figure"] = fig_path
        sdid_paths.append(path)
        sdid_summary.append(summary_row.to_dict())
        figures.append(fig_path)

    pd.concat(twfe_coefs, ignore_index=True).to_csv(TABLES / "table_memo_v2_twfe_event_coefficients.csv", index=False)
    pd.DataFrame(twfe_meta).to_csv(TABLES / "table_memo_v2_twfe_event_summary.csv", index=False)
    pd.concat(sdid_paths, ignore_index=True).to_csv(TABLES / "table_memo_v2_sdid_event_paths.csv", index=False)
    pd.DataFrame(sdid_summary).to_csv(TABLES / "table_memo_v2_sdid_event_summary.csv", index=False)

    metadata = {
        "figures": figures,
        "tables": [
            "output/tables/table_memo_v2_twfe_event_coefficients.csv",
            "output/tables/table_memo_v2_twfe_event_summary.csv",
            "output/tables/table_memo_v2_sdid_event_paths.csv",
            "output/tables/table_memo_v2_sdid_event_summary.csv",
        ],
        "threshold_p75": threshold,
        "note": "Traditional graphs are TWFE event studies. SDID event graphs are centered treated-synthetic gaps using SDID-style unit and time weights.",
    }
    (PROCESSED / "memo_v2_event_assets_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
