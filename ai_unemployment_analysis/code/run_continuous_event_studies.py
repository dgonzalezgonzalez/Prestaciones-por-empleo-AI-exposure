from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "sepe_cno4_monthly_ai_exposure.csv"
PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES = PROJECT_ROOT / "output" / "tables"
FIGURES = PROJECT_ROOT / "output" / "figures"

EXPOSURE = "observed_exposure_cosine_nearest"
EVENT_MONTH_ID = 2022 * 12 + 9


def ensure_dirs() -> None:
    for path in [PROCESSED, TABLES, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def month_id(period: pd.Series) -> pd.Series:
    return period.str.slice(0, 4).astype(int) * 12 + period.str.slice(5, 7).astype(int)


def read_dimension(dimension: str) -> pd.DataFrame:
    usecols = [
        "period",
        "cno4",
        "occupation_title",
        "dimension",
        "category",
        "gender",
        "contratos",
        "parados",
        EXPOSURE,
    ]
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


def event_var_name(t: int) -> str:
    if t < 0:
        return f"cevent_m{abs(t)}"
    if t == 0:
        return "cevent_0"
    return f"cevent_p{t}"


def demean_multiway(values: np.ndarray, group_codes: list[np.ndarray], max_iter: int = 200, tol: float = 1e-10) -> tuple[np.ndarray, int]:
    resid = values.astype(float).copy()
    counts = [np.bincount(codes) for codes in group_codes]
    for iteration in range(1, max_iter + 1):
        old = resid.copy()
        for codes, count in zip(group_codes, counts):
            sums = np.bincount(codes, weights=resid, minlength=len(count))
            means = sums / np.where(count == 0, 1, count)
            resid -= means[codes]
        if float(np.max(np.abs(resid - old))) < tol:
            return resid, iteration
    return resid, max_iter


def multi_regressor_absorbed(
    df: pd.DataFrame,
    outcome: str,
    regressors: list[str],
    fe_cols: list[str],
    cluster_col: str,
) -> tuple[pd.DataFrame, dict]:
    data = df.dropna(subset=[outcome, cluster_col, *fe_cols, *regressors]).copy().reset_index(drop=True)
    y = data[outcome].to_numpy(dtype=float)
    x = data[regressors].to_numpy(dtype=float)
    group_codes = [pd.factorize(data[col], sort=False)[0] for col in fe_cols]
    y_res, y_iters = demean_multiway(y, group_codes)
    x_res = []
    x_iters = []
    for i in range(x.shape[1]):
        resid, iters = demean_multiway(x[:, i], group_codes)
        x_res.append(resid)
        x_iters.append(iters)
    x_res = np.column_stack(x_res)
    xtx = x_res.T @ x_res
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (x_res.T @ y_res)
    u = y_res - x_res @ beta
    cluster_codes = pd.Series(data[cluster_col].to_numpy())
    meat = np.zeros((len(regressors), len(regressors)))
    for _, idx in cluster_codes.groupby(cluster_codes).groups.items():
        pos = np.array(list(idx), dtype=int)
        score = x_res[pos].T @ u[pos]
        meat += np.outer(score, score)
    n = len(data)
    g = int(cluster_codes.nunique())
    fe_rank_proxy = sum(data[col].nunique() for col in fe_cols) - len(fe_cols) + 1
    k_params = min(n - 1, fe_rank_proxy + len(regressors))
    correction = (g / (g - 1)) * ((n - 1) / (n - k_params)) if g > 1 and n > k_params else 1.0
    cov = correction * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.diag(cov))
    rows = []
    for i, regressor in enumerate(regressors):
        p_value = float(2 * (1 - st.norm.cdf(abs(beta[i] / se[i])))) if se[i] else np.nan
        rows.append(
            {
                "term": regressor,
                "estimate": float(beta[i]),
                "se": float(se[i]),
                "p_value": p_value,
            }
        )
    meta = {
        "nobs": n,
        "clusters": g,
        "units": int(data[fe_cols[0]].nunique()),
        "periods": int(data[fe_cols[1]].nunique()),
        "cov": cov,
        "regressors": regressors,
        "demean_iterations_y": y_iters,
        "demean_iterations_x_max": max(x_iters),
    }
    return pd.DataFrame(rows), meta


def build_samples() -> list[dict]:
    total = pd.read_csv(PROCESSED / "analysis_panel.csv", dtype={"cno4": "string"})
    total["unit"] = total["cno4"]

    province = read_dimension("province")
    province["unit"] = province["category"].astype(str) + "::" + province["cno4"].astype(str)

    gender = read_dimension("gender")
    gender["unit"] = gender["category"].astype(str) + "::" + gender["cno4"].astype(str)

    age = read_dimension("age")
    age["unit"] = age["category"].astype(str) + "::" + age["cno4"].astype(str)

    unavailable = pd.DataFrame(
        [
            {
                "sample": "sample5_province_age_ocup4d",
                "status": "not_available",
                "reason": "Current SEPE CSV has province and age as separate dimensions; no joint province-by-age CNO4 cells.",
            },
            {
                "sample": "sample6_gender_age_ocup4d",
                "status": "not_available",
                "reason": "Current SEPE CSV has gender and age as separate dimensions; no joint gender-by-age CNO4 cells.",
            },
        ]
    )
    unavailable.to_csv(TABLES / "table_continuous_event_study_unavailable_samples.csv", index=False)

    return [
        {
            "sample": "sample1_total_ocup4d",
            "data": total,
            "outcome": "ln_parados",
            "unit": "unit",
            "cluster": "cno4",
            "note": "National total CNO4 panel.",
        },
        {
            "sample": "sample2_province_ocup4d",
            "data": province,
            "outcome": "ln_parados_p1",
            "unit": "unit",
            "cluster": "cno4",
            "note": "Province-CNO4 panel; log(+1) because province cells often contain zeros.",
        },
        {
            "sample": "sample3_gender_ocup4d",
            "data": gender,
            "outcome": "ln_parados",
            "unit": "unit",
            "cluster": "cno4",
            "note": "Gender-CNO4 panel.",
        },
        {
            "sample": "sample4_age_ocup4d",
            "data": age,
            "outcome": "ln_parados",
            "unit": "unit",
            "cluster": "cno4",
            "note": "Age-band-CNO4 panel.",
        },
    ]


def run_continuous_event_study(sample: dict) -> tuple[pd.DataFrame, dict]:
    df = sample["data"].dropna(subset=[sample["outcome"], EXPOSURE, sample["unit"], "period", "cno4"]).copy()
    df["event_time"] = df["month_id"] - EVENT_MONTH_ID
    df["exposure_10pp"] = df[EXPOSURE] / 0.10
    event_times = sorted(int(x) for x in df["event_time"].unique())
    terms = []
    for t in event_times:
        if t == -1:
            continue
        name = event_var_name(t)
        df[name] = df["exposure_10pp"] * df["event_time"].eq(t).astype(int)
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
            rows.append({"event_time": -1, "term": "reference", "estimate": 0.0, "se": np.nan})
        else:
            term = event_var_name(t)
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
    out["note"] = sample["note"]

    pre_terms = [event_var_name(t) for t in event_times if t <= -2]
    pre_idx = [terms.index(term) for term in pre_terms]
    b = term_index.loc[pre_terms, "estimate"].to_numpy()
    cov = absorb_meta["cov"][np.ix_(pre_idx, pre_idx)]
    wald = float(b.T @ np.linalg.pinv(cov) @ b)
    pretrend_p = float(1 - st.chi2.cdf(wald, len(pre_terms)))
    meta = {
        "sample": sample["sample"],
        "status": "estimated",
        "outcome": sample["outcome"],
        "unit_fe": sample["unit"],
        "time_fe": "period",
        "cluster": sample["cluster"],
        "nobs": absorb_meta["nobs"],
        "clusters": absorb_meta["clusters"],
        "units": absorb_meta["units"],
        "periods": absorb_meta["periods"],
        "pretrend_wald_chi2": wald,
        "pretrend_df": len(pre_terms),
        "pretrend_p_value": pretrend_p,
        "note": sample["note"],
    }
    return out, meta


def plot_event_study(coefs: pd.DataFrame, meta: dict) -> None:
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
        color="#325d79",
        ecolor="#8aa7ba",
        capsize=2,
    )
    ax.set_title(f"Continuous exposure event study: {meta['sample']}")
    ax.set_xlabel("Months relative to September 2022")
    ax.set_ylabel("Log-point effect per 10pp exposure")
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
    fig.savefig(FIGURES / f"figure_continuous_event_study_{meta['sample']}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    coef_tables = []
    meta_rows = []
    for sample in build_samples():
        coefs, meta = run_continuous_event_study(sample)
        coef_tables.append(coefs)
        meta_rows.append(meta)
        coefs.to_csv(TABLES / f"table_continuous_event_study_{sample['sample']}.csv", index=False)
        plot_event_study(coefs, meta)
    all_coefs = pd.concat(coef_tables, ignore_index=True)
    all_coefs.to_csv(TABLES / "table_continuous_event_study_all_samples.csv", index=False)
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(TABLES / "table_continuous_event_study_summary.csv", index=False)
    metadata = {
        "summary": "output/tables/table_continuous_event_study_summary.csv",
        "coefficients": "output/tables/table_continuous_event_study_all_samples.csv",
        "unavailable": "output/tables/table_continuous_event_study_unavailable_samples.csv",
        "note": "Continuous-treatment event studies use exposure per 10pp interacted with event-month indicators, unit FE, and month FE. This is the estimable TWFE analogue to contdid in this same-treatment-timing setting.",
    }
    (PROCESSED / "continuous_event_study_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
