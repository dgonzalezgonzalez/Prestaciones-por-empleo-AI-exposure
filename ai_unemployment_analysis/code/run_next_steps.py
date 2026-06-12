from __future__ import annotations

import json
import math
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf
from honestdid import (
    computeConditionalCS_DeltaRM,
    computeConditionalCS_DeltaSD,
    constructOriginalCS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "sepe_cno4_monthly_ai_exposure.csv"
EPA_65134 = PROJECT_ROOT / "data" / "raw" / "ine_epa_ocupados_65134.csv"
MATCH_NEAREST = PROJECT_ROOT / "data" / "raw" / "spanish_occupation_matches_cosine_nearest.csv"
MATCH_WEIGHTED = PROJECT_ROOT / "data" / "raw" / "spanish_occupation_matches_cosine_weighted.csv"
PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES = PROJECT_ROOT / "output" / "tables"
MODELS = PROJECT_ROOT / "output" / "models"
FIGURES = PROJECT_ROOT / "output" / "figures"

EXPOSURE = "observed_exposure_cosine_nearest"
EVENT_MONTH = "2022-09"
EVENT_MONTH_ID = 2022 * 12 + 9


@dataclass
class SimpleSpec:
    name: str
    formula: str
    data: pd.DataFrame
    coefficient: str
    outcome: str
    cluster: str
    notes: str = ""


def ensure_dirs() -> None:
    for path in [PROCESSED, TABLES, MODELS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def period_to_month_id(period: pd.Series) -> pd.Series:
    return period.str.slice(0, 4).astype(int) * 12 + period.str.slice(5, 7).astype(int)


def month_to_quarter(period: pd.Series) -> pd.Series:
    month = period.str.slice(5, 7).astype(int)
    quarter = ((month - 1) // 3 + 1).astype(str)
    return period.str.slice(0, 4) + "T" + quarter


def event_id(event_month: str) -> int:
    year, month = event_month.split("-")
    return int(year) * 12 + int(month)


def spanish_number_to_float(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"", ".."}:
        return np.nan
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    else:
        text = text.replace(".", "")
    return float(text)


def load_main_panel() -> pd.DataFrame:
    panel_path = PROCESSED / "analysis_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError("Run code/run_analysis.py before code/run_next_steps.py")
    df = pd.read_csv(panel_path, dtype={"cno4": "string"})
    df["cno4"] = df["cno4"].str.zfill(4)
    return df


def read_dimension(dimension: str, usecols_extra: list[str] | None = None) -> pd.DataFrame:
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
        "observed_exposure_cosine_weighted",
        "observed_exposure_rf",
    ]
    if usecols_extra:
        usecols.extend(usecols_extra)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(RAW_CSV, usecols=usecols, dtype={"cno4": "string"}, chunksize=500_000):
        chunk["cno4"] = chunk["cno4"].str.zfill(4)
        keep = chunk["cno4"].str.len().eq(4) & chunk["dimension"].eq(dimension)
        chunks.append(chunk.loc[keep].copy())
    out = pd.concat(chunks, ignore_index=True)
    out["month_id"] = period_to_month_id(out["period"])
    out["quarter"] = month_to_quarter(out["period"])
    out["parados"] = pd.to_numeric(out["parados"], errors="coerce")
    out["contratos"] = pd.to_numeric(out["contratos"], errors="coerce")
    out["ln_parados"] = np.nan
    positive = out["parados"] > 0
    out.loc[positive, "ln_parados"] = np.log(out.loc[positive, "parados"])
    out["ln_parados_p1"] = np.log(out["parados"].fillna(0) + 1)
    out["ln_contratos_p1"] = np.log(out["contratos"].fillna(0) + 1)
    return out


def add_binary_design(df: pd.DataFrame, event_month: str = EVENT_MONTH, exposure: str = EXPOSURE) -> pd.DataFrame:
    threshold = float(load_main_panel()[exposure].quantile(0.75))
    out = df.copy()
    out["treat"] = out[exposure] > threshold
    out["control"] = out[exposure] == 0
    out = out.loc[out["treat"] | out["control"]].copy()
    out["post"] = out["month_id"] > event_id(event_month)
    out["did_post"] = out["treat"].astype(int) * out["post"].astype(int)
    out["event_month"] = event_month
    out["threshold_p75"] = threshold
    out["trend"] = out["month_id"] - out["month_id"].min()
    return out


def fit_clustered(spec: SimpleSpec) -> dict:
    data = spec.data.dropna(subset=[spec.outcome, spec.coefficient, spec.cluster, "period"]).copy()
    result = smf.ols(spec.formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data[spec.cluster], "use_correction": True},
    )
    beta = float(result.params[spec.coefficient])
    se = float(result.bse[spec.coefficient])
    return {
        "spec": spec.name,
        "outcome": spec.outcome,
        "beta_log_points": beta,
        "se": se,
        "p_value": float(result.pvalues[spec.coefficient]),
        "ci95_low": beta - 1.96 * se,
        "ci95_high": beta + 1.96 * se,
        "effect_pct": 100 * (math.exp(beta) - 1),
        "nobs": int(result.nobs),
        "clusters": int(data[spec.cluster].nunique()),
        "periods": int(data["period"].nunique()),
        "notes": spec.notes,
    }


def run_subgroup_models() -> pd.DataFrame:
    rows: list[dict] = []

    gender = add_binary_design(read_dimension("gender"))
    for cat in ["Hombre", "Mujer"]:
        sub = gender.loc[gender["category"].eq(cat)].copy()
        rows.append(
            fit_clustered(
                SimpleSpec(
                    name=f"gender_{cat.lower()}_parados",
                    formula="ln_parados ~ did_post + C(cno4) + C(period)",
                    data=sub,
                    coefficient="did_post",
                    outcome="ln_parados",
                    cluster="cno4",
                    notes=f"Gender subgroup: {cat}",
                )
            )
        )
        rows.append(
            fit_clustered(
                SimpleSpec(
                    name=f"gender_{cat.lower()}_contratos",
                    formula="ln_contratos_p1 ~ did_post + C(cno4) + C(period)",
                    data=sub,
                    coefficient="did_post",
                    outcome="ln_contratos_p1",
                    cluster="cno4",
                    notes=f"Gender subgroup: {cat}",
                )
            )
        )

    age = read_dimension("age")
    age_groups = {
        "<18": ["<18"],
        "18-24": ["18-24"],
        "25-29": ["25-29"],
        "under_30": ["<18", "18-24", "25-29"],
        "30-39": ["30-39"],
        "40-44": ["40-44"],
        ">44": [">44"],
    }
    for label, cats in age_groups.items():
        if len(cats) == 1:
            sub = age.loc[age["category"].eq(cats[0])].copy()
        else:
            sub = (
                age.loc[age["category"].isin(cats)]
                .groupby(["period", "cno4", "occupation_title"], as_index=False)
                .agg(
                    parados=("parados", "sum"),
                    contratos=("contratos", "sum"),
                    observed_exposure_cosine_nearest=(EXPOSURE, "first"),
                    observed_exposure_cosine_weighted=("observed_exposure_cosine_weighted", "first"),
                    observed_exposure_rf=("observed_exposure_rf", "first"),
                    month_id=("month_id", "first"),
                    quarter=("quarter", "first"),
                )
            )
            sub["category"] = label
            sub["ln_parados"] = np.nan
            positive = sub["parados"] > 0
            sub.loc[positive, "ln_parados"] = np.log(sub.loc[positive, "parados"])
            sub["ln_parados_p1"] = np.log(sub["parados"].fillna(0) + 1)
            sub["ln_contratos_p1"] = np.log(sub["contratos"].fillna(0) + 1)
        sub = add_binary_design(sub)
        rows.append(
            fit_clustered(
                SimpleSpec(
                    name=f"age_{label}_parados",
                    formula="ln_parados ~ did_post + C(cno4) + C(period)",
                    data=sub,
                    coefficient="did_post",
                    outcome="ln_parados",
                    cluster="cno4",
                    notes=f"Age subgroup: {label}",
                )
            )
        )
        rows.append(
            fit_clustered(
                SimpleSpec(
                    name=f"age_{label}_contratos",
                    formula="ln_contratos_p1 ~ did_post + C(cno4) + C(period)",
                    data=sub,
                    coefficient="did_post",
                    outcome="ln_contratos_p1",
                    cluster="cno4",
                    notes=f"Age subgroup: {label}",
                )
            )
        )

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "table_next_step_subgroup_heterogeneity.csv", index=False)
    return out


def demean_multiway(values: np.ndarray, group_codes: list[np.ndarray], max_iter: int = 200, tol: float = 1e-10) -> tuple[np.ndarray, int]:
    resid = values.astype(float).copy()
    valid = np.ones(len(resid), dtype=bool)
    counts = [np.bincount(codes[valid]) for codes in group_codes]
    for iteration in range(1, max_iter + 1):
        old = resid.copy()
        for codes, count in zip(group_codes, counts):
            sums = np.bincount(codes[valid], weights=resid[valid], minlength=len(count))
            means = sums / np.where(count == 0, 1, count)
            resid[valid] -= means[codes[valid]]
        delta = float(np.nanmax(np.abs(resid - old)))
        if delta < tol:
            return resid, iteration
    return resid, max_iter


def one_regressor_absorbed(
    df: pd.DataFrame,
    outcome: str,
    regressor: str,
    fe_cols: list[str],
    cluster_col: str,
) -> dict:
    data = df.dropna(subset=[outcome, regressor, cluster_col, *fe_cols]).copy().reset_index(drop=True)
    y = data[outcome].to_numpy(dtype=float)
    x = data[regressor].to_numpy(dtype=float)
    group_codes = [pd.factorize(data[col], sort=False)[0] for col in fe_cols]
    y_res, y_iters = demean_multiway(y, group_codes)
    x_res, x_iters = demean_multiway(x, group_codes)
    beta = float((x_res @ y_res) / (x_res @ x_res))
    u = y_res - beta * x_res
    xtx = float(x_res @ x_res)

    cluster_codes = pd.Series(data[cluster_col].to_numpy())
    meat = 0.0
    for _, idx in cluster_codes.groupby(cluster_codes).groups.items():
        pos = np.array(list(idx), dtype=int)
        score = float(x_res[pos] @ u[pos])
        meat += score * score

    n = len(data)
    g = int(cluster_codes.nunique())
    fe_rank_proxy = sum(data[col].nunique() for col in fe_cols) - len(fe_cols) + 1
    k_params = min(n - 1, fe_rank_proxy + 1)
    correction = (g / (g - 1)) * ((n - 1) / (n - k_params)) if g > 1 and n > k_params else 1.0
    se = float(np.sqrt(correction * meat / (xtx * xtx)))
    z = beta / se if se else np.nan
    p = float(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))) if se else np.nan
    return {
        "beta_log_points": beta,
        "se": se,
        "p_value_normal": p,
        "ci95_low": beta - 1.96 * se,
        "ci95_high": beta + 1.96 * se,
        "effect_pct": 100 * (math.exp(beta) - 1),
        "nobs": n,
        "clusters": g,
        "fe_cols": " + ".join(fe_cols),
        "fe_groups": json.dumps({col: int(data[col].nunique()) for col in fe_cols}),
        "demean_iterations_y": y_iters,
        "demean_iterations_x": x_iters,
    }


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
    x_res_cols = []
    x_iters = {}
    for i, regressor in enumerate(regressors):
        resid, iterations = demean_multiway(x[:, i], group_codes)
        x_res_cols.append(resid)
        x_iters[regressor] = iterations
    x_res = np.column_stack(x_res_cols)
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
                "beta_log_points": float(beta[i]),
                "se": float(se[i]),
                "p_value_normal": p_value,
                "ci95_low": float(beta[i] - 1.96 * se[i]),
                "ci95_high": float(beta[i] + 1.96 * se[i]),
                "effect_pct": float(100 * (math.exp(beta[i]) - 1)),
            }
        )
    metadata = {
        "nobs": n,
        "clusters": g,
        "fe_cols": " + ".join(fe_cols),
        "fe_groups": json.dumps({col: int(data[col].nunique()) for col in fe_cols}),
        "demean_iterations_y": y_iters,
        "demean_iterations_x_max": max(x_iters.values()) if x_iters else 0,
        "cov": cov,
        "regressors": regressors,
    }
    return pd.DataFrame(rows), metadata


def event_var_name(t: int) -> str:
    if t < 0:
        return f"event_m{abs(t)}"
    if t == 0:
        return "event_0"
    return f"event_p{t}"


def run_cno2_month_fe_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = load_main_panel()
    panel["cno2"] = panel["cno4"].str.slice(0, 2)
    panel["cno2_period"] = panel["cno2"] + "::" + panel["period"]

    binary = add_binary_design(panel)
    rows = []
    for outcome in ["ln_parados", "ln_parados_p1", "ln_contratos_p1"]:
        base = one_regressor_absorbed(
            binary,
            outcome=outcome,
            regressor="did_post",
            fe_cols=["cno4", "cno2_period"],
            cluster_col="cno4",
        )
        base.update(
            {
                "spec": f"cno2_month_fe_binary_{outcome}",
                "outcome": outcome,
                "regressor": "did_post",
                "notes": "High exposure vs zero exposure with CNO4 FE and CNO2-by-month FE; clustered by CNO4.",
            }
        )
        rows.append(base)

    continuous = panel.copy()
    continuous["post_sep"] = continuous["month_id"] > EVENT_MONTH_ID
    continuous["exposure_10pp"] = continuous[EXPOSURE] / 0.10
    continuous["exposure_10pp_post"] = continuous["exposure_10pp"] * continuous["post_sep"].astype(int)
    for outcome in ["ln_parados", "ln_contratos_p1"]:
        base = one_regressor_absorbed(
            continuous,
            outcome=outcome,
            regressor="exposure_10pp_post",
            fe_cols=["cno4", "cno2_period"],
            cluster_col="cno4",
        )
        base.update(
            {
                "spec": f"cno2_month_fe_continuous_{outcome}",
                "outcome": outcome,
                "regressor": "exposure_10pp_post",
                "notes": "Continuous exposure per 10pp with CNO4 FE and CNO2-by-month FE; clustered by CNO4.",
            }
        )
        rows.append(base)

    spec_table = pd.DataFrame(rows)
    spec_table.to_csv(TABLES / "table_v1_cno2_month_fe_specifications.csv", index=False)

    event = binary.dropna(subset=["ln_parados"]).copy()
    event["event_time"] = event["month_id"] - EVENT_MONTH_ID
    event_times = sorted(int(x) for x in event["event_time"].unique())
    event_terms = []
    for t in event_times:
        if t == -1:
            continue
        name = event_var_name(t)
        event[name] = event["treat"].astype(int) * event["event_time"].eq(t).astype(int)
        event_terms.append(name)

    es_terms, meta = multi_regressor_absorbed(
        event,
        outcome="ln_parados",
        regressors=event_terms,
        fe_cols=["cno4", "cno2_period"],
        cluster_col="cno4",
    )
    term_to_time = {event_var_name(t): t for t in event_times if t != -1}
    es_terms["event_time"] = es_terms["term"].map(term_to_time)

    pre_terms = [event_var_name(t) for t in event_times if t <= -2]
    pre_idx = [event_terms.index(term) for term in pre_terms]
    b = es_terms.set_index("term").loc[pre_terms, "beta_log_points"].to_numpy()
    cov = meta["cov"][np.ix_(pre_idx, pre_idx)]
    wald = float(b.T @ np.linalg.pinv(cov) @ b)
    pretrend_p = float(1 - st.chi2.cdf(wald, len(pre_terms)))

    ref = pd.DataFrame(
        [
            {
                "term": "reference",
                "beta_log_points": 0.0,
                "se": np.nan,
                "p_value_normal": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "effect_pct": 0.0,
                "event_time": -1,
            }
        ]
    )
    event_study = pd.concat([es_terms, ref], ignore_index=True).sort_values("event_time")
    event_study["pretrend_wald_chi2"] = wald
    event_study["pretrend_df"] = len(pre_terms)
    event_study["pretrend_p_value"] = pretrend_p
    event_study["nobs"] = meta["nobs"]
    event_study["clusters"] = meta["clusters"]
    event_study["fe_cols"] = meta["fe_cols"]
    event_study["fe_groups"] = meta["fe_groups"]
    event_study.to_csv(TABLES / "table_v1_cno2_month_fe_event_study.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    plot_es = event_study.sort_values("event_time")
    ax.axhline(0, color="#777777", lw=0.9)
    ax.axvline(0, color="#444444", ls="--", lw=1)
    ax.errorbar(
        plot_es["event_time"],
        plot_es["beta_log_points"],
        yerr=[
            plot_es["beta_log_points"] - plot_es["ci95_low"],
            plot_es["ci95_high"] - plot_es["beta_log_points"],
        ],
        fmt="o",
        ms=3.5,
        lw=1.0,
        color="#8f2445",
        ecolor="#b98a9b",
        capsize=2,
    )
    ax.set_title("Event study with CNO2-by-month fixed effects")
    ax.set_xlabel("Months relative to September 2022")
    ax.set_ylabel("Log-point effect on parados")
    ax.text(
        0.01,
        0.03,
        f"Reference month: -1; pretrend p={pretrend_p:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_07_v1_cno2_month_fe_event_study.png", bbox_inches="tight")
    plt.close(fig)
    return spec_table, event_study


def load_cno2_labels() -> pd.DataFrame:
    if not EPA_65134.exists() or EPA_65134.stat().st_size <= 1000:
        url = "https://www.ine.es/jaxiT3/files/t/csv_bdsc/65134.csv"
        urllib.request.urlretrieve(url, EPA_65134)
    epa = pd.read_csv(EPA_65134, sep=";", encoding="utf-8")
    occ_col = [col for col in epa.columns if col.startswith("Ocup")][0]
    labels = epa.loc[epa[occ_col].str.match(r"^\d{2}\s", na=False), [occ_col]].drop_duplicates().copy()
    labels["cno2"] = labels[occ_col].str.extract(r"^(\d{2})")
    labels["cno2_label"] = labels[occ_col].str.replace(r"^\d{2}\s+", "", regex=True)
    return labels[["cno2", "cno2_label"]].drop_duplicates("cno2")


def run_recommended_family_steps() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = load_main_panel()
    cno2_labels = load_cno2_labels()
    cno2_label_map = cno2_labels.set_index("cno2")["cno2_label"].to_dict()
    panel["cno2"] = panel["cno4"].str.slice(0, 2)
    data = add_binary_design(panel).dropna(subset=["ln_parados", "did_post", "cno4", "period", "cno2"]).copy()

    group_codes = [pd.factorize(data[col], sort=False)[0] for col in ["cno4", "period"]]
    y_res, _ = demean_multiway(data["ln_parados"].to_numpy(dtype=float), group_codes)
    d_res, _ = demean_multiway(data["did_post"].to_numpy(dtype=float), group_codes)
    denominator = float(d_res @ d_res)
    beta = float((d_res @ y_res) / denominator)
    data["_fwl_num"] = d_res * y_res
    data["_fwl_den"] = d_res * d_res

    contribution = (
        data.groupby("cno2", as_index=False)
        .agg(
            numerator=("_fwl_num", "sum"),
            denominator_part=("_fwl_den", "sum"),
            nobs=("cno4", "size"),
            clusters=("cno4", "nunique"),
            treated_occupations=("treat", lambda x: int(data.loc[x.index].loc[data.loc[x.index, "treat"], "cno4"].nunique())),
            control_occupations=("control", lambda x: int(data.loc[x.index].loc[data.loc[x.index, "control"], "cno4"].nunique())),
            mean_parados=("parados", "mean"),
        )
    )
    contribution["contribution_to_baseline_beta"] = contribution["numerator"] / denominator
    contribution["share_of_beta"] = contribution["contribution_to_baseline_beta"] / beta
    contribution["baseline_beta_reconstructed"] = beta
    contribution = contribution.merge(cno2_labels, on="cno2", how="left")
    contribution = contribution.sort_values("contribution_to_baseline_beta", ascending=False)
    contribution.to_csv(TABLES / "table_recommended_cno2_fwl_decomposition.csv", index=False)

    occ_summary = (
        panel.groupby(["cno2", "cno4"], as_index=False)
        .agg(
            exposure=(EXPOSURE, "first"),
            mean_parados=("parados", "mean"),
            occupation_title=("occupation_title", "first"),
        )
    )
    threshold = float(panel[EXPOSURE].quantile(0.75))
    occ_summary["treat"] = occ_summary["exposure"] > threshold
    occ_summary["control"] = occ_summary["exposure"] == 0
    family_summary = (
        occ_summary.groupby("cno2", as_index=False)
        .agg(
            occupations=("cno4", "nunique"),
            treated_occupations=("treat", "sum"),
            control_occupations=("control", "sum"),
            mean_exposure=("exposure", "mean"),
            mean_parados=("mean_parados", "mean"),
            total_mean_parados=("mean_parados", "sum"),
        )
        .merge(
            contribution[
                [
                    "cno2",
                    "cno2_label",
                    "contribution_to_baseline_beta",
                    "share_of_beta",
                    "baseline_beta_reconstructed",
                ]
            ],
            on="cno2",
            how="left",
        )
        .sort_values("contribution_to_baseline_beta", ascending=False)
    )
    family_summary.to_csv(TABLES / "table_recommended_cno2_family_summary.csv", index=False)

    rows = []
    for cno2, sub in data.groupby("cno2"):
        treated = int(sub.loc[sub["treat"], "cno4"].nunique())
        controls = int(sub.loc[sub["control"], "cno4"].nunique())
        clusters = int(sub["cno4"].nunique())
        if treated < 2 or controls < 2 or clusters < 6:
            continue
        try:
            row = one_regressor_absorbed(
                sub,
                outcome="ln_parados",
                regressor="did_post",
                fe_cols=["cno4", "period"],
                cluster_col="cno4",
            )
            row.update(
                {
                    "cno2": cno2,
                    "cno2_label": cno2_label_map.get(cno2, ""),
                    "spec": "within_cno2_family_binary_ln_parados",
                    "treated_occupations": treated,
                    "control_occupations": controls,
                    "notes": "Within-CNO2 high-vs-zero DiD with CNO4 and month FE; small-cluster estimates are diagnostic only.",
                }
            )
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "cno2": cno2,
                    "cno2_label": cno2_label_map.get(cno2, ""),
                    "spec": "within_cno2_family_binary_ln_parados",
                    "treated_occupations": treated,
                    "control_occupations": controls,
                    "error": str(exc),
                }
            )

    within = pd.DataFrame(rows).sort_values("beta_log_points", ascending=False, na_position="last")
    within.to_csv(TABLES / "table_recommended_within_cno2_family_did.csv", index=False)
    return contribution, family_summary, within


def run_age3039_cno2_month_fe() -> tuple[pd.DataFrame, pd.DataFrame]:
    age = read_dimension("age")
    age3039 = age.loc[age["category"].eq("30-39")].copy()
    age3039["cno2"] = age3039["cno4"].str.slice(0, 2)
    age3039["cno2_period"] = age3039["cno2"] + "::" + age3039["period"]
    binary = add_binary_design(age3039)

    rows = []
    for outcome in ["ln_parados", "ln_parados_p1", "ln_contratos_p1"]:
        base = one_regressor_absorbed(
            binary,
            outcome=outcome,
            regressor="did_post",
            fe_cols=["cno4", "cno2_period"],
            cluster_col="cno4",
        )
        base.update(
            {
                "spec": f"age_30_39_cno2_month_fe_binary_{outcome}",
                "outcome": outcome,
                "notes": "Age 30-39, high-vs-zero treatment, CNO4 FE and CNO2-by-month FE; clustered by CNO4. Literal CNO4-by-month FE would absorb the treatment variation.",
            }
        )
        rows.append(base)
    spec_table = pd.DataFrame(rows)
    spec_table.to_csv(TABLES / "table_recommended_age3039_cno2_month_fe.csv", index=False)

    event = binary.dropna(subset=["ln_parados"]).copy()
    event["event_time"] = event["month_id"] - EVENT_MONTH_ID
    event_times = sorted(int(x) for x in event["event_time"].unique())
    event_terms = []
    for t in event_times:
        if t == -1:
            continue
        name = event_var_name(t)
        event[name] = event["treat"].astype(int) * event["event_time"].eq(t).astype(int)
        event_terms.append(name)

    es_terms, meta = multi_regressor_absorbed(
        event,
        outcome="ln_parados",
        regressors=event_terms,
        fe_cols=["cno4", "cno2_period"],
        cluster_col="cno4",
    )
    term_to_time = {event_var_name(t): t for t in event_times if t != -1}
    es_terms["event_time"] = es_terms["term"].map(term_to_time)
    pre_terms = [event_var_name(t) for t in event_times if t <= -2]
    pre_idx = [event_terms.index(term) for term in pre_terms]
    b = es_terms.set_index("term").loc[pre_terms, "beta_log_points"].to_numpy()
    cov = meta["cov"][np.ix_(pre_idx, pre_idx)]
    wald = float(b.T @ np.linalg.pinv(cov) @ b)
    pretrend_p = float(1 - st.chi2.cdf(wald, len(pre_terms)))
    ref = pd.DataFrame(
        [
            {
                "term": "reference",
                "beta_log_points": 0.0,
                "se": np.nan,
                "p_value_normal": np.nan,
                "ci95_low": np.nan,
                "ci95_high": np.nan,
                "effect_pct": 0.0,
                "event_time": -1,
            }
        ]
    )
    event_study = pd.concat([es_terms, ref], ignore_index=True).sort_values("event_time")
    event_study["pretrend_wald_chi2"] = wald
    event_study["pretrend_df"] = len(pre_terms)
    event_study["pretrend_p_value"] = pretrend_p
    event_study["nobs"] = meta["nobs"]
    event_study["clusters"] = meta["clusters"]
    event_study["fe_cols"] = meta["fe_cols"]
    event_study["fe_groups"] = meta["fe_groups"]
    event_study.to_csv(TABLES / "table_recommended_age3039_cno2_month_fe_event_study.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    plot_es = event_study.sort_values("event_time")
    ax.axhline(0, color="#777777", lw=0.9)
    ax.axvline(0, color="#444444", ls="--", lw=1)
    ax.errorbar(
        plot_es["event_time"],
        plot_es["beta_log_points"],
        yerr=[
            plot_es["beta_log_points"] - plot_es["ci95_low"],
            plot_es["ci95_high"] - plot_es["beta_log_points"],
        ],
        fmt="o",
        ms=3.5,
        lw=1.0,
        color="#8f2445",
        ecolor="#b98a9b",
        capsize=2,
    )
    ax.set_title("Age 30-39 event study with CNO2-by-month fixed effects")
    ax.set_xlabel("Months relative to September 2022")
    ax.set_ylabel("Log-point effect on parados")
    ax.text(
        0.01,
        0.03,
        f"Reference month: -1; pretrend p={pretrend_p:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_08_recommended_age3039_cno2_month_fe_event_study.png", bbox_inches="tight")
    plt.close(fig)
    return spec_table, event_study


def run_honestdid_original_event_study() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = load_main_panel()
    data = add_binary_design(panel).dropna(subset=["ln_parados"]).copy()
    data["event_time"] = data["month_id"] - EVENT_MONTH_ID
    event_times = sorted(int(x) for x in data["event_time"].unique())
    event_terms = []
    for t in event_times:
        if t == -1:
            continue
        name = event_var_name(t)
        data[name] = data["treat"].astype(int) * data["event_time"].eq(t).astype(int)
        event_terms.append(name)

    formula = "ln_parados ~ " + " + ".join(event_terms) + " + C(cno4) + C(period)"
    result = smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["cno4"], "use_correction": True},
    )

    # HonestDiD is applied to the original event-study estimates, focusing on
    # the first post-treatment year. This keeps the Rambachan-Roth grid
    # inversion computationally tractable while using the original graph's
    # pre-period coefficients and first-year post coefficients.
    ordered_times = [t for t in event_times if -12 <= t < -1] + [t for t in event_times if 0 <= t <= 12]
    ordered_terms = [event_var_name(t) for t in ordered_times]
    beta = result.params.loc[ordered_terms].to_numpy(dtype=float)
    sigma = result.cov_params().loc[ordered_terms, ordered_terms].to_numpy(dtype=float)
    num_pre = sum(t < -1 for t in ordered_times)
    num_post = sum(t >= 0 for t in ordered_times)
    post_times = [t for t in ordered_times if t >= 0]

    coef_table = pd.DataFrame(
        {
            "event_time": ordered_times,
            "term": ordered_terms,
            "beta": beta,
            "se": np.sqrt(np.diag(sigma)),
            "num_pre_periods": num_pre,
            "num_post_periods": num_post,
        }
    )
    coef_table.to_csv(TABLES / "table_recommended_honestdid_original_event_coefficients.csv", index=False)
    pd.DataFrame(sigma, index=ordered_terms, columns=ordered_terms).to_csv(
        TABLES / "table_recommended_honestdid_original_event_covariance.csv"
    )

    targets: dict[str, np.ndarray] = {}
    targets["average_event_0_to_12"] = np.ones(num_post) / num_post

    smooth_rows = []
    rm_rows = []
    original_rows = []
    mvec = np.array([0.0, 0.025, 0.05])
    mbarvec = np.array([0.5, 1.0, 2.0])

    def accepted_ci(grid_df: pd.DataFrame) -> tuple[float, float, int]:
        accepted = grid_df.loc[grid_df["accept"].astype(bool), "grid"]
        if accepted.empty:
            return np.nan, np.nan, 0
        return float(accepted.min()), float(accepted.max()), int(accepted.shape[0])
    for target_name, l_vec in targets.items():
        try:
            original = constructOriginalCS(
                betahat=beta,
                sigma=sigma,
                numPrePeriods=num_pre,
                numPostPeriods=num_post,
                l_vec=l_vec,
            )
            if isinstance(original, pd.DataFrame):
                orig_lb = float(original["lb"].iloc[0])
                orig_ub = float(original["ub"].iloc[0])
            else:
                orig_lb = float(original[0])
                orig_ub = float(original[1])
            original_rows.append(
                {
                    "target": target_name,
                    "lb": orig_lb,
                    "ub": orig_ub,
                    "method": "Original parallel trends",
                    "window": "event_time -12 to +12",
                }
            )
        except Exception as exc:
            original_rows.append({"target": target_name, "error": str(exc), "method": "Original parallel trends"})

        try:
            target_rows = []
            for m in mvec:
                grid = computeConditionalCS_DeltaSD(
                    betahat=beta,
                    sigma=sigma,
                    numPrePeriods=num_pre,
                    numPostPeriods=num_post,
                    M=float(m),
                    l_vec=l_vec,
                    hybrid_flag="ARP",
                    gridPoints=80,
                    seed=0,
                )
                lb, ub, accepted_count = accepted_ci(grid)
                target_rows.append(
                    {
                        "target": target_name,
                        "lb": lb,
                        "ub": ub,
                        "method": "Conditional-ARP",
                        "Delta": "DeltaSD",
                        "M": float(m),
                        "accepted_grid_points": accepted_count,
                        "grid_points": 80,
                    }
                )
            smooth_rows.append(pd.DataFrame(target_rows))
        except Exception as exc:
            smooth_rows.append(pd.DataFrame([{"target": target_name, "error": str(exc)}]))

        try:
            target_rows = []
            for mbar in mbarvec:
                grid = computeConditionalCS_DeltaRM(
                    betahat=beta,
                    sigma=sigma,
                    numPrePeriods=num_pre,
                    numPostPeriods=num_post,
                    Mbar=float(mbar),
                    l_vec=l_vec,
                    hybrid_flag="LF",
                    gridPoints=80,
                    seed=0,
                )
                lb, ub, accepted_count = accepted_ci(grid)
                target_rows.append(
                    {
                        "target": target_name,
                        "lb": lb,
                        "ub": ub,
                        "method": "Conditional-LF",
                        "Delta": "DeltaRM",
                        "Mbar": float(mbar),
                        "accepted_grid_points": accepted_count,
                        "grid_points": 80,
                    }
                )
            rm_rows.append(pd.DataFrame(target_rows))
        except Exception as exc:
            rm_rows.append(pd.DataFrame([{"target": target_name, "error": str(exc)}]))

    original_df = pd.DataFrame(original_rows)
    smooth_df = pd.concat(smooth_rows, ignore_index=True)
    rm_df = pd.concat(rm_rows, ignore_index=True)
    original_df.to_csv(TABLES / "table_recommended_honestdid_original_ci.csv", index=False)
    smooth_df.to_csv(TABLES / "table_recommended_honestdid_smoothness.csv", index=False)
    rm_df.to_csv(TABLES / "table_recommended_honestdid_relative_magnitude.csv", index=False)

    plot = smooth_df.loc[(smooth_df["target"].eq("average_event_0_to_12")) & smooth_df["lb"].notna()].copy()
    if not plot.empty:
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        ax.axhline(0, color="#777777", lw=0.9)
        ax.fill_between(plot["M"].astype(float), plot["lb"].astype(float), plot["ub"].astype(float), color="#8aa7ba", alpha=0.35)
        ax.plot(plot["M"].astype(float), plot["lb"].astype(float), color="#325d79", lw=1.4)
        ax.plot(plot["M"].astype(float), plot["ub"].astype(float), color="#325d79", lw=1.4)
        ax.set_title("HonestDiD smoothness sensitivity: original event study")
        ax.set_xlabel("M smoothness parameter")
        ax.set_ylabel("Robust confidence interval, avg. event 0 to 12")
        fig.tight_layout()
        fig.savefig(FIGURES / "figure_09_recommended_honestdid_smoothness_original.png", bbox_inches="tight")
        plt.close(fig)

    return original_df, smooth_df, rm_df


def run_province_models() -> pd.DataFrame:
    province = read_dimension("province")
    province = add_binary_design(province)
    province["cno4_province"] = province["cno4"] + "::" + province["category"]
    province["province_period"] = province["category"] + "::" + province["period"]
    province["province"] = province["category"]
    province.to_csv(PROCESSED / "province_panel_high_vs_zero.csv", index=False)

    rows = []
    for outcome in ["ln_parados_p1", "ln_contratos_p1"]:
        base = one_regressor_absorbed(
            province,
            outcome=outcome,
            regressor="did_post",
            fe_cols=["cno4_province", "province_period"],
            cluster_col="cno4",
        )
        base.update(
            {
                "spec": f"province_fe_{outcome}",
                "outcome": outcome,
                "notes": "CNO4-province FE and province-month FE; cluster by CNO4; log(+1) outcome because province cells often contain zeros.",
            }
        )
        rows.append(base)

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "table_next_step_province_fe.csv", index=False)
    return out


def download_epa_65134() -> None:
    if EPA_65134.exists() and EPA_65134.stat().st_size > 1000:
        return
    url = "https://www.ine.es/jaxiT3/files/t/csv_bdsc/65134.csv"
    urllib.request.urlretrieve(url, EPA_65134)


def run_cno2_denominator_proxy() -> pd.DataFrame:
    download_epa_65134()
    panel = load_main_panel()
    panel["cno2"] = panel["cno4"].str.slice(0, 2)
    panel["quarter"] = month_to_quarter(panel["period"])
    sepe_cno2 = (
        panel.groupby(["cno2", "quarter"], as_index=False)
        .agg(
            mean_parados=("parados", "mean"),
            mean_contratos=("contratos", "mean"),
            exposure_nearest=(EXPOSURE, "mean"),
            cno4_count=("cno4", "nunique"),
        )
    )

    epa = pd.read_csv(EPA_65134, sep=";", encoding="utf-8")
    occ_col = "Ocupación CNO-11"
    epa = epa.loc[
        epa["Sexo"].eq("Ambos sexos")
        & epa["Unidad"].eq("Valor absoluto")
        & epa[occ_col].str.match(r"^\d{2}\s", na=False)
    ].copy()
    epa["cno2"] = epa[occ_col].str.extract(r"^(\d{2})")
    epa["quarter"] = epa["Periodo"]
    epa["epa_occupied_thousands"] = epa["Total"].map(spanish_number_to_float)
    epa["epa_occupied_persons"] = epa["epa_occupied_thousands"] * 1000.0
    epa = epa[["cno2", "quarter", occ_col, "epa_occupied_persons"]].rename(columns={occ_col: "cno2_label"})

    merged = sepe_cno2.merge(epa, on=["cno2", "quarter"], how="inner")
    merged["rate_proxy"] = merged["mean_parados"] / (merged["mean_parados"] + merged["epa_occupied_persons"])
    merged["ln_rate_proxy"] = np.log(merged["rate_proxy"])
    merged["quarter_order"] = merged["quarter"].str.slice(0, 4).astype(int) * 4 + merged["quarter"].str.slice(5, 6).astype(int)
    merged["trend"] = merged["quarter_order"] - merged["quarter_order"].min()
    merged["period"] = merged["quarter"]
    merged["post"] = merged["quarter"].ge("2022T4")
    merged["exposure_10pp"] = merged["exposure_nearest"] / 0.10
    merged["exposure_10pp_post"] = merged["exposure_10pp"] * merged["post"].astype(int)
    merged.to_csv(PROCESSED / "cno2_sepe_epa_rate_proxy_panel.csv", index=False)

    rows = []
    for name, formula, notes in [
        (
            "cno2_rate_proxy_continuous",
            "ln_rate_proxy ~ exposure_10pp_post + C(cno2) + C(quarter)",
            "SEPE parados / (SEPE parados + EPA ocupados). CNO2-quarter proxy, not official unemployment rate.",
        ),
        (
            "cno2_rate_proxy_continuous_trends",
            "ln_rate_proxy ~ exposure_10pp_post + C(cno2) + C(quarter) + C(cno2):trend",
            "Adds CNO2-specific linear trends to the SEPE/EPA denominator proxy model.",
        ),
    ]:
        spec = SimpleSpec(
            name=name,
            formula=formula,
            data=merged,
            coefficient="exposure_10pp_post",
            outcome="ln_rate_proxy",
            cluster="cno2",
            notes=notes,
        )
        row = fit_clustered(spec)
        row["rate_proxy_mean"] = float(merged["rate_proxy"].mean())
        row["rate_proxy_note"] = "Denominator uses INE EPA table 65134 occupied persons; numerator uses SEPE registered unemployed aggregated to CNO2-quarter."
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "table_next_step_cno2_rate_proxy.csv", index=False)
    return out


def run_timing_and_continuous_extensions() -> pd.DataFrame:
    panel = load_main_panel()
    rows: list[dict] = []
    for event_month in ["2022-09", "2022-11", "2022-12", "2023-01"]:
        data = add_binary_design(panel, event_month=event_month)
        for outcome in ["ln_parados", "ln_contratos_p1"]:
            rows.append(
                fit_clustered(
                    SimpleSpec(
                        name=f"binary_{event_month}_{outcome}",
                        formula=f"{outcome} ~ did_post + C(cno4) + C(period)",
                        data=data,
                        coefficient="did_post",
                        outcome=outcome,
                        cluster="cno4",
                        notes=f"Post starts after {event_month}",
                    )
                )
            )

    cont = panel.dropna(subset=["ln_parados"]).copy()
    cont["post_sep"] = cont["month_id"] > EVENT_MONTH_ID
    cont["exposure_10pp"] = cont[EXPOSURE] / 0.10
    cont["exposure_10pp_post"] = cont["exposure_10pp"] * cont["post_sep"].astype(int)
    cont["trend"] = cont["month_id"] - cont["month_id"].min()
    for outcome in ["ln_parados", "ln_contratos_p1"]:
        rows.append(
            fit_clustered(
                SimpleSpec(
                    name=f"continuous_sep_{outcome}",
                    formula=f"{outcome} ~ exposure_10pp_post + C(cno4) + C(period)",
                    data=cont,
                    coefficient="exposure_10pp_post",
                    outcome=outcome,
                    cluster="cno4",
                    notes="Continuous exposure per 10pp; all CNO4 occupations",
                )
            )
        )
        rows.append(
            fit_clustered(
                SimpleSpec(
                    name=f"continuous_sep_trends_{outcome}",
                    formula=f"{outcome} ~ exposure_10pp_post + C(cno4) + C(period) + C(cno4):trend",
                    data=cont,
                    coefficient="exposure_10pp_post",
                    outcome=outcome,
                    cluster="cno4",
                    notes="Continuous exposure per 10pp with CNO4-specific linear trends",
                )
            )
        )

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "table_next_step_timing_continuous_contracts.csv", index=False)
    return out


def download_match_diagnostics() -> None:
    files = {
        MATCH_NEAREST: "https://raw.githubusercontent.com/dgonzalezgonzalez/Prestaciones-por-empleo-AI-exposure/main/data/processed/spanish_occupation_matches_cosine_nearest.csv",
        MATCH_WEIGHTED: "https://raw.githubusercontent.com/dgonzalezgonzalez/Prestaciones-por-empleo-AI-exposure/main/data/processed/spanish_occupation_matches_cosine_weighted.csv",
    }
    for path, url in files.items():
        if path.exists() and path.stat().st_size > 1000:
            continue
        urllib.request.urlretrieve(url, path)


def build_validation_queue() -> pd.DataFrame:
    download_match_diagnostics()
    occ = pd.read_csv(PROCESSED / "occupation_exposure_table.csv", dtype={"cno4": "string"})
    occ["cno2"] = occ["cno4"].str.slice(0, 2)
    occ["rank_exposure"] = occ["exposure_nearest"].rank(method="first", ascending=False).astype(int)
    occ["validation_priority"] = np.select(
        [
            occ["rank_exposure"].le(25),
            occ["mean_parados"].rank(method="first", ascending=False).le(25) & occ["exposure_group"].eq("high"),
            occ["exposure_group"].eq("high"),
        ],
        ["top_exposure", "large_high_exposure", "high_exposure"],
        default="lower_priority",
    )
    queue = occ.loc[occ["validation_priority"].ne("lower_priority")].copy()
    queue["manual_check"] = ""
    queue["notes_for_reviewer"] = np.where(
        queue["rank_exposure"].le(25),
        "Check whether CNO4 title is semantically close to nearest Anthropic/O*NET task exposure.",
        "Check large-cell influence and whether high exposure is plausible.",
    )

    nearest = pd.read_csv(
        MATCH_NEAREST,
        dtype={"CNO4": "string", "spanish_code": "string", "anthropic_occ_code": "string"},
    )
    nearest["CNO4"] = nearest["CNO4"].str.zfill(4)
    nearest = nearest.rename(
        columns={
            "CNO4": "cno4",
            "spanish_title": "match_spanish_title",
            "anthropic_occ_code": "nearest_anthropic_occ_code",
            "anthropic_title": "nearest_anthropic_title",
            "anthropic_observed_exposure": "nearest_anthropic_observed_exposure",
            "cosine_similarity": "nearest_cosine_similarity",
            "spanish_embedding_text": "spanish_embedding_text",
        }
    )
    nearest["spanish_embedding_text_short"] = nearest["spanish_embedding_text"].fillna("").str.slice(0, 600)
    nearest_cols = [
        "cno4",
        "match_spanish_title",
        "nearest_anthropic_occ_code",
        "nearest_anthropic_title",
        "nearest_anthropic_observed_exposure",
        "nearest_cosine_similarity",
        "spanish_embedding_text_short",
    ]
    queue = queue.merge(nearest[nearest_cols], on="cno4", how="left")

    weighted = pd.read_csv(
        MATCH_WEIGHTED,
        dtype={"CNO4": "string", "spanish_code": "string", "anthropic_occ_code": "string"},
    )
    weighted["CNO4"] = weighted["CNO4"].str.zfill(4)
    weighted_summary = (
        weighted.groupby("CNO4", as_index=False)
        .agg(
            weighted_match_rows=("anthropic_occ_code", "count"),
            weighted_unique_anthropic_titles=("anthropic_title", "nunique"),
            weighted_mean_cosine=("cosine_similarity", "mean"),
            weighted_max_cosine=("cosine_similarity", "max"),
            weighted_mean_anthropic_exposure=("anthropic_observed_exposure", "mean"),
        )
        .rename(columns={"CNO4": "cno4"})
    )
    queue = queue.merge(weighted_summary, on="cno4", how="left")
    queue["match_review_flag"] = np.select(
        [
            queue["nearest_cosine_similarity"].lt(0.55),
            queue["nearest_anthropic_title"].isna(),
            queue["nearest_anthropic_observed_exposure"].sub(queue["exposure_nearest"]).abs().gt(1e-9),
        ],
        ["low_similarity", "missing_match", "exposure_mismatch"],
        default="ok",
    )
    priority_order = {"top_exposure": 0, "large_high_exposure": 1, "high_exposure": 2}
    queue["validation_priority_order"] = queue["validation_priority"].map(priority_order).fillna(99).astype(int)
    queue = queue.sort_values(["validation_priority_order", "rank_exposure", "mean_parados"], ascending=[True, True, False])
    queue.to_csv(TABLES / "table_next_step_exposure_validation_queue.csv", index=False)
    queue.to_csv(TABLES / "table_v1_exposure_match_validation_queue.csv", index=False)
    return queue


def write_next_step_summary(
    subgroup: pd.DataFrame,
    province: pd.DataFrame,
    rate_proxy: pd.DataFrame,
    timing: pd.DataFrame,
    validation: pd.DataFrame,
    cno2_month_fe: pd.DataFrame,
    cno2_month_event: pd.DataFrame,
    cno2_contribution: pd.DataFrame,
    within_cno2: pd.DataFrame,
    age3039_cno2_month_fe: pd.DataFrame,
    age3039_cno2_month_event: pd.DataFrame,
    honest_smooth: pd.DataFrame,
    honest_rm: pd.DataFrame,
) -> None:
    summary = {
        "subgroup_rows": int(len(subgroup)),
        "province_specs": int(len(province)),
        "rate_proxy_specs": int(len(rate_proxy)),
        "timing_specs": int(len(timing)),
        "validation_queue_rows": int(len(validation)),
        "cno2_month_fe_specs": int(len(cno2_month_fe)),
        "cno2_month_fe_event_rows": int(len(cno2_month_event)),
        "cno2_month_fe_pretrend_p_value": float(cno2_month_event["pretrend_p_value"].dropna().iloc[0]),
        "cno2_decomposition_rows": int(len(cno2_contribution)),
        "within_cno2_family_rows": int(len(within_cno2)),
        "age3039_cno2_month_fe_specs": int(len(age3039_cno2_month_fe)),
        "age3039_cno2_month_fe_pretrend_p_value": float(age3039_cno2_month_event["pretrend_p_value"].dropna().iloc[0]),
        "honestdid_smoothness_rows": int(len(honest_smooth)),
        "honestdid_relative_magnitude_rows": int(len(honest_rm)),
        "outputs": {
            "subgroup": "output/tables/table_next_step_subgroup_heterogeneity.csv",
            "province": "output/tables/table_next_step_province_fe.csv",
            "rate_proxy": "output/tables/table_next_step_cno2_rate_proxy.csv",
            "timing_continuous_contracts": "output/tables/table_next_step_timing_continuous_contracts.csv",
            "validation_queue": "output/tables/table_next_step_exposure_validation_queue.csv",
            "validation_queue_v1": "output/tables/table_v1_exposure_match_validation_queue.csv",
            "cno2_month_fe": "output/tables/table_v1_cno2_month_fe_specifications.csv",
            "cno2_month_fe_event_study": "output/tables/table_v1_cno2_month_fe_event_study.csv",
            "cno2_fwl_decomposition": "output/tables/table_recommended_cno2_fwl_decomposition.csv",
            "within_cno2_family_did": "output/tables/table_recommended_within_cno2_family_did.csv",
            "age3039_cno2_month_fe": "output/tables/table_recommended_age3039_cno2_month_fe.csv",
            "age3039_cno2_month_fe_event_study": "output/tables/table_recommended_age3039_cno2_month_fe_event_study.csv",
            "honestdid_smoothness": "output/tables/table_recommended_honestdid_smoothness.csv",
            "honestdid_relative_magnitude": "output/tables/table_recommended_honestdid_relative_magnitude.csv",
        },
    }
    (PROCESSED / "next_steps_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def make_next_step_figures(
    subgroup: pd.DataFrame,
    province: pd.DataFrame,
    rate_proxy: pd.DataFrame,
    timing: pd.DataFrame,
    cno2_month_fe: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )

    label_map = {
        "gender_hombre_parados": "Men",
        "gender_mujer_parados": "Women",
        "age_<18_parados": "<18",
        "age_18-24_parados": "18-24",
        "age_25-29_parados": "25-29",
        "age_under_30_parados": "Under 30",
        "age_30-39_parados": "30-39",
        "age_40-44_parados": "40-44",
        "age_>44_parados": ">44",
    }
    plot = subgroup.loc[subgroup["spec"].isin(label_map)].copy()
    plot["label"] = plot["spec"].map(label_map)
    plot["ci_low"] = plot["beta_log_points"].astype(float) - 1.96 * plot["se"].astype(float)
    plot["ci_high"] = plot["beta_log_points"].astype(float) + 1.96 * plot["se"].astype(float)
    order = ["Men", "Women", "<18", "18-24", "25-29", "Under 30", "30-39", "40-44", ">44"]
    plot["order"] = plot["label"].map({label: i for i, label in enumerate(order)})
    plot = plot.sort_values("order")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    y = np.arange(len(plot))
    beta = plot["beta_log_points"].astype(float).to_numpy()
    low = plot["ci_low"].to_numpy()
    high = plot["ci_high"].to_numpy()
    ax.axvline(0, color="#777777", lw=0.9)
    ax.errorbar(beta, y, xerr=[beta - low, high - beta], fmt="o", color="#8f2445", ecolor="#bd8da0", capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Log-point DiD estimate on parados")
    ax.set_title("Subgroup heterogeneity: high vs zero AI exposure")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_05_next_step_subgroup_parados.png", bbox_inches="tight")
    plt.close(fig)

    rows = []
    for _, r in province.iterrows():
        rows.append(
            {
                "label": "Province FE: parados" if r["outcome"] == "ln_parados_p1" else "Province FE: contratos",
                "beta": float(r["beta_log_points"]),
                "se": float(r["se"]),
            }
        )
    for _, r in rate_proxy.iterrows():
        rows.append(
            {
                "label": "CNO2 rate proxy" if "trends" not in r["spec"] else "CNO2 rate proxy + trends",
                "beta": float(r["beta_log_points"]),
                "se": float(r["se"]),
            }
        )
    keep = timing.loc[timing["spec"].isin(["continuous_sep_ln_parados", "continuous_sep_trends_ln_parados"])].copy()
    for _, r in keep.iterrows():
        rows.append(
            {
                "label": "Continuous exposure" if "trends" not in r["spec"] else "Continuous exposure + trends",
                "beta": float(r["beta_log_points"]),
                "se": float(r["se"]),
            }
        )
    for _, r in cno2_month_fe.loc[
        cno2_month_fe["spec"].isin(
            [
                "cno2_month_fe_binary_ln_parados",
                "cno2_month_fe_continuous_ln_parados",
                "cno2_month_fe_binary_ln_contratos_p1",
            ]
        )
    ].iterrows():
        label = {
            "cno2_month_fe_binary_ln_parados": "CNO2-month FE: parados",
            "cno2_month_fe_continuous_ln_parados": "CNO2-month FE: continuous",
            "cno2_month_fe_binary_ln_contratos_p1": "CNO2-month FE: contratos",
        }[r["spec"]]
        rows.append({"label": label, "beta": float(r["beta_log_points"]), "se": float(r["se"])})
    robust = pd.DataFrame(rows)
    robust["ci_low"] = robust["beta"] - 1.96 * robust["se"]
    robust["ci_high"] = robust["beta"] + 1.96 * robust["se"]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    y = np.arange(len(robust))
    ax.axvline(0, color="#777777", lw=0.9)
    ax.errorbar(
        robust["beta"],
        y,
        xerr=[robust["beta"] - robust["ci_low"], robust["ci_high"] - robust["beta"]],
        fmt="o",
        color="#325d79",
        ecolor="#8aa7ba",
        capsize=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(robust["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Log-point estimate")
    ax.set_title("Next-step robustness diagnostics")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_06_next_step_robustness_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    subgroup = run_subgroup_models()
    province = run_province_models()
    rate_proxy = run_cno2_denominator_proxy()
    timing = run_timing_and_continuous_extensions()
    cno2_month_fe, cno2_month_event = run_cno2_month_fe_models()
    cno2_contribution, cno2_family_summary, within_cno2 = run_recommended_family_steps()
    age3039_cno2_month_fe, age3039_cno2_month_event = run_age3039_cno2_month_fe()
    honest_original, honest_smooth, honest_rm = run_honestdid_original_event_study()
    validation = build_validation_queue()
    make_next_step_figures(subgroup, province, rate_proxy, timing, cno2_month_fe)
    write_next_step_summary(
        subgroup,
        province,
        rate_proxy,
        timing,
        validation,
        cno2_month_fe,
        cno2_month_event,
        cno2_contribution,
        within_cno2,
        age3039_cno2_month_fe,
        age3039_cno2_month_event,
        honest_smooth,
        honest_rm,
    )
    print("Next-step outputs written.")
    print(json.dumps(json.loads((PROCESSED / "next_steps_metadata.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
