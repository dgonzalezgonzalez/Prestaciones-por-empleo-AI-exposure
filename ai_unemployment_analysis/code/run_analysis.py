from __future__ import annotations

import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "sepe_cno4_monthly_ai_exposure.csv"
PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "output" / "figures"
TABLES = PROJECT_ROOT / "output" / "tables"
MODELS = PROJECT_ROOT / "output" / "models"

MAIN_EXPOSURE = "observed_exposure_cosine_nearest"
EVENT_MONTH = "2022-09"
RNG_SEED = 20260603


@dataclass
class ModelSpec:
    name: str
    estimand: str
    formula: str
    data: pd.DataFrame
    coefficient: str
    outcome: str = "ln_parados"
    weight_col: str | None = None
    notes: str = ""


def ensure_dirs() -> None:
    for path in [PROCESSED, FIGURES, TABLES, MODELS]:
        path.mkdir(parents=True, exist_ok=True)


def month_id(period: pd.Series) -> pd.Series:
    year = period.str.slice(0, 4).astype(int)
    month = period.str.slice(5, 7).astype(int)
    return year * 12 + month


def event_month_id(event_month: str) -> int:
    year, month = event_month.split("-")
    return int(year) * 12 + int(month)


def read_total_panel() -> pd.DataFrame:
    usecols = [
        "period",
        "cno4",
        "occupation_title",
        "dimension",
        "category",
        "gender",
        "contratos",
        "parados",
        "personas",
        "exposure_occupation_title",
        "observed_exposure_rf",
        "observed_exposure_cosine_weighted",
        "observed_exposure_cosine_nearest",
    ]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        RAW_CSV,
        usecols=usecols,
        dtype={"cno4": "string"},
        chunksize=500_000,
    ):
        chunk["cno4"] = chunk["cno4"].str.zfill(4)
        keep = (
            chunk["cno4"].str.len().eq(4)
            & chunk["dimension"].eq("total")
            & chunk["category"].eq("Total")
            & chunk["gender"].eq("Total")
        )
        chunks.append(chunk.loc[keep].copy())

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop(columns=["dimension", "category", "gender"])
    df["month_id"] = month_id(df["period"])
    df["period_date"] = pd.to_datetime(df["period"] + "-01")
    df["parados"] = pd.to_numeric(df["parados"], errors="coerce")
    df["contratos"] = pd.to_numeric(df["contratos"], errors="coerce")
    df["ln_parados"] = np.nan
    positive_parados = df["parados"] > 0
    df.loc[positive_parados, "ln_parados"] = np.log(df.loc[positive_parados, "parados"])
    df["ln_parados_p1"] = np.log(df["parados"].fillna(0) + 1)
    df["ln_contratos_p1"] = np.log(df["contratos"].fillna(0) + 1)
    return df


def add_group(panel: pd.DataFrame, measure: str, prefix: str, q: float = 0.75) -> tuple[pd.DataFrame, float]:
    out = panel.copy()
    # Match the preliminary Stata do-file: the threshold is computed on the
    # filtered occupation-month panel, not only on unique occupations.
    threshold = float(out[measure].quantile(q))
    out[f"{prefix}_high"] = out[measure] > threshold
    out[f"{prefix}_zero"] = out[measure] == 0
    out[f"{prefix}_group"] = "middle"
    out.loc[out[f"{prefix}_zero"], f"{prefix}_group"] = "zero"
    out.loc[out[f"{prefix}_high"], f"{prefix}_group"] = "high"
    return out, threshold


def make_analysis_panel() -> tuple[pd.DataFrame, dict]:
    panel = read_total_panel()
    thresholds: dict[str, float] = {}
    exposure_specs = {
        "nearest": "observed_exposure_cosine_nearest",
        "weighted": "observed_exposure_cosine_weighted",
        "rf": "observed_exposure_rf",
    }
    for prefix, measure in exposure_specs.items():
        panel, threshold = add_group(panel, measure, prefix)
        thresholds[prefix] = threshold

    pre_event = event_month_id(EVENT_MONTH)
    panel["post_sep2022"] = panel["month_id"] > pre_event
    panel["event_time_sep2022"] = panel["month_id"] - pre_event
    panel["exposure_nearest_10pp"] = panel[MAIN_EXPOSURE] / 0.10
    panel["exposure_nearest_10pp_post"] = panel["exposure_nearest_10pp"] * panel["post_sep2022"].astype(int)

    weights = (
        panel.loc[panel["month_id"] <= pre_event]
        .groupby("cno4", as_index=True)["parados"]
        .mean()
        .rename("pre_mean_parados")
    )
    panel = panel.merge(weights, left_on="cno4", right_index=True, how="left")
    panel["pre_mean_parados"] = panel["pre_mean_parados"].fillna(panel["pre_mean_parados"].median()).clip(lower=1)

    panel.to_csv(PROCESSED / "analysis_panel.csv", index=False)
    metadata = {
        "raw_csv": str(RAW_CSV),
        "rows": int(len(panel)),
        "occupations": int(panel["cno4"].nunique()),
        "periods": int(panel["period"].nunique()),
        "period_min": str(panel["period"].min()),
        "period_max": str(panel["period"].max()),
        "thresholds": thresholds,
        "event_month": EVENT_MONTH,
        "post_definition": "period month_id > event month_id; main post starts 2022-10",
        "python": platform.python_version(),
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (PROCESSED / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return panel, metadata


def occupation_table(panel: pd.DataFrame) -> pd.DataFrame:
    occ = (
        panel.groupby(["cno4", "occupation_title"], as_index=False)
        .agg(
            exposure_nearest=(MAIN_EXPOSURE, "first"),
            exposure_weighted=("observed_exposure_cosine_weighted", "first"),
            exposure_rf=("observed_exposure_rf", "first"),
            exposure_group=("nearest_group", "first"),
            mean_parados=("parados", "mean"),
            mean_contratos=("contratos", "mean"),
            pre_mean_parados=("pre_mean_parados", "first"),
        )
        .sort_values(["exposure_nearest", "mean_parados"], ascending=[False, False])
    )
    occ.to_csv(PROCESSED / "occupation_exposure_table.csv", index=False)
    occ.head(50).to_csv(TABLES / "table_top50_exposed_occupations.csv", index=False)
    occ.loc[occ["exposure_group"].eq("zero")].head(50).to_csv(TABLES / "table_first50_zero_exposure_occupations.csv", index=False)

    top_latex = occ.head(25).copy()
    top_latex["exposure_nearest"] = top_latex["exposure_nearest"].round(4)
    top_latex["mean_parados"] = top_latex["mean_parados"].round(0).astype("Int64")
    top_latex[["cno4", "occupation_title", "exposure_nearest", "mean_parados"]].to_latex(
        TABLES / "table_top25_exposed_occupations.tex",
        index=False,
        escape=True,
    )
    return occ


def descriptive_tables(panel: pd.DataFrame, occ: pd.DataFrame) -> None:
    group_counts = (
        occ.groupby("exposure_group", as_index=False)
        .agg(
            occupations=("cno4", "count"),
            mean_exposure=("exposure_nearest", "mean"),
            median_exposure=("exposure_nearest", "median"),
            mean_pre_parados=("pre_mean_parados", "mean"),
            mean_parados=("mean_parados", "mean"),
            total_mean_parados=("mean_parados", "sum"),
        )
        .sort_values("exposure_group")
    )
    group_counts.to_csv(TABLES / "table_exposure_group_summary.csv", index=False)
    group_counts.to_latex(TABLES / "table_exposure_group_summary.tex", index=False, float_format="%.3f")

    by_month = (
        panel.groupby(["period", "period_date", "nearest_group"], as_index=False)
        .agg(
            occupations=("cno4", "nunique"),
            total_parados=("parados", "sum"),
            mean_parados=("parados", "mean"),
            mean_ln_parados=("ln_parados", "mean"),
            total_contratos=("contratos", "sum"),
        )
        .sort_values(["period", "nearest_group"])
    )
    by_month["ln_total_parados"] = np.log(by_month["total_parados"])
    base = (
        by_month.loc[by_month["period"].between("2022-01", "2022-08")]
        .groupby("nearest_group")["ln_total_parados"]
        .mean()
        .rename("base_ln_total_parados")
    )
    by_month = by_month.merge(base, on="nearest_group", how="left")
    by_month["ln_total_parados_index"] = by_month["ln_total_parados"] - by_month["base_ln_total_parados"]
    by_month.to_csv(TABLES / "table_monthly_descriptives_by_exposure_group.csv", index=False)


def make_figures(panel: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    colors = {"zero": "#325d79", "middle": "#8c7a3f", "high": "#8f2445"}
    labels = {"zero": "Zero exposure", "middle": "Middle exposure", "high": "High exposure"}
    monthly = (
        panel.groupby(["period_date", "nearest_group"], as_index=False)
        .agg(
            mean_ln_parados=("ln_parados", "mean"),
            total_parados=("parados", "sum"),
        )
        .sort_values("period_date")
    )
    monthly["ln_total_parados"] = np.log(monthly["total_parados"])
    base = (
        monthly.loc[(monthly["period_date"] >= "2022-01-01") & (monthly["period_date"] <= "2022-08-01")]
        .groupby("nearest_group")["ln_total_parados"]
        .mean()
    )
    monthly["ln_total_index"] = monthly.apply(
        lambda r: r["ln_total_parados"] - base.loc[r["nearest_group"]], axis=1
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for group in ["zero", "middle", "high"]:
        sub = monthly.loc[monthly["nearest_group"].eq(group)]
        ax.plot(sub["period_date"], sub["mean_ln_parados"], lw=2.0, color=colors[group], label=labels[group])
    ax.axvline(pd.to_datetime(EVENT_MONTH + "-01"), color="#444444", ls="--", lw=1)
    ax.set_title("Mean log unemployed by AI exposure group")
    ax.set_xlabel("")
    ax.set_ylabel("Mean log(parados)")
    ax.legend(frameon=False, ncols=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_01_mean_log_parados_by_exposure_group.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for group in ["zero", "middle", "high"]:
        sub = monthly.loc[monthly["nearest_group"].eq(group)]
        ax.plot(sub["period_date"], sub["ln_total_index"], lw=2.0, color=colors[group], label=labels[group])
    ax.axhline(0, color="#777777", lw=0.8)
    ax.axvline(pd.to_datetime(EVENT_MONTH + "-01"), color="#444444", ls="--", lw=1)
    ax.set_title("Log total unemployed, indexed to Jan-Aug 2022")
    ax.set_xlabel("")
    ax.set_ylabel("Log points relative to pre-period mean")
    ax.legend(frameon=False, ncols=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_02_indexed_total_log_parados_by_exposure_group.png", bbox_inches="tight")
    plt.close(fig)

    occ = panel.drop_duplicates("cno4").copy()
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    bins = np.linspace(0, occ[MAIN_EXPOSURE].max(), 35)
    ax.hist(occ[MAIN_EXPOSURE], bins=bins, color="#5d7186", edgecolor="white")
    ax.axvline(panel[MAIN_EXPOSURE].quantile(0.75), color="#8f2445", ls="--", lw=1.3, label="Panel p75")
    ax.set_title("Distribution of CNO4 AI observed exposure")
    ax.set_xlabel("Observed exposure, cosine nearest")
    ax.set_ylabel("Occupations")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_03_exposure_distribution.png", bbox_inches="tight")
    plt.close(fig)


def sample_binary(
    panel: pd.DataFrame,
    measure_prefix: str = "nearest",
    event_month: str = EVENT_MONTH,
    q: float = 0.75,
    control: str = "zero",
) -> pd.DataFrame:
    event_id = event_month_id(event_month)
    measure_map = {
        "nearest": MAIN_EXPOSURE,
        "weighted": "observed_exposure_cosine_weighted",
        "rf": "observed_exposure_rf",
    }
    measure = measure_map[measure_prefix]
    threshold = float(panel[measure].quantile(q))
    out = panel.copy()
    out["treat"] = out[measure] > threshold
    if control == "zero":
        out["control"] = out[measure] == 0
        control_label = "zero exposure"
    elif control == "bottom_quartile":
        low_threshold = float(panel[measure].quantile(0.25))
        out["control"] = out[measure] <= low_threshold
        control_label = "bottom quartile"
    else:
        raise ValueError(f"Unknown control: {control}")

    out = out.loc[out["treat"] | out["control"]].copy()
    out["post"] = out["month_id"] > event_id
    out["did_post"] = out["treat"].astype(int) * out["post"].astype(int)
    out["event_time"] = out["month_id"] - event_id
    out["trend"] = out["month_id"] - out["month_id"].min()
    out["sample_label"] = f"{measure_prefix}: high>{q:.2f} vs {control_label}; event={event_month}"
    return out


def fit_clustered(spec: ModelSpec):
    data = spec.data.dropna(subset=[spec.outcome, spec.coefficient, "cno4", "period"]).copy()
    if spec.weight_col:
        model = smf.wls(spec.formula, data=data, weights=data[spec.weight_col])
    else:
        model = smf.ols(spec.formula, data=data)
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": data["cno4"], "use_correction": True},
    )
    return result, data


def model_row(spec: ModelSpec, result, data: pd.DataFrame) -> dict:
    beta = float(result.params[spec.coefficient])
    se = float(result.bse[spec.coefficient])
    pvalue = float(result.pvalues[spec.coefficient])
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se
    return {
        "spec": spec.name,
        "estimand": spec.estimand,
        "coefficient": spec.coefficient,
        "outcome": spec.outcome,
        "beta_log_points": beta,
        "se_cluster_cno4": se,
        "p_value": pvalue,
        "ci95_low_log_points": ci_low,
        "ci95_high_log_points": ci_high,
        "effect_pct": 100 * (math.exp(beta) - 1),
        "ci95_low_pct": 100 * (math.exp(ci_low) - 1),
        "ci95_high_pct": 100 * (math.exp(ci_high) - 1),
        "nobs": int(result.nobs),
        "clusters": int(data["cno4"].nunique()),
        "occupations": int(data["cno4"].nunique()),
        "periods": int(data["period"].nunique()),
        "r2": float(result.rsquared),
        "weights": spec.weight_col or "",
        "notes": spec.notes,
    }


def run_specifications(panel: pd.DataFrame) -> pd.DataFrame:
    main = sample_binary(panel, "nearest", EVENT_MONTH, 0.75, "zero")
    nov = sample_binary(panel, "nearest", "2022-11", 0.75, "zero")
    top90 = sample_binary(panel, "nearest", EVENT_MONTH, 0.90, "zero")
    bottomq = sample_binary(panel, "nearest", EVENT_MONTH, 0.75, "bottom_quartile")
    weighted = sample_binary(panel, "weighted", EVENT_MONTH, 0.75, "zero")
    rf = sample_binary(panel, "rf", EVENT_MONTH, 0.75, "bottom_quartile")
    sample_2022_onward = sample_binary(panel.loc[panel["period"].ge("2022-01")], "nearest", EVENT_MONTH, 0.75, "zero")
    placebo_pre = sample_binary(panel.loc[panel["period"].le(EVENT_MONTH)], "nearest", "2022-01", 0.75, "zero")
    continuous = panel.dropna(subset=["ln_parados"]).copy()
    continuous["trend"] = continuous["month_id"] - continuous["month_id"].min()

    specs = [
        ModelSpec(
            name="1. Main DiD",
            estimand="High exposure (>p75) vs zero exposure, post Sep 2022",
            formula="ln_parados ~ did_post + C(cno4) + C(period)",
            data=main,
            coefficient="did_post",
        ),
        ModelSpec(
            name="2. Main + occupation trends",
            estimand="High exposure (>p75) vs zero exposure, post Sep 2022",
            formula="ln_parados ~ did_post + C(cno4) + C(period) + C(cno4):trend",
            data=main,
            coefficient="did_post",
            notes="Adds CNO4-specific linear trends",
        ),
        ModelSpec(
            name="3. log(parados + 1)",
            estimand="High exposure (>p75) vs zero exposure, post Sep 2022",
            formula="ln_parados_p1 ~ did_post + C(cno4) + C(period)",
            data=main,
            coefficient="did_post",
            outcome="ln_parados_p1",
            notes="Keeps zero-parados rows",
        ),
        ModelSpec(
            name="4. Weighted by pre parados",
            estimand="High exposure (>p75) vs zero exposure, post Sep 2022",
            formula="ln_parados ~ did_post + C(cno4) + C(period)",
            data=main,
            coefficient="did_post",
            weight_col="pre_mean_parados",
            notes="Occupation weights are pre-Sep-2022 mean parados",
        ),
        ModelSpec(
            name="5. Event Nov 2022",
            estimand="High exposure (>p75) vs zero exposure, post Nov 2022",
            formula="ln_parados ~ did_post + C(cno4) + C(period)",
            data=nov,
            coefficient="did_post",
            notes="Post starts Dec 2022",
        ),
        ModelSpec(
            name="6. Top decile",
            estimand="High exposure (>p90) vs zero exposure, post Sep 2022",
            formula="ln_parados ~ did_post + C(cno4) + C(period)",
            data=top90,
            coefficient="did_post",
        ),
        ModelSpec(
            name="7. Continuous exposure",
            estimand="Exposure x post Sep 2022, all occupations, per 10pp exposure",
            formula="ln_parados ~ exposure_nearest_10pp_post + C(cno4) + C(period)",
            data=continuous,
            coefficient="exposure_nearest_10pp_post",
        ),
        ModelSpec(
            name="8. Cosine weighted exposure",
            estimand="High exposure (>p75) vs zero exposure, post Sep 2022",
            formula="ln_parados ~ did_post + C(cno4) + C(period)",
            data=weighted,
            coefficient="did_post",
        ),
        ModelSpec(
            name="9. RF exposure",
            estimand="RF top quartile vs RF bottom quartile, post Sep 2022",
            formula="ln_parados ~ did_post + C(cno4) + C(period)",
            data=rf,
            coefficient="did_post",
            notes="RF exposure has no zero-exposure occupations",
        ),
        ModelSpec(
            name="10. Nearest bottom quartile control",
            estimand="High exposure (>p75) vs bottom exposure quartile, post Sep 2022",
            formula="ln_parados ~ did_post + C(cno4) + C(period)",
            data=bottomq,
            coefficient="did_post",
        ),
        ModelSpec(
            name="11. 2022 onward sample",
            estimand="High exposure (>p75) vs zero exposure, post Sep 2022",
            formula="ln_parados ~ did_post + C(cno4) + C(period)",
            data=sample_2022_onward,
            coefficient="did_post",
            notes="Drops 2021 recovery period",
        ),
        ModelSpec(
            name="12. Pre-period placebo",
            estimand="Fake post Jan 2022, actual post-AI months excluded",
            formula="ln_parados ~ did_post + C(cno4) + C(period)",
            data=placebo_pre,
            coefficient="did_post",
            notes="Uses only months through Sep 2022",
        ),
        ModelSpec(
            name="13. Contracts outcome",
            estimand="High exposure (>p75) vs zero exposure, post Sep 2022",
            formula="ln_contratos_p1 ~ did_post + C(cno4) + C(period)",
            data=main,
            coefficient="did_post",
            outcome="ln_contratos_p1",
            notes="Proxy for hiring/labor demand, not unemployment",
        ),
    ]

    rows = []
    for spec in specs:
        result, data = fit_clustered(spec)
        rows.append(model_row(spec, result, data))
        with open(MODELS / f"{spec.name.split('.')[0].zfill(2)}_summary.txt", "w", encoding="utf-8") as fh:
            fh.write(str(result.summary()))

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "table_regression_specifications.csv", index=False)
    display = out[
        [
            "spec",
            "estimand",
            "beta_log_points",
            "se_cluster_cno4",
            "p_value",
            "effect_pct",
            "nobs",
            "clusters",
            "notes",
        ]
    ].copy()
    for col in ["beta_log_points", "se_cluster_cno4", "p_value", "effect_pct"]:
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    display.to_latex(TABLES / "table_regression_specifications.tex", index=False, escape=True)
    return out


def event_var_name(t: int) -> str:
    if t < 0:
        return f"event_m{abs(t)}"
    if t == 0:
        return "event_0"
    return f"event_p{t}"


def run_event_study(panel: pd.DataFrame) -> pd.DataFrame:
    data = sample_binary(panel, "nearest", EVENT_MONTH, 0.75, "zero")
    data = data.dropna(subset=["ln_parados"]).copy()
    event_times = sorted(int(x) for x in data["event_time"].unique())
    event_terms = []
    for t in event_times:
        if t == -1:
            continue
        name = event_var_name(t)
        data[name] = data["treat"].astype(int) * data["event_time"].eq(t).astype(int)
        event_terms.append(name)

    formula = "ln_parados ~ " + " + ".join(event_terms) + " + C(cno4) + C(period)"
    model = smf.ols(formula, data=data)
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": data["cno4"], "use_correction": True},
    )
    rows = []
    for t in event_times:
        if t == -1:
            rows.append(
                {
                    "event_time": t,
                    "term": "reference",
                    "estimate": 0.0,
                    "se": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "p_value": np.nan,
                    "reference_period": True,
                }
            )
            continue
        name = event_var_name(t)
        beta = float(result.params[name])
        se = float(result.bse[name])
        rows.append(
            {
                "event_time": t,
                "term": name,
                "estimate": beta,
                "se": se,
                "ci95_low": beta - 1.96 * se,
                "ci95_high": beta + 1.96 * se,
                "p_value": float(result.pvalues[name]),
                "reference_period": False,
            }
        )
    es = pd.DataFrame(rows)

    pre_terms = [event_var_name(t) for t in event_times if t <= -2]
    b = result.params.loc[pre_terms].to_numpy()
    cov = result.cov_params().loc[pre_terms, pre_terms].to_numpy()
    wald = float(b.T @ np.linalg.pinv(cov) @ b)
    pretrend_p = float(1 - st.chi2.cdf(wald, len(pre_terms)))
    es["pretrend_wald_chi2"] = wald
    es["pretrend_df"] = len(pre_terms)
    es["pretrend_p_value"] = pretrend_p
    es.to_csv(TABLES / "table_event_study_coefficients.csv", index=False)

    with open(MODELS / "event_study_summary.txt", "w", encoding="utf-8") as fh:
        fh.write(str(result.summary()))
        fh.write(f"\n\nPretrend Wald chi2({len(pre_terms)}): {wald:.6f}, p={pretrend_p:.6g}\n")

    plot_es = es.sort_values("event_time")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.axhline(0, color="#777777", lw=0.9)
    ax.axvline(0, color="#444444", ls="--", lw=1)
    ax.errorbar(
        plot_es["event_time"],
        plot_es["estimate"],
        yerr=[
            plot_es["estimate"] - plot_es["ci95_low"],
            plot_es["ci95_high"] - plot_es["estimate"],
        ],
        fmt="o",
        ms=3.5,
        lw=1.0,
        color="#8f2445",
        ecolor="#b98a9b",
        capsize=2,
    )
    ax.set_title("Event study: high AI exposure vs zero exposure")
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
    fig.savefig(FIGURES / "figure_04_event_study_high_vs_zero_sep2022.png", bbox_inches="tight")
    plt.close(fig)
    return es


def write_blindspot_seed(panel: pd.DataFrame, specs: pd.DataFrame, event_study: pd.DataFrame) -> None:
    main = specs.loc[specs["spec"].eq("1. Main DiD")].iloc[0]
    pre_p = float(event_study["pretrend_p_value"].dropna().iloc[0])
    text = f"""# Blindspot Seed Notes

Output audited: descriptive figures, specification table, and event-study estimates.

Main finding entering the audit:

- Main DiD coefficient: {main['beta_log_points']:.4f} log points ({main['effect_pct']:.2f} percent), clustered by CNO4.
- Event-study pretrend Wald p-value: {pre_p:.4f}.
- Sample: high exposure (> p75) occupations versus zero-exposure occupations, with CNO4 and month fixed effects.

These notes are intentionally short; the full Blindspot report is written after inspecting the output.
"""
    (PROJECT_ROOT / "correspondence" / "blindspot" / "blindspot_seed.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    panel, metadata = make_analysis_panel()
    occ = occupation_table(panel)
    descriptive_tables(panel, occ)
    make_figures(panel)
    specs = run_specifications(panel)
    event_study = run_event_study(panel)
    write_blindspot_seed(panel, specs, event_study)
    print(json.dumps(metadata, indent=2))
    print("Wrote analysis outputs to", PROJECT_ROOT)


if __name__ == "__main__":
    main()
