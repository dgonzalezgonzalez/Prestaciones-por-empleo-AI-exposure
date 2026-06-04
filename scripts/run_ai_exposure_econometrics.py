"""Run SEPE CNO AI-exposure econometric analysis.

Outputs:
- OLS LaTeX table and compiled PDF.
- TWFE event-study coefficient CSVs and plots.
- Clean analysis samples and run metadata.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "sepe_cno4_monthly_ai_exposure.csv"
OUTPUT_DIR = ROOT / "analysis" / "econometrics_outputs"
EVENT_DIR = OUTPUT_DIR / "event_studies"
TABLE_DIR = OUTPUT_DIR / "tables"
GRAFICOS_DIR = OUTPUT_DIR / "Graficos"

INTERVENTION_PERIOD = "2022-09"
OLS_START = "2021-01"
OLS_END = "2026-01"
BASE_EVENT_PERIOD = -1

OUTCOME = "parados"
EXPOSURES = {
    "rf": {
        "column": "observed_exposure_rf",
        "label": "RF",
    },
    "cosine_weighted": {
        "column": "observed_exposure_cosine_weighted",
        "label": "Cosine weighted",
    },
    "cosine_nearest": {
        "column": "observed_exposure_cosine_nearest",
        "label": "Cosine nearest",
    },
}
AIREF_COLORS = {
    "burgundy": "#83082A",
    "text": "#404040",
    "axis_label": "#4D4D4D",
    "grid": "#CCCCCC",
    "ci": "#E397A0",
}
AIREF_FIGSIZE = (14.5 / 2.54, 7.25 / 2.54)


@dataclass
class RegressionResult:
    beta: np.ndarray
    se: np.ndarray
    pvalue: np.ndarray
    r2: float
    nobs: int
    k: int


@dataclass(frozen=True)
class EventOutcome:
    key: str
    source_column: str
    regression_column: str
    label: str
    ylabel: str
    transform: str


EVENT_OUTCOMES = [
    EventOutcome(
        key="unemployment",
        source_column="parados",
        regression_column="event_unemployment",
        label="Registered unemployment",
        ylabel="Effect on log registered unemployment",
        transform="log1p",
    ),
    EventOutcome(
        key="contracts",
        source_column="contratos",
        regression_column="event_contracts",
        label="Contracts",
        ylabel="Effect on contracts",
        transform="levels",
    ),
]


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, EVENT_DIR, TABLE_DIR, GRAFICOS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    for outcome in EVENT_OUTCOMES:
        (EVENT_DIR / outcome.key).mkdir(parents=True, exist_ok=True)
        (GRAFICOS_DIR / outcome.key).mkdir(parents=True, exist_ok=True)


def read_total_panel() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    total = df[
        (df["dimension"] == "total")
        & (df["category"] == "Total")
        & (df["gender"] == "Total")
    ].copy()
    total["period_dt"] = pd.to_datetime(total["period"] + "-01")
    total["cno4"] = total["cno4"].astype(str).str.zfill(4)
    total["log_unemployment"] = np.log1p(total[OUTCOME])
    total["event_unemployment"] = total["log_unemployment"]
    total["event_contracts"] = total["contratos"]
    total["intervention_dt"] = pd.to_datetime(INTERVENTION_PERIOD + "-01")
    total["event_time"] = (
        (total["period_dt"].dt.year - total["intervention_dt"].dt.year) * 12
        + (total["period_dt"].dt.month - total["intervention_dt"].dt.month)
    )
    return total.sort_values(["cno4", "period_dt"]).reset_index(drop=True)


def ols_hc1(y: np.ndarray, x: np.ndarray) -> RegressionResult:
    nobs, k = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    sst = np.sum((y - y.mean()) ** 2)
    ssr = np.sum(resid**2)
    r2 = 1.0 - ssr / sst if sst > 0 else np.nan
    meat = x.T @ ((resid**2)[:, None] * x)
    vcov = xtx_inv @ meat @ xtx_inv
    if nobs > k:
        vcov *= nobs / (nobs - k)
    se = np.sqrt(np.maximum(np.diag(vcov), 0))
    tstat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pvalue = 2 * stats.t.sf(np.abs(tstat), max(nobs - k, 1))
    return RegressionResult(beta=beta, se=se, pvalue=pvalue, r2=r2, nobs=nobs, k=k)


def cluster_ols(y: np.ndarray, x: np.ndarray, clusters: np.ndarray) -> RegressionResult:
    nobs, k = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    sst = np.sum((y - y.mean()) ** 2)
    ssr = np.sum(resid**2)
    r2 = 1.0 - ssr / sst if sst > 0 else np.nan

    meat = np.zeros((k, k))
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        idx = clusters == cluster
        score = x[idx].T @ resid[idx]
        meat += np.outer(score, score)
    vcov = xtx_inv @ meat @ xtx_inv
    g = len(unique_clusters)
    if g > 1 and nobs > k:
        vcov *= (g / (g - 1)) * ((nobs - 1) / (nobs - k))
    se = np.sqrt(np.maximum(np.diag(vcov), 0))
    tstat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pvalue = 2 * stats.t.sf(np.abs(tstat), max(g - 1, 1))
    return RegressionResult(beta=beta, se=se, pvalue=pvalue, r2=r2, nobs=nobs, k=k)


def demean_two_way(matrix: np.ndarray, entity: np.ndarray, time: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(matrix)
    df["_entity"] = entity
    df["_time"] = time
    cols = list(range(matrix.shape[1]))
    entity_mean = df.groupby("_entity")[cols].transform("mean").to_numpy()
    time_mean = df.groupby("_time")[cols].transform("mean").to_numpy()
    grand_mean = matrix.mean(axis=0)
    return matrix - entity_mean - time_mean + grand_mean


def format_number(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def significance_stars(pvalue: float) -> str:
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.10:
        return "*"
    return ""


def make_ols_sample(panel: pd.DataFrame) -> pd.DataFrame:
    base = panel[panel["period"].isin([OLS_START, OLS_END])].copy()
    wide = base.pivot_table(
        index=["cno4", "occupation_title"],
        columns="period",
        values=OUTCOME,
        aggfunc="first",
    ).reset_index()
    exposure = panel.drop_duplicates("cno4")[
        ["cno4"] + [spec["column"] for spec in EXPOSURES.values()]
    ]
    sample = wide.merge(exposure, on="cno4", how="left")
    months = (
        (pd.Period(OLS_END, freq="M") - pd.Period(OLS_START, freq="M")).n
    )
    sample = sample[(sample[OLS_START] > 0) & (sample[OLS_END] > 0)].copy()
    sample["avg_monthly_unemployment_growth_pp"] = (
        (np.log(sample[OLS_END]) - np.log(sample[OLS_START])) / months * 100
    )
    return sample


def run_ols_table(sample: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, RegressionResult]]:
    y = sample["avg_monthly_unemployment_growth_pp"].to_numpy(float)
    rows = []
    results: dict[str, RegressionResult] = {}
    for key, spec in EXPOSURES.items():
        x = np.column_stack([np.ones(len(sample)), sample[spec["column"]].to_numpy(float)])
        result = ols_hc1(y, x)
        results[key] = result
        rows.append(
            {
                "measure": spec["label"],
                "coefficient": result.beta[1],
                "std_error": result.se[1],
                "p_value": result.pvalue[1],
                "r2": result.r2,
                "nobs": result.nobs,
                "depvar_mean": y.mean(),
            }
        )
    table_df = pd.DataFrame(rows)
    table_df.to_csv(TABLE_DIR / "ols_growth_regressions.csv", index=False)
    return table_df, results


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def write_ols_latex(results: dict[str, RegressionResult], sample: pd.DataFrame) -> Path:
    tex_path = TABLE_DIR / "ols_growth_regressions.tex"
    cols = list(EXPOSURES.keys())
    exposure_cells = []
    se_cells = []
    for key in cols:
        res = results[key]
        exposure_cells.append(format_number(res.beta[1]) + significance_stars(res.pvalue[1]))
        se_cells.append("(" + format_number(res.se[1]) + ")")
    r2_cells = [format_number(results[key].r2) for key in cols]
    n_cells = [str(results[key].nobs) for key in cols]
    mean_cells = [format_number(sample["avg_monthly_unemployment_growth_pp"].mean())] * 3
    labels = [EXPOSURES[key]["label"] for key in cols]

    lines = [
        r"\begin{table}[!htbp]\centering",
        r"\caption{AI exposure and unemployment growth by occupation}",
        r"\label{tab:ai_exposure_unemployment_growth}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        " & " + " & ".join(f"({i})" for i in range(1, 4)) + r" \\",
        " & " + " & ".join(latex_escape(label) for label in labels) + r" \\",
        r"\midrule",
        "AI exposure & " + " & ".join(exposure_cells) + r" \\",
        " & " + " & ".join(se_cells) + r" \\",
        r"\midrule",
        "Observations & " + " & ".join(n_cells) + r" \\",
        "$R^2$ & " + " & ".join(r2_cells) + r" \\",
        "Mean dep. var. & " + " & ".join(mean_cells) + r" \\",
        "CNO unit & Yes & Yes & Yes" + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.25em}",
        r"\begin{minipage}{0.96\linewidth}",
        r"Notes: "
        + (
            r"Unit of observation is CNO occupation. "
            r"Dependent variable is average monthly log growth in registered unemployed "
            rf"from {OLS_START} to {OLS_END}, in percentage points. "
            r"Only aggregate rows with dimension=total, "
            r"category=Total, and gender=Total are used. "
            r"Each column reports a separate OLS regression with HC1 robust standard errors "
            r"(one observation per CNO, so CNO clustering is not applicable for this cross-section) "
            r"in parentheses. *** $p<0.01$, ** $p<0.05$, * $p<0.10$."
        ),
        r"\end{minipage}",
        r"\end{table}",
    ]
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tex_path


def write_pdf_wrapper(table_tex: Path) -> Path:
    wrapper = TABLE_DIR / "ols_growth_regressions_document.tex"
    wrapper.write_text(
        "\n".join(
            [
                r"\documentclass[11pt]{article}",
                r"\usepackage[margin=1in]{geometry}",
                r"\usepackage{booktabs}",
                r"\begin{document}",
                rf"\input{{{table_tex.name}}}",
                r"\end{document}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return wrapper


def compile_latex(wrapper: Path) -> dict[str, str | bool]:
    compile_script = (
        Path.home()
        / ".codex"
        / "plugins"
        / "cache"
        / "openai-bundled"
        / "latex"
        / "0.2.2"
        / "scripts"
        / "compile_latex.py"
    )
    candidates = []
    if compile_script.exists():
        candidates.append([sys.executable, str(compile_script), str(wrapper), "--compiler", "tectonic"])
    tectonic = shutil.which("tectonic")
    if tectonic:
        candidates.append([tectonic, str(wrapper)])
    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        candidates.append([pdflatex, "-interaction=nonstopmode", wrapper.name])

    for cmd in candidates:
        try:
            proc = subprocess.run(
                cmd,
                cwd=wrapper.parent,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            continue
        if proc.returncode == 0 and wrapper.with_suffix(".pdf").exists():
            return {
                "compiled": True,
                "compiler_command": " ".join(cmd),
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
            }
    return {
        "compiled": False,
        "compiler_command": "",
        "stdout_tail": "",
        "stderr_tail": "No usable LaTeX compiler found or compilation failed.",
    }


def event_study_design(
    data: pd.DataFrame,
    outcome: EventOutcome,
    exposure_col: str | None,
    treatment_col: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    needed = [outcome.regression_column, "cno4", "period", "event_time"]
    if exposure_col is not None:
        needed.append(exposure_col)
    if treatment_col is not None:
        needed.append(treatment_col)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).copy()
    event_times = sorted(t for t in data["event_time"].unique() if t != BASE_EVENT_PERIOD)
    if exposure_col is not None:
        treatment_values = data[exposure_col].to_numpy(float)
    elif treatment_col is not None:
        treatment_values = data[treatment_col].to_numpy(float)
    else:
        raise ValueError("Need exposure_col or treatment_col")
    x_cols = []
    for event_time in event_times:
        x_cols.append(treatment_values * (data["event_time"].to_numpy() == event_time))
    event_x = np.column_stack(x_cols)
    keep = np.where(np.nanstd(event_x, axis=0) > 0)[0]
    event_x = event_x[:, keep]
    event_times = [event_times[i] for i in keep]

    entity_dummies = pd.get_dummies(data["cno4"], drop_first=True, dtype=float).to_numpy()
    time_dummies = pd.get_dummies(data["period"], drop_first=True, dtype=float).to_numpy()
    intercept = np.ones((len(data), 1))
    x = np.column_stack([event_x, entity_dummies, time_dummies, intercept])
    y = data[outcome.regression_column].to_numpy(float)
    entity = pd.Categorical(data["cno4"]).codes

    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError(f"Non-finite values remain in event-study design for {exposure_col or treatment_col}")
    return y, x, entity, event_times


def run_event_study(
    panel: pd.DataFrame,
    outcome: EventOutcome,
    spec_name: str,
    exposure_key: str,
    mode: str,
) -> pd.DataFrame:
    exposure_col = EXPOSURES[exposure_key]["column"]
    data = panel.copy()
    treatment_col = None
    x_exposure_col: str | None = exposure_col

    if mode == "continuous":
        sample_note = "continuous exposure"
    else:
        cno_exposure = data.drop_duplicates("cno4")[["cno4", exposure_col]].copy()
        q25 = cno_exposure[exposure_col].quantile(0.25)
        q75 = cno_exposure[exposure_col].quantile(0.75)
        if mode == "top_vs_zero":
            eligible = cno_exposure[
                (cno_exposure[exposure_col] >= q75) | (cno_exposure[exposure_col] == 0)
            ].copy()
            sample_note = f"top quartile vs zero exposure; q75={q75:.6f}"
        elif mode == "top_vs_bottom":
            eligible = cno_exposure[
                (cno_exposure[exposure_col] >= q75) | (cno_exposure[exposure_col] <= q25)
            ].copy()
            sample_note = f"top quartile vs bottom quartile; q25={q25:.6f}; q75={q75:.6f}"
        else:
            raise ValueError(f"Unknown mode: {mode}")
        eligible["treated"] = (eligible[exposure_col] >= q75).astype(float)
        data = data.merge(eligible[["cno4", "treated"]], on="cno4", how="inner")
        treatment_col = "treated"
        x_exposure_col = None

    y, x, entity, kept_event_times = event_study_design(data, outcome, x_exposure_col, treatment_col)
    result = cluster_ols(y, x, entity)
    n_event_coefficients = len(kept_event_times)
    if np.isnan(result.beta[:n_event_coefficients]).any():
        raise ValueError(f"NaN event coefficients in {spec_name}")
    rows = []
    for i, event_time in enumerate(kept_event_times):
        rows.append(
            {
                "spec": spec_name,
                "outcome": outcome.key,
                "outcome_label": outcome.label,
                "exposure_measure": EXPOSURES[exposure_key]["label"],
                "mode": mode,
                "event_time": event_time,
                "coefficient": result.beta[i],
                "std_error": result.se[i],
                "p_value": result.pvalue[i],
                "ci_low": result.beta[i] - 1.96 * result.se[i],
                "ci_high": result.beta[i] + 1.96 * result.se[i],
                "nobs": result.nobs,
                "n_cno": data["cno4"].nunique(),
                "r2_twfe": result.r2,
                "sample_note": sample_note,
            }
        )
    out = pd.DataFrame(rows)
    out_path = EVENT_DIR / outcome.key / f"{spec_name}.csv"
    out.to_csv(out_path, index=False)
    plot_event_study(out, outcome, spec_name)
    return out


def plot_event_study(result: pd.DataFrame, outcome: EventOutcome, spec_name: str) -> None:
    plot_df = pd.concat(
        [
            result,
            pd.DataFrame(
                [
                    {
                        "event_time": BASE_EVENT_PERIOD,
                        "coefficient": 0.0,
                        "ci_low": 0.0,
                        "ci_high": 0.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    ).sort_values("event_time")
    plt.rcParams["font.family"] = ["Century Gothic", "DejaVu Sans"]
    plt.rcParams["font.size"] = 9
    plt.rcParams["font.weight"] = "bold"
    fig, ax = plt.subplots(figsize=AIREF_FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axhline(0, color=AIREF_COLORS["text"], linewidth=0.9)
    ax.axvline(-0.5, color=AIREF_COLORS["text"], linestyle="--", linewidth=1)
    ax.fill_between(
        plot_df["event_time"],
        plot_df["ci_low"],
        plot_df["ci_high"],
        color=AIREF_COLORS["ci"],
        alpha=0.55,
        linewidth=0,
    )
    ax.plot(
        plot_df["event_time"],
        plot_df["coefficient"],
        color=AIREF_COLORS["burgundy"],
        marker="o",
        markersize=3.2,
        linewidth=1.6,
    )
    ax.set_title("")
    ax.set_xlabel(f"Months relative to {INTERVENTION_PERIOD} (baseline = {BASE_EVENT_PERIOD})")
    ax.set_ylabel(outcome.ylabel)
    ax.tick_params(axis="both", colors=AIREF_COLORS["text"], direction="out", length=3, width=1.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AIREF_COLORS["text"])
        ax.spines[side].set_linewidth(1.0)
    ax.grid(axis="y", color=AIREF_COLORS["grid"], linewidth=1.0)
    ax.grid(axis="x", visible=False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(AIREF_COLORS["axis_label"])
        label.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(GRAFICOS_DIR / outcome.key / f"{spec_name}.svg", format="svg", facecolor="white")
    fig.savefig(GRAFICOS_DIR / outcome.key / f"{spec_name}.pdf", facecolor="white")
    fig.savefig(GRAFICOS_DIR / outcome.key / f"{spec_name}.png", dpi=300, facecolor="white")
    plot_df.to_excel(GRAFICOS_DIR / outcome.key / f"{spec_name}.xlsx", index=False)
    plt.close(fig)


def run_all_event_studies_for_outcome(panel: pd.DataFrame, outcome: EventOutcome) -> pd.DataFrame:
    specs: list[tuple[str, str, str]] = []
    for key in ["rf", "cosine_weighted", "cosine_nearest"]:
        specs.append((f"continuous_{key}", key, "continuous"))
    for key in ["cosine_weighted", "cosine_nearest"]:
        specs.append((f"top_quartile_vs_zero_{key}", key, "top_vs_zero"))
    for key in ["rf", "cosine_weighted", "cosine_nearest"]:
        specs.append((f"top_quartile_vs_bottom_quartile_{key}", key, "top_vs_bottom"))

    outputs = []
    for spec_name, exposure_key, mode in specs:
        outputs.append(run_event_study(panel, outcome, spec_name, exposure_key, mode))
    combined = pd.concat(outputs, ignore_index=True)
    combined.to_csv(EVENT_DIR / outcome.key / "event_study_coefficients_all.csv", index=False)
    return combined


def run_all_event_studies(panel: pd.DataFrame) -> pd.DataFrame:
    outputs = [run_all_event_studies_for_outcome(panel, outcome) for outcome in EVENT_OUTCOMES]
    combined = pd.concat(outputs, ignore_index=True)
    combined.to_csv(EVENT_DIR / "event_study_coefficients_all_outcomes.csv", index=False)
    return combined


def write_metadata(panel: pd.DataFrame, ols_sample: pd.DataFrame, latex_status: dict[str, str | bool]) -> None:
    metadata = {
        "data_path": str(DATA_PATH),
        "output_dir": str(OUTPUT_DIR),
        "filters": {
            "dimension": "total",
            "category": "Total",
            "gender": "Total",
        },
        "panel": {
            "period_min": panel["period"].min(),
            "period_max": panel["period"].max(),
            "n_periods": int(panel["period"].nunique()),
            "n_cno": int(panel["cno4"].nunique()),
            "n_rows": int(len(panel)),
        },
        "ols": {
            "start": OLS_START,
            "end": OLS_END,
            "outcome": "average monthly log growth in parados, percentage points",
            "n_cno": int(len(ols_sample)),
            "dropped_zero_endpoint_cno": int(panel["cno4"].nunique() - len(ols_sample)),
        },
        "event_studies": {
            "intervention_period_used": INTERVENTION_PERIOD,
            "baseline_event_period": BASE_EVENT_PERIOD,
            "outcomes": {
                outcome.key: {
                    "source_column": outcome.source_column,
                    "regression_column": outcome.regression_column,
                    "transform": outcome.transform,
                    "subfolder": str(EVENT_DIR / outcome.key),
                    "figures_subfolder": str(GRAFICOS_DIR / outcome.key),
                }
                for outcome in EVENT_OUTCOMES
            },
            "estimator": "TWFE OLS with CNO and period fixed effects included explicitly",
            "standard_errors": "clustered by CNO",
            "figure_style": "AIReF: 14.5 cm x 7.25 cm, burgundy main series, gray axes/grid, SVG/PDF/PNG outputs, no in-plot title.",
            "date_note": (
                "User requested September 2022. ChatGPT's public release date was "
                "November 30, 2022; this run preserves the requested September 2022 intervention."
            ),
        },
        "latex": latex_status,
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    ensure_dirs()
    panel = read_total_panel()
    panel.to_csv(OUTPUT_DIR / "total_cno_monthly_panel.csv", index=False)

    ols_sample = make_ols_sample(panel)
    ols_sample.to_csv(OUTPUT_DIR / "ols_growth_sample.csv", index=False)
    _, ols_results = run_ols_table(ols_sample)
    table_tex = write_ols_latex(ols_results, ols_sample)
    wrapper = write_pdf_wrapper(table_tex)
    latex_status = compile_latex(wrapper)

    event_results = run_all_event_studies(panel)
    write_metadata(panel, ols_sample, latex_status)

    print(f"Wrote outputs to: {OUTPUT_DIR}")
    print(f"OLS table: {table_tex}")
    print(f"OLS PDF compiled: {latex_status['compiled']}")
    print(f"Event-study specs: {event_results.groupby('outcome')['spec'].nunique().to_dict()}")
    return 0 if latex_status["compiled"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
