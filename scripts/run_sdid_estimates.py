"""Synthetic difference-in-differences estimates for SEPE CNO4 panel.

This is a Python implementation of the core SDID weighting idea because Stata
is not available in the local execution environment. It estimates binary
treatment specifications only: top exposure quartile vs zero exposure and top
exposure quartile vs bottom exposure quartile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "sepe_cno4_monthly_ai_exposure.csv"
OUTPUT_DIR = ROOT / "analysis" / "econometrics_outputs" / "sdid"
FIG_DIR = OUTPUT_DIR / "Graficos"
INTERVENTION_PERIOD = "2022-09"
RANDOM_SEED = 20260604
PLACEBO_REPS = 10
BASE_EVENT_PERIOD = -1
AIREF_COLORS = {
    "burgundy": "#83082A",
    "text": "#404040",
    "axis_label": "#4D4D4D",
    "grid": "#CCCCCC",
    "ci": "#E397A0",
}
AIREF_FIGSIZE = (14.5 / 2.54, 7.25 / 2.54)

EXPOSURES = {
    "rf": "observed_exposure_rf",
    "cosine_weighted": "observed_exposure_cosine_weighted",
    "cosine_nearest": "observed_exposure_cosine_nearest",
}


@dataclass(frozen=True)
class Outcome:
    key: str
    source: str
    transform: str
    label: str


OUTCOMES = [
    Outcome("unemployment", "parados", "log1p", "log registered unemployment"),
    Outcome("contracts", "contratos", "levels", "contracts"),
]


def read_total_panel() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    total = df[
        (df["dimension"] == "total")
        & (df["category"] == "Total")
        & (df["gender"] == "Total")
    ].copy()
    total["cno4"] = total["cno4"].astype(str).str.zfill(4)
    total["period_dt"] = pd.to_datetime(total["period"] + "-01")
    intervention_dt = pd.to_datetime(INTERVENTION_PERIOD + "-01")
    total["post"] = total["period_dt"] >= intervention_dt
    total["event_time"] = (
        (total["period_dt"].dt.year - intervention_dt.year) * 12
        + (total["period_dt"].dt.month - intervention_dt.month)
    )
    total["sdid_unemployment"] = np.log1p(total["parados"])
    total["sdid_contracts"] = total["contratos"]
    return total.sort_values(["cno4", "period_dt"]).reset_index(drop=True)


def simplex_weights(target: np.ndarray, donors: np.ndarray) -> np.ndarray:
    n = donors.shape[0]
    if n == 1:
        return np.ones(1)
    target = np.asarray(target, dtype=float)
    donors = np.asarray(donors, dtype=float)

    def objective(w: np.ndarray) -> float:
        diff = target - w @ donors
        return float(diff @ diff)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    start = np.full(n, 1.0 / n)
    result = optimize.minimize(
        objective,
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )
    if not result.success:
        return start
    weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()


def sdid_att(matrix: pd.DataFrame, treated_units: list[str], control_units: list[str], post: pd.Series) -> dict[str, float]:
    pre_periods = post.index[~post.to_numpy()]
    post_periods = post.index[post.to_numpy()]
    treated = matrix.loc[treated_units]
    control = matrix.loc[control_units]

    treated_pre_mean = treated[pre_periods].mean(axis=0).to_numpy()
    control_pre = control[pre_periods].to_numpy()
    omega = simplex_weights(treated_pre_mean, control_pre)

    control_post_mean = control[post_periods].mean(axis=1).to_numpy()
    control_pre_by_time = control[pre_periods].T.to_numpy()
    lambd = simplex_weights(control_post_mean, control_pre_by_time)

    treated_post = treated[post_periods].to_numpy().mean()
    control_post = float(omega @ control[post_periods].mean(axis=1).to_numpy())
    treated_pre = float(treated[pre_periods].mean(axis=0).to_numpy() @ lambd)
    control_pre_weighted = float(omega @ control[pre_periods].to_numpy() @ lambd)
    att = (treated_post - control_post) - (treated_pre - control_pre_weighted)
    return {
        "att": att,
        "treated_post": treated_post,
        "synthetic_control_post": control_post,
        "treated_pre": treated_pre,
        "synthetic_control_pre": control_pre_weighted,
        "n_positive_unit_weights": int((omega > 1e-8).sum()),
        "n_positive_time_weights": int((lambd > 1e-8).sum()),
    }


def sdid_event_path(matrix: pd.DataFrame, treated_units: list[str], control_units: list[str]) -> pd.DataFrame:
    event_times = pd.Series(matrix.columns, index=matrix.columns).str.slice(0, 7)
    intervention = pd.Period(INTERVENTION_PERIOD, freq="M")
    event_time_values = np.array([(pd.Period(period, freq="M") - intervention).n for period in event_times])
    pre_periods = matrix.columns[event_time_values < 0]
    treated = matrix.loc[treated_units]
    control = matrix.loc[control_units]
    treated_pre_mean = treated[pre_periods].mean(axis=0).to_numpy()
    control_pre = control[pre_periods].to_numpy()
    omega = simplex_weights(treated_pre_mean, control_pre)
    gap = treated.mean(axis=0).to_numpy() - omega @ control.to_numpy()
    try:
        base_gap = float(gap[np.where(event_time_values == BASE_EVENT_PERIOD)[0][0]])
    except IndexError:
        base_gap = float(gap[event_time_values < 0][-1])
    return pd.DataFrame(
        {
            "period": matrix.columns,
            "event_time": event_time_values,
            "estimate": gap - base_gap,
            "treated_mean": treated.mean(axis=0).to_numpy(),
            "synthetic_control_mean": omega @ control.to_numpy(),
        }
    )


def placebo_se(
    matrix: pd.DataFrame,
    n_treated: int,
    post: pd.Series,
    reps: int,
    rng: np.random.Generator,
) -> tuple[float, list[float]]:
    units = np.array(matrix.index)
    estimates = []
    max_draws = reps * 10
    draws = 0
    while len(estimates) < reps and draws < max_draws:
        draws += 1
        fake_treated = rng.choice(units, size=n_treated, replace=False)
        fake_controls = [unit for unit in units if unit not in set(fake_treated)]
        if not fake_controls:
            continue
        estimate = sdid_att(matrix, list(fake_treated), fake_controls, post)["att"]
        if np.isfinite(estimate):
            estimates.append(float(estimate))
    if len(estimates) < 2:
        return np.nan, estimates
    return float(np.std(estimates, ddof=1)), estimates


def build_matrix(panel: pd.DataFrame, outcome: Outcome, units: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    y_col = f"sdid_{outcome.key}"
    sample = panel[panel["cno4"].isin(units)].replace([np.inf, -np.inf], np.nan)
    matrix = sample.pivot_table(index="cno4", columns="period", values=y_col, aggfunc="first")
    post = sample.drop_duplicates("period").set_index("period").loc[matrix.columns, "post"]
    matrix = matrix.dropna(axis=0, how="any")
    return matrix, post


def run_spec(panel: pd.DataFrame, outcome: Outcome, exposure_key: str, mode: str, rng: np.random.Generator) -> dict[str, object]:
    exposure_col = EXPOSURES[exposure_key]
    exposure = panel.drop_duplicates("cno4")[["cno4", exposure_col]].copy()
    q25 = exposure[exposure_col].quantile(0.25)
    q75 = exposure[exposure_col].quantile(0.75)
    if mode == "top_vs_zero":
        if exposure_key == "rf":
            raise ValueError("RF has no zero-exposure control group.")
        eligible = exposure[(exposure[exposure_col] >= q75) | (exposure[exposure_col] == 0)].copy()
        control_rule = "zero exposure"
    elif mode == "top_vs_bottom":
        eligible = exposure[(exposure[exposure_col] >= q75) | (exposure[exposure_col] <= q25)].copy()
        control_rule = "bottom quartile"
    else:
        raise ValueError(mode)
    eligible["treated"] = eligible[exposure_col] >= q75
    units = eligible["cno4"].tolist()
    matrix, post = build_matrix(panel, outcome, units)
    eligible = eligible[eligible["cno4"].isin(matrix.index)]
    treated_units = eligible.loc[eligible["treated"], "cno4"].tolist()
    control_units = eligible.loc[~eligible["treated"], "cno4"].tolist()
    estimate = sdid_att(matrix, treated_units, control_units, post)
    se, placebo_estimates = placebo_se(matrix, len(treated_units), post, PLACEBO_REPS, rng)
    t_stat = estimate["att"] / se if se and np.isfinite(se) and se > 0 else np.nan
    p_value = 2 * stats.norm.sf(abs(t_stat)) if np.isfinite(t_stat) else np.nan
    return {
        "outcome": outcome.key,
        "outcome_label": outcome.label,
        "exposure_measure": exposure_key,
        "mode": mode,
        "control_rule": control_rule,
        "intervention_period": INTERVENTION_PERIOD,
        "att": estimate["att"],
        "std_error_placebo": se,
        "p_value_normal": p_value,
        "ci_low": estimate["att"] - 1.96 * se if np.isfinite(se) else np.nan,
        "ci_high": estimate["att"] + 1.96 * se if np.isfinite(se) else np.nan,
        "n_treated": len(treated_units),
        "n_control": len(control_units),
        "n_periods": matrix.shape[1],
        "n_pre_periods": int((~post).sum()),
        "n_post_periods": int(post.sum()),
        "q25": q25,
        "q75": q75,
        "n_positive_unit_weights": estimate["n_positive_unit_weights"],
        "n_positive_time_weights": estimate["n_positive_time_weights"],
        "placebo_reps_requested": PLACEBO_REPS,
        "placebo_reps_completed": len(placebo_estimates),
        "estimator_note": "Python SDID-style implementation; Stata sdid command unavailable locally.",
    }


def plot_sdid_event_study(event_df: pd.DataFrame, spec_row: dict[str, object]) -> None:
    outcome = str(spec_row["outcome"])
    exposure_measure = str(spec_row["exposure_measure"])
    mode = str(spec_row["mode"])
    se = float(spec_row["std_error_placebo"])
    plot_df = event_df.copy()
    if np.isfinite(se):
        plot_df["ci_low"] = plot_df["estimate"] - 1.96 * se
        plot_df["ci_high"] = plot_df["estimate"] + 1.96 * se
    else:
        plot_df["ci_low"] = np.nan
        plot_df["ci_high"] = np.nan

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
        plot_df["estimate"],
        color=AIREF_COLORS["burgundy"],
        marker="o",
        markersize=3.2,
        linewidth=1.6,
    )
    ax.set_title("")
    ax.set_xlabel(f"Months relative to {INTERVENTION_PERIOD} (baseline = {BASE_EVENT_PERIOD})")
    ylabel = "SDID gap on log registered unemployment" if outcome == "unemployment" else "SDID gap on contracts"
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", colors=AIREF_COLORS["text"], direction="out", length=3, width=1.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AIREF_COLORS["text"])
        ax.spines[side].set_linewidth(1.0)
    ax.grid(axis="y", color=AIREF_COLORS["grid"], linewidth=1.0)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()

    outdir = FIG_DIR / outcome
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"sdid_eventstudy_{exposure_measure}_{mode}"
    fig.savefig(outdir / f"{stem}.svg", format="svg", facecolor="white")
    fig.savefig(outdir / f"{stem}.pdf", facecolor="white")
    fig.savefig(outdir / f"{stem}.png", dpi=300, facecolor="white")
    plot_df.to_excel(outdir / f"{stem}.xlsx", index=False)
    plt.close(fig)


def plot_sdid_levels(event_df: pd.DataFrame, spec_row: dict[str, object]) -> None:
    outcome = str(spec_row["outcome"])
    exposure_measure = str(spec_row["exposure_measure"])
    mode = str(spec_row["mode"])
    plt.rcParams["font.family"] = ["Century Gothic", "DejaVu Sans"]
    plt.rcParams["font.size"] = 9
    plt.rcParams["font.weight"] = "bold"
    fig, ax = plt.subplots(figsize=AIREF_FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axvline(-0.5, color=AIREF_COLORS["text"], linestyle="--", linewidth=1)
    ax.plot(
        event_df["event_time"],
        event_df["treated_mean"],
        color=AIREF_COLORS["burgundy"],
        marker="o",
        markersize=3.0,
        linewidth=1.5,
        label="Treated",
    )
    ax.plot(
        event_df["event_time"],
        event_df["synthetic_control_mean"],
        color=AIREF_COLORS["text"],
        marker="s",
        markersize=2.8,
        linewidth=1.3,
        linestyle="--",
        label="Synthetic control",
    )
    ax.set_title("")
    ax.set_xlabel(f"Months relative to {INTERVENTION_PERIOD}")
    ylabel = "Log registered unemployment" if outcome == "unemployment" else "Contracts"
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", colors=AIREF_COLORS["text"], direction="out", length=3, width=1.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AIREF_COLORS["text"])
        ax.spines[side].set_linewidth(1.0)
    ax.grid(axis="y", color=AIREF_COLORS["grid"], linewidth=1.0)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()

    outdir = FIG_DIR / outcome
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"sdid_levels_{exposure_measure}_{mode}"
    fig.savefig(outdir / f"{stem}.svg", format="svg", facecolor="white")
    fig.savefig(outdir / f"{stem}.pdf", facecolor="white")
    fig.savefig(outdir / f"{stem}.png", dpi=300, facecolor="white")
    event_df.to_excel(outdir / f"{stem}.xlsx", index=False)
    plt.close(fig)


def write_event_study_graphs(panel: pd.DataFrame, results: pd.DataFrame) -> None:
    event_rows = []
    for row in results.to_dict("records"):
        outcome = next(outcome for outcome in OUTCOMES if outcome.key == row["outcome"])
        exposure_key = str(row["exposure_measure"])
        exposure_col = EXPOSURES[exposure_key]
        exposure = panel.drop_duplicates("cno4")[["cno4", exposure_col]].copy()
        q25 = exposure[exposure_col].quantile(0.25)
        q75 = exposure[exposure_col].quantile(0.75)
        if row["mode"] == "top_vs_zero":
            eligible = exposure[(exposure[exposure_col] >= q75) | (exposure[exposure_col] == 0)].copy()
        elif row["mode"] == "top_vs_bottom":
            eligible = exposure[(exposure[exposure_col] >= q75) | (exposure[exposure_col] <= q25)].copy()
        else:
            continue
        eligible["treated"] = eligible[exposure_col] >= q75
        matrix, _ = build_matrix(panel, outcome, eligible["cno4"].tolist())
        eligible = eligible[eligible["cno4"].isin(matrix.index)]
        treated_units = eligible.loc[eligible["treated"], "cno4"].tolist()
        control_units = eligible.loc[~eligible["treated"], "cno4"].tolist()
        event_df = sdid_event_path(matrix, treated_units, control_units)
        event_df.insert(0, "mode", row["mode"])
        event_df.insert(0, "exposure_measure", exposure_key)
        event_df.insert(0, "outcome", outcome.key)
        event_df["std_error_placebo"] = row["std_error_placebo"]
        event_df["ci_low"] = event_df["estimate"] - 1.96 * float(row["std_error_placebo"])
        event_df["ci_high"] = event_df["estimate"] + 1.96 * float(row["std_error_placebo"])
        plot_sdid_event_study(event_df, row)
        plot_sdid_levels(event_df, row)
        event_rows.append(event_df)
    pd.concat(event_rows, ignore_index=True).to_csv(OUTPUT_DIR / "sdid_eventstudy_paths.csv", index=False)


def write_latex_table(results: pd.DataFrame) -> Path:
    path = OUTPUT_DIR / "sdid_estimates.tex"
    view = results.copy()
    view["spec"] = view["exposure_measure"] + ", " + view["mode"].str.replace("_", " ")
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Outcome & Specification & ATT & SE & Treated & Control \\",
        r"\midrule",
    ]
    for _, row in view.iterrows():
        lines.append(
            f"{row['outcome']} & {row['spec']} & {row['att']:.3f} & "
            f"{row['std_error_placebo']:.3f} & {int(row['n_treated'])} & {int(row['n_control'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    panel = read_total_panel()
    specs = []
    for outcome in OUTCOMES:
        for exposure_key in ["cosine_weighted", "cosine_nearest"]:
            specs.append((outcome, exposure_key, "top_vs_zero"))
        for exposure_key in ["rf", "cosine_weighted", "cosine_nearest"]:
            specs.append((outcome, exposure_key, "top_vs_bottom"))

    rows = []
    for outcome, exposure_key, mode in specs:
        rows.append(run_spec(panel, outcome, exposure_key, mode, rng))
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "sdid_estimates_partial.csv", index=False)
        print(pd.DataFrame([rows[-1]])[["outcome", "exposure_measure", "mode", "att", "std_error_placebo", "n_treated", "n_control"]].to_string(index=False), flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "sdid_estimates.csv", index=False)
    write_latex_table(results)
    write_event_study_graphs(panel, results)
    print(results[["outcome", "exposure_measure", "mode", "att", "std_error_placebo", "n_treated", "n_control"]].to_string(index=False))
    print(f"Wrote outputs to: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
