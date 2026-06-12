from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "sepe_cno4_monthly_ai_exposure.csv"
PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES = PROJECT_ROOT / "output" / "tables"
FIGURES = PROJECT_ROOT / "output" / "figures"

EXPOSURE = "observed_exposure_cosine_nearest"
EVENT_MONTH_ID = 2022 * 12 + 9


@dataclass
class SyntheticPanel:
    design: str
    status: str
    unit_col: str
    outcome: str
    panel: pd.DataFrame | None = None
    note: str = ""
    max_donors: int = 600


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
    chunks: list[pd.DataFrame] = []
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


def add_treatment(panel: pd.DataFrame) -> pd.DataFrame:
    total = pd.read_csv(PROCESSED / "analysis_panel.csv", dtype={"cno4": "string"})
    threshold = float(total[EXPOSURE].quantile(0.75))
    out = panel.copy()
    out["treat"] = out[EXPOSURE] > threshold
    out["control"] = out[EXPOSURE] == 0
    out["threshold_p75"] = threshold
    return out.loc[out["treat"] | out["control"]].copy()


def build_total_panel() -> SyntheticPanel:
    panel = pd.read_csv(PROCESSED / "analysis_panel.csv", dtype={"cno4": "string"})
    panel = add_treatment(panel)
    panel["unit"] = panel["cno4"]
    return SyntheticPanel("total_ocup4d", "available", "unit", "ln_parados", panel, "CNO4 national total panel.")


def build_province_panel() -> SyntheticPanel:
    panel = add_treatment(read_dimension("province"))
    panel["unit"] = panel["category"].astype(str) + "::" + panel["cno4"].astype(str)
    return SyntheticPanel(
        "province_ocup4d",
        "available",
        "unit",
        "ln_parados_p1",
        panel,
        "Province-CNO4 panel. Uses log(parados + 1) because province cells often contain zeros.",
        max_donors=600,
    )


def build_age3039_panel() -> SyntheticPanel:
    panel = read_dimension("age")
    panel = panel.loc[panel["category"].eq("30-39")].copy()
    panel = add_treatment(panel)
    panel["unit"] = panel["cno4"]
    return SyntheticPanel("age3039_ocup4d", "available", "unit", "ln_parados", panel, "National age 30-39 CNO4 panel.")


def build_province_age3039_panel() -> SyntheticPanel:
    # The SEPE CSV stores age and province as separate dimensions; it does not
    # contain province-by-age-by-CNO4 cells. This cannot be recovered without a
    # joint cross-tab from the source.
    dimensions = []
    for chunk in pd.read_csv(RAW_CSV, usecols=["dimension"], chunksize=500_000):
        dimensions.extend(chunk["dimension"].dropna().unique().tolist())
    dims = sorted(set(dimensions))
    return SyntheticPanel(
        "province_age3039_ocup4d",
        "not_available",
        "unit",
        "ln_parados_p1",
        None,
        "Not estimable from current CSV: available dimensions are "
        + ", ".join(dims)
        + "; no joint province-by-age cells exist.",
    )


def solve_simplex(
    donor_pre: np.ndarray,
    target_pre: np.ndarray,
    ridge: float = 0.0,
    intercept: bool = False,
) -> tuple[np.ndarray, float, str]:
    n_donors = donor_pre.shape[0]
    weights = cp.Variable(n_donors)
    alpha = cp.Variable() if intercept else 0.0
    objective = cp.sum_squares(target_pre - alpha - donor_pre.T @ weights)
    if ridge > 0:
        objective += ridge * cp.sum_squares(weights)
    problem = cp.Problem(cp.Minimize(objective), [weights >= 0, cp.sum(weights) == 1])
    for solver in ["CLARABEL", "ECOS", "OSQP", "SCS"]:
        try:
            problem.solve(solver=solver, verbose=False)
            if weights.value is not None:
                return np.maximum(weights.value, 0) / np.maximum(weights.value, 0).sum(), float(alpha.value) if intercept else 0.0, solver
        except Exception:
            continue
    raise RuntimeError("No cvxpy solver produced simplex weights.")


def solve_time_weights(control_pre: np.ndarray, control_post_mean: np.ndarray, ridge: float = 0.0) -> tuple[np.ndarray, float, str]:
    t_pre = control_pre.shape[1]
    lamb = cp.Variable(t_pre)
    beta = cp.Variable()
    objective = cp.sum_squares(control_post_mean - beta - control_pre @ lamb)
    if ridge > 0:
        objective += ridge * cp.sum_squares(lamb)
    problem = cp.Problem(cp.Minimize(objective), [lamb >= 0, cp.sum(lamb) == 1])
    for solver in ["CLARABEL", "ECOS", "OSQP", "SCS"]:
        try:
            problem.solve(solver=solver, verbose=False)
            if lamb.value is not None:
                return np.maximum(lamb.value, 0) / np.maximum(lamb.value, 0).sum(), float(beta.value), solver
        except Exception:
            continue
    raise RuntimeError("No cvxpy solver produced time weights.")


def prepare_matrix(synth_panel: SyntheticPanel) -> dict:
    if synth_panel.panel is None:
        return {"status": synth_panel.status, "note": synth_panel.note}
    df = synth_panel.panel.dropna(subset=[synth_panel.outcome, synth_panel.unit_col, "period", "month_id"]).copy()
    unit_treatment = df.groupby(synth_panel.unit_col)["treat"].first()
    wide = df.pivot_table(index=synth_panel.unit_col, columns="month_id", values=synth_panel.outcome, aggfunc="mean")
    wide = wide.dropna(axis=0)
    common_units = wide.index.intersection(unit_treatment.index)
    wide = wide.loc[common_units].sort_index()
    unit_treatment = unit_treatment.loc[wide.index]
    periods = np.array(wide.columns.tolist(), dtype=int)
    pre_mask = periods <= EVENT_MONTH_ID
    post_mask = periods > EVENT_MONTH_ID
    treated_units = wide.index[unit_treatment.to_numpy(dtype=bool)].tolist()
    control_units = wide.index[(~unit_treatment.to_numpy(dtype=bool))].tolist()
    if len(treated_units) == 0 or len(control_units) == 0:
        return {"status": "failed", "note": "No treated or control units after balancing."}
    return {
        "status": "available",
        "wide": wide,
        "periods": periods,
        "pre_mask": pre_mask,
        "post_mask": post_mask,
        "treated_units": treated_units,
        "control_units": control_units,
    }


def estimate_synthetic_design(synth_panel: SyntheticPanel) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    prepared = prepare_matrix(synth_panel)
    if prepared["status"] != "available":
        return (
            {
                "design": synth_panel.design,
                "status": prepared["status"],
                "note": prepared["note"],
            },
            pd.DataFrame(),
            pd.DataFrame(),
        )

    wide: pd.DataFrame = prepared["wide"]
    periods = prepared["periods"]
    pre_mask = prepared["pre_mask"]
    post_mask = prepared["post_mask"]
    treated_units = prepared["treated_units"]
    control_units = prepared["control_units"]

    y_treat = wide.loc[treated_units].to_numpy(dtype=float)
    y_control_all = wide.loc[control_units].to_numpy(dtype=float)
    treated_pre = y_treat[:, pre_mask].mean(axis=0)
    treated_post = y_treat[:, post_mask].mean(axis=0)

    distances = np.sqrt(((y_control_all[:, pre_mask] - treated_pre) ** 2).mean(axis=1))
    order = np.argsort(distances)
    keep_n = min(synth_panel.max_donors, len(control_units))
    keep_idx = order[:keep_n]
    selected_controls = [control_units[i] for i in keep_idx]
    y_control = y_control_all[keep_idx]

    sc_weights, sc_alpha, sc_solver = solve_simplex(y_control[:, pre_mask], treated_pre, ridge=0.0, intercept=False)
    sdid_unit_weights, sdid_alpha, sdid_unit_solver = solve_simplex(
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

    sc_path = sc_weights @ y_control
    sdid_control_pre = sdid_unit_weights @ y_control[:, pre_mask]
    sdid_control_post = sdid_unit_weights @ y_control[:, post_mask]

    treated_pre_gap_sc = treated_pre - sc_path[pre_mask]
    treated_post_gap_sc = treated_post - sc_path[post_mask]
    sc_att = float(treated_post_gap_sc.mean())
    sc_did_adjusted = float(treated_post_gap_sc.mean() - treated_pre_gap_sc.mean())

    sdid_post_gap = float(treated_post.mean() - sdid_control_post.mean())
    sdid_pre_gap_weighted = float(time_weights @ (treated_pre - sdid_control_pre))
    sdid_att = float(sdid_post_gap - sdid_pre_gap_weighted)

    uniform_control = y_control_all.mean(axis=0)
    did_att = float((treated_post.mean() - uniform_control[post_mask].mean()) - (treated_pre.mean() - uniform_control[pre_mask].mean()))

    result = {
        "design": synth_panel.design,
        "status": "estimated",
        "outcome": synth_panel.outcome,
        "treated_units": len(treated_units),
        "control_units_available": len(control_units),
        "control_units_selected": len(selected_controls),
        "periods": len(periods),
        "pre_periods": int(pre_mask.sum()),
        "post_periods": int(post_mask.sum()),
        "first_period_month_id": int(periods.min()),
        "last_period_month_id": int(periods.max()),
        "sdid_att_log_points": sdid_att,
        "synthetic_control_att_log_points": sc_att,
        "synthetic_control_did_adjusted_log_points": sc_did_adjusted,
        "uniform_did_log_points": did_att,
        "sdid_effect_pct": 100 * (math.exp(sdid_att) - 1),
        "sc_effect_pct": 100 * (math.exp(sc_att) - 1),
        "sc_did_adjusted_effect_pct": 100 * (math.exp(sc_did_adjusted) - 1),
        "uniform_did_effect_pct": 100 * (math.exp(did_att) - 1),
        "sc_pre_rmspe": float(np.sqrt(np.mean(treated_pre_gap_sc**2))),
        "sdid_pre_rmspe": float(np.sqrt(np.mean((treated_pre - sdid_alpha - sdid_control_pre) ** 2))),
        "sdid_unit_intercept": float(sdid_alpha),
        "sdid_time_intercept": float(time_alpha),
        "donor_screening_note": f"Selected {len(selected_controls)} controls with lowest pre-period RMS distance to treated aggregate.",
        "sc_solver": sc_solver,
        "sdid_unit_solver": sdid_unit_solver,
        "sdid_time_solver": time_solver,
        "note": synth_panel.note,
    }

    weight_rows = pd.DataFrame(
        {
            "design": synth_panel.design,
            "control_unit": selected_controls,
            "screening_pre_distance": distances[keep_idx],
            "sc_weight": sc_weights,
            "sdid_unit_weight": sdid_unit_weights,
        }
    ).sort_values("sdid_unit_weight", ascending=False)

    path = pd.DataFrame(
        {
            "design": synth_panel.design,
            "month_id": periods,
            "relative_month": periods - EVENT_MONTH_ID,
            "treated_mean": y_treat.mean(axis=0),
            "uniform_control_mean": y_control_all.mean(axis=0),
            "synthetic_control_mean": sc_path,
            "sdid_unit_weighted_control_mean": sdid_unit_weights @ y_control,
            "period_type": np.where(pre_mask, "pre", "post"),
        }
    )
    return result, weight_rows, path


def plot_paths(paths: pd.DataFrame) -> None:
    for design, sub in paths.groupby("design"):
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        ax.plot(sub["relative_month"], sub["treated_mean"], color="#8f2445", lw=2.0, label="Treated high exposure")
        ax.plot(sub["relative_month"], sub["synthetic_control_mean"], color="#325d79", lw=1.8, label="Synthetic control")
        ax.plot(sub["relative_month"], sub["sdid_unit_weighted_control_mean"], color="#8c7a3f", lw=1.6, label="SDID unit-weighted control")
        ax.axvline(0, color="#444444", ls="--", lw=1)
        ax.set_title(f"Synthetic fit: {design}")
        ax.set_xlabel("Months relative to September 2022")
        ax.set_ylabel("Outcome")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIGURES / f"figure_synthetic_{design}.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ensure_dirs()
    designs = [
        build_total_panel(),
        build_province_panel(),
        build_age3039_panel(),
        build_province_age3039_panel(),
    ]
    results = []
    weights = []
    paths = []
    for design in designs:
        result, weight_table, path_table = estimate_synthetic_design(design)
        results.append(result)
        if not weight_table.empty:
            weights.append(weight_table)
        if not path_table.empty:
            paths.append(path_table)

    result_df = pd.DataFrame(results)
    result_df.to_csv(TABLES / "table_synthetic_did_control_results.csv", index=False)
    if weights:
        pd.concat(weights, ignore_index=True).to_csv(TABLES / "table_synthetic_donor_weights.csv", index=False)
    if paths:
        path_df = pd.concat(paths, ignore_index=True)
        path_df.to_csv(TABLES / "table_synthetic_paths.csv", index=False)
        plot_paths(path_df)

    metadata = {
        "results": "output/tables/table_synthetic_did_control_results.csv",
        "weights": "output/tables/table_synthetic_donor_weights.csv",
        "paths": "output/tables/table_synthetic_paths.csv",
        "notes": "Synthetic control and SDID-style estimators use high-exposure treated aggregate and zero-exposure donor units. Province-age 30-39 is not observed in the current SEPE CSV.",
    }
    (PROCESSED / "synthetic_methods_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
