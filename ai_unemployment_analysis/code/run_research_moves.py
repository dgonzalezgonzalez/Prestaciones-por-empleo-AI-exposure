from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES = PROJECT_ROOT / "output" / "tables"
FIGURES = PROJECT_ROOT / "output" / "figures"

FOCUS_CNO2 = ["27", "38", "24", "59"]
EVENT_MONTH_ID = 2022 * 12 + 9


def ensure_dirs() -> None:
    for path in [PROCESSED, TABLES, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def event_var_name(t: int) -> str:
    if t < 0:
        return f"event_m{abs(t)}"
    if t == 0:
        return "event_0"
    return f"event_p{t}"


def add_treatment(panel: pd.DataFrame) -> pd.DataFrame:
    threshold = float(panel["observed_exposure_cosine_nearest"].quantile(0.75))
    out = panel.copy()
    out["treat"] = out["observed_exposure_cosine_nearest"] > threshold
    out["control"] = out["observed_exposure_cosine_nearest"] == 0
    out["post"] = out["month_id"] > EVENT_MONTH_ID
    out["did_post"] = out["treat"].astype(int) * out["post"].astype(int)
    out["event_time"] = out["month_id"] - EVENT_MONTH_ID
    return out


def family_profiles(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    occ = pd.read_csv(PROCESSED / "occupation_exposure_table.csv", dtype={"cno4": "string"})
    validation = pd.read_csv(TABLES / "table_v1_exposure_match_validation_queue.csv", dtype={"cno4": "string"})
    occ["cno2"] = occ["cno4"].str.slice(0, 2)
    validation["cno2"] = validation["cno4"].str.slice(0, 2)
    threshold = float(panel["observed_exposure_cosine_nearest"].quantile(0.75))
    occ["treat"] = occ["exposure_nearest"] > threshold
    occ["control"] = occ["exposure_nearest"] == 0

    profiles = (
        occ.loc[occ["cno2"].isin(FOCUS_CNO2)]
        .sort_values(["cno2", "exposure_nearest"], ascending=[True, False])
        .merge(
            validation[
                [
                    "cno4",
                    "nearest_anthropic_title",
                    "nearest_anthropic_observed_exposure",
                    "nearest_cosine_similarity",
                    "match_review_flag",
                    "weighted_match_rows",
                ]
            ],
            on="cno4",
            how="left",
        )
    )
    profiles.to_csv(TABLES / "table_research_move_focus_cno2_occupation_profiles.csv", index=False)

    summary = (
        profiles.groupby("cno2", as_index=False)
        .agg(
            occupations=("cno4", "nunique"),
            treated_occupations=("treat", "sum"),
            zero_exposure_occupations=("control", "sum"),
            mean_exposure=("exposure_nearest", "mean"),
            max_exposure=("exposure_nearest", "max"),
            total_mean_parados=("mean_parados", "sum"),
            mean_cosine=("nearest_cosine_similarity", "mean"),
            low_similarity_matches=("match_review_flag", lambda x: int((x == "low_similarity").sum())),
        )
    )
    summary.to_csv(TABLES / "table_research_move_focus_cno2_summary.csv", index=False)
    return summary, profiles


def family_event_studies(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    plot_rows = []
    for cno2 in FOCUS_CNO2:
        data = panel.loc[panel["cno2"].eq(cno2) & (panel["treat"] | panel["control"])].copy()
        treated = int(data.loc[data["treat"], "cno4"].nunique())
        controls = int(data.loc[data["control"], "cno4"].nunique())
        clusters = int(data["cno4"].nunique())
        if treated < 2 or controls < 2 or clusters < 6:
            rows.append(
                {
                    "cno2": cno2,
                    "status": "skipped",
                    "treated_occupations": treated,
                    "control_occupations": controls,
                    "clusters": clusters,
                    "reason": "Insufficient treated/control occupations for a within-family event study.",
                }
            )
            continue

        event_times = sorted(int(x) for x in data["event_time"].unique())
        event_terms = []
        for t in event_times:
            if t == -1:
                continue
            name = event_var_name(t)
            data[name] = data["treat"].astype(int) * data["event_time"].eq(t).astype(int)
            event_terms.append(name)

        formula = "ln_parados ~ " + " + ".join(event_terms) + " + C(cno4) + C(period)"
        result = smf.ols(formula, data=data.dropna(subset=["ln_parados"])).fit(
            cov_type="cluster",
            cov_kwds={"groups": data.dropna(subset=["ln_parados"])["cno4"], "use_correction": True},
        )
        pre_terms = [event_var_name(t) for t in event_times if t <= -2]
        b = result.params.loc[pre_terms].to_numpy()
        cov = result.cov_params().loc[pre_terms, pre_terms].to_numpy()
        wald = float(b.T @ np.linalg.pinv(cov) @ b)
        pretrend_p = float(1 - st.chi2.cdf(wald, len(pre_terms)))
        rows.append(
            {
                "cno2": cno2,
                "status": "estimated",
                "treated_occupations": treated,
                "control_occupations": controls,
                "clusters": clusters,
                "nobs": int(result.nobs),
                "pretrend_wald_chi2": wald,
                "pretrend_df": len(pre_terms),
                "pretrend_p_value": pretrend_p,
            }
        )
        for t in event_times:
            if t == -1:
                plot_rows.append({"cno2": cno2, "event_time": -1, "estimate": 0.0, "se": np.nan})
            else:
                term = event_var_name(t)
                plot_rows.append(
                    {
                        "cno2": cno2,
                        "event_time": t,
                        "estimate": float(result.params[term]),
                        "se": float(result.bse[term]),
                    }
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "table_research_move_focus_cno2_event_summary.csv", index=False)
    coefficients = pd.DataFrame(plot_rows)
    if not coefficients.empty:
        coefficients["ci95_low"] = coefficients["estimate"] - 1.96 * coefficients["se"]
        coefficients["ci95_high"] = coefficients["estimate"] + 1.96 * coefficients["se"]
        coefficients.to_csv(TABLES / "table_research_move_focus_cno2_event_coefficients.csv", index=False)
        for cno2, sub in coefficients.groupby("cno2"):
            fig, ax = plt.subplots(figsize=(8.0, 4.2))
            ax.axhline(0, color="#777777", lw=0.9)
            ax.axvline(0, color="#444444", ls="--", lw=1)
            ax.errorbar(
                sub["event_time"],
                sub["estimate"],
                yerr=[sub["estimate"] - sub["ci95_low"], sub["ci95_high"] - sub["estimate"]],
                fmt="o",
                color="#8f2445",
                ecolor="#bd8da0",
                capsize=2,
            )
            ax.set_title(f"Within-family event study: CNO2 {cno2}")
            ax.set_xlabel("Months relative to September 2022")
            ax.set_ylabel("Log-point effect on parados")
            fig.tight_layout()
            fig.savefig(FIGURES / f"figure_research_move_cno2_{cno2}_event_study.png", bbox_inches="tight")
            plt.close(fig)
    return summary


def denominator_inventory() -> pd.DataFrame:
    rows = [
        {
            "source": "INE EPA table 65134",
            "url": "https://www.ine.es/jaxiT3/files/t/csv_bdsc/65134.csv",
            "granularity": "National CNO2-quarter occupied workers by sex",
            "usable_for": "CNO2 national denominator proxy",
            "limitations": "Not CNO4, not monthly, not province-age joint.",
            "status": "implemented_proxy",
        },
        {
            "source": "INE EPA table 66075",
            "url": "https://www.ine.es/jaxiT3/Tabla.htm?L=0&t=66075",
            "granularity": "Occupation by sex and autonomous community, but search result indicates CNO-94/historical structure",
            "usable_for": "Not directly used for CNO4/CNO2 2021-2026 design",
            "limitations": "Classification/history mismatch; needs manual API validation before use.",
            "status": "screened_not_used",
        },
        {
            "source": "EPA microdata",
            "url": "https://www.ine.es/",
            "granularity": "Potentially individual-level occupation, age, region, quarter",
            "usable_for": "Best candidate for true occupation-age-region denominators",
            "limitations": "Requires microdata extraction and disclosure-aware aggregation; likely CNO2/CNO3 rather than CNO4.",
            "status": "recommended_next_data_task",
        },
        {
            "source": "Administrative affiliation / Social Security records",
            "url": "",
            "granularity": "Potential employment stocks by occupation if available internally",
            "usable_for": "Best candidate for monthly administrative denominators",
            "limitations": "Not available in current project folder.",
            "status": "not_available_current_project",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "table_research_move_denominator_inventory.csv", index=False)
    return out


def main() -> None:
    ensure_dirs()
    panel = pd.read_csv(PROCESSED / "analysis_panel.csv", dtype={"cno4": "string"})
    panel["cno2"] = panel["cno4"].str.slice(0, 2)
    panel = add_treatment(panel)
    summary, profiles = family_profiles(panel)
    event_summary = family_event_studies(panel)
    inventory = denominator_inventory()
    metadata = {
        "focus_cno2": FOCUS_CNO2,
        "summary": "output/tables/table_research_move_focus_cno2_summary.csv",
        "profiles": "output/tables/table_research_move_focus_cno2_occupation_profiles.csv",
        "event_summary": "output/tables/table_research_move_focus_cno2_event_summary.csv",
        "denominator_inventory": "output/tables/table_research_move_denominator_inventory.csv",
        "note": "Focus families are from the previous FWL decomposition and memo recommendation.",
    }
    (PROCESSED / "research_moves_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
