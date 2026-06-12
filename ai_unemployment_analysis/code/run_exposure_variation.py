from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES = PROJECT_ROOT / "output" / "tables"
FIGURES = PROJECT_ROOT / "output" / "figures"


def ensure_dirs() -> None:
    for path in [PROCESSED, TABLES, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dirs()
    occ = pd.read_csv(PROCESSED / "occupation_exposure_table.csv", dtype={"cno4": "string"})
    occ["cno4"] = occ["cno4"].str.zfill(4)
    occ["cno2"] = occ["cno4"].str.slice(0, 2)
    exposure = "exposure_nearest"
    panel = pd.read_csv(PROCESSED / "analysis_panel.csv", usecols=["observed_exposure_cosine_nearest"])
    threshold = float(panel["observed_exposure_cosine_nearest"].quantile(0.75))
    occ["is_zero"] = occ[exposure].eq(0)
    occ["is_high"] = occ[exposure].gt(threshold)

    def q25(x: pd.Series) -> float:
        return float(x.quantile(0.25))

    def q75(x: pd.Series) -> float:
        return float(x.quantile(0.75))

    rows = []
    for cno2, group in occ.groupby("cno2", sort=True):
        x = group[exposure]
        highest = group.sort_values(exposure, ascending=False).iloc[0]
        lowest = group.sort_values(exposure, ascending=True).iloc[0]
        rows.append(
            {
                "cno2": cno2,
                "n_ocup4d": int(group["cno4"].nunique()),
                "exposure_min": float(x.min()),
                "exposure_p25": q25(x),
                "exposure_mean": float(x.mean()),
                "exposure_median": float(x.median()),
                "exposure_p75": q75(x),
                "exposure_max": float(x.max()),
                "exposure_range": float(x.max() - x.min()),
                "exposure_sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                "exposure_iqr": float(q75(x) - q25(x)),
                "n_zero_exposure": int(group["is_zero"].sum()),
                "share_zero_exposure": float(group["is_zero"].mean()),
                "n_high_exposure": int(group["is_high"].sum()),
                "share_high_exposure": float(group["is_high"].mean()),
                "highest_cno4": highest["cno4"],
                "highest_occupation_title": highest["occupation_title"],
                "highest_exposure": float(highest[exposure]),
                "lowest_cno4": lowest["cno4"],
                "lowest_occupation_title": lowest["occupation_title"],
                "lowest_exposure": float(lowest[exposure]),
            }
        )
    table = pd.DataFrame(rows).sort_values(["exposure_range", "exposure_sd"], ascending=False)
    table.to_csv(TABLES / "table_exposure_variation_within_cno2.csv", index=False)

    plot = table.loc[table["n_ocup4d"].ge(2)].sort_values("exposure_range", ascending=True)
    fig_height = max(4.8, min(11.5, 0.26 * len(plot)))
    fig, ax = plt.subplots(figsize=(8.6, fig_height))
    ax.barh(plot["cno2"], plot["exposure_range"], color="#456f6b")
    ax.set_xlabel("Within-CNO2 range of nearest exposure")
    ax.set_ylabel("CNO2")
    ax.set_title("AI exposure variation within 2-digit occupations")
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_exposure_variation_within_cno2.png", bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "table": "output/tables/table_exposure_variation_within_cno2.csv",
        "figure": "output/figures/figure_exposure_variation_within_cno2.png",
        "exposure": exposure,
        "high_exposure_threshold_p75": threshold,
        "high_exposure_threshold_source": "data/processed/analysis_panel.csv",
        "interpretation": "Variation is computed across unique CNO4 occupations inside each 2-digit CNO2 family.",
    }
    (PROCESSED / "exposure_variation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
