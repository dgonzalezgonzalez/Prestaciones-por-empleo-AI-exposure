from __future__ import annotations

import pandas as pd

from .download_ine import detect_weight_column


def aggregate_industry_quarter_exposure(
    microdata: pd.DataFrame,
    occupation_exposure: pd.DataFrame,
    weight_column: str | None = None,
) -> pd.DataFrame:
    df = microdata.copy()
    df["OCUP1"] = df["OCUP1"].astype(str).str.strip()
    df["ACT1"] = df["ACT1"].astype(str).str.strip()
    weight_column = weight_column or detect_weight_column(df)
    if weight_column and weight_column in df.columns:
        df["_weight"] = pd.to_numeric(df[weight_column], errors="coerce").fillna(0.0)
    else:
        df["_weight"] = 1.0

    exposure = occupation_exposure[["OCUP1", "observed_exposure"]].copy()
    exposure["OCUP1"] = exposure["OCUP1"].astype(str).str.strip()
    exposure["observed_exposure"] = pd.to_numeric(exposure["observed_exposure"], errors="coerce")
    merged = df.merge(exposure, on="OCUP1", how="left")
    merged["_covered_weight"] = merged["_weight"].where(merged["observed_exposure"].notna(), 0.0)
    merged["_weighted_exposure"] = merged["_covered_weight"] * merged["observed_exposure"].fillna(0.0)

    grouped = merged.groupby(["ACT1", "quarter"], dropna=False)
    out = grouped.agg(
        total_weight=("_weight", "sum"),
        covered_weight=("_covered_weight", "sum"),
        weighted_exposure=("_weighted_exposure", "sum"),
        occupation_count=("OCUP1", "nunique"),
    ).reset_index()
    out = out[out["total_weight"] > 0].copy()
    out["coverage_share"] = out["covered_weight"] / out["total_weight"]
    out["observed_exposure_cnae"] = out["weighted_exposure"] / out["covered_weight"]
    out = out[out["covered_weight"] > 0].copy()
    out = out.rename(columns={"ACT1": "cnae"})
    return out[
        [
            "cnae",
            "quarter",
            "observed_exposure_cnae",
            "total_weight",
            "covered_weight",
            "coverage_share",
            "occupation_count",
        ]
    ].sort_values(["quarter", "cnae"]).reset_index(drop=True)
