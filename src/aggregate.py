from __future__ import annotations

import pandas as pd

from .download_ine import detect_weight_column


EXPOSURE_COLUMN_ORDER = [
    "observed_exposure_rf",
    "observed_exposure_cosine_weighted",
    "observed_exposure_cosine_nearest",
]


def exposure_value_columns(occupation_exposure: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in occupation_exposure.columns
        if column.startswith("observed_exposure_")
    ]
    return sorted(
        columns,
        key=lambda column: (
            EXPOSURE_COLUMN_ORDER.index(column)
            if column in EXPOSURE_COLUMN_ORDER
            else len(EXPOSURE_COLUMN_ORDER),
            column,
        ),
    )


def aggregate_industry_quarter_exposure(
    microdata: pd.DataFrame,
    occupation_exposure: pd.DataFrame,
    weight_column: str | None = None,
    industry_mapping: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = microdata.copy()
    df["OCUP1"] = df["OCUP1"].astype(str).str.strip()
    df["ACT1"] = df["ACT1"].astype(str).str.strip()
    weight_column = weight_column or detect_weight_column(df)
    if weight_column and weight_column in df.columns:
        df["_weight"] = pd.to_numeric(df[weight_column], errors="coerce").fillna(0.0)
    else:
        df["_weight"] = 1.0

    exposure_columns = exposure_value_columns(occupation_exposure)
    if not exposure_columns:
        raise ValueError("Occupation exposure data has no observed_exposure columns.")

    exposure = occupation_exposure[["OCUP1", *exposure_columns]].copy()
    exposure["OCUP1"] = exposure["OCUP1"].astype(str).str.strip()
    for column in exposure_columns:
        exposure[column] = pd.to_numeric(exposure[column], errors="coerce")
    merged = df.merge(exposure, on="OCUP1", how="left")
    primary_column = "observed_exposure_rf" if "observed_exposure_rf" in exposure_columns else exposure_columns[0]
    merged["_covered_weight"] = merged["_weight"].where(merged[primary_column].notna(), 0.0)
    for column in exposure_columns:
        merged[f"_weighted_{column}"] = merged["_weight"].where(merged[column].notna(), 0.0) * merged[column].fillna(0.0)

    grouped = merged.groupby(["ACT1", "quarter"], dropna=False)
    aggregations = {
        "total_weight": ("_weight", "sum"),
        "covered_weight": ("_covered_weight", "sum"),
        "occupation_count": ("OCUP1", "nunique"),
    }
    aggregations.update({f"weighted_{column}": (f"_weighted_{column}", "sum") for column in exposure_columns})
    out = grouped.agg(
        **aggregations,
    ).reset_index()
    out = out[out["total_weight"] > 0].copy()
    out["coverage_share"] = out["covered_weight"] / out["total_weight"]
    out = out[out["covered_weight"] > 0].copy()
    for column in exposure_columns:
        suffix = column.removeprefix("observed_exposure")
        out[f"observed_exposure_cnae{suffix}"] = out[f"weighted_{column}"] / out["covered_weight"]
    out = out.rename(columns={"ACT1": "cnae"})
    if industry_mapping is not None and not industry_mapping.empty:
        labels = industry_mapping.copy()
        labels["cnae"] = labels["cnae"].astype(str).str.strip()
        out["cnae"] = out["cnae"].astype(str).str.strip()
        out = out.merge(labels[["cnae", "industry_name"]], on="cnae", how="left")
    else:
        out["industry_name"] = pd.NA
    exposure_out_columns = [
        f"observed_exposure_cnae{column.removeprefix('observed_exposure')}"
        for column in exposure_columns
    ]
    return out[
        [
            "cnae",
            "industry_name",
            "quarter",
            *exposure_out_columns,
            "total_weight",
            "covered_weight",
            "coverage_share",
            "occupation_count",
        ]
    ].sort_values(["quarter", "cnae"]).reset_index(drop=True)
