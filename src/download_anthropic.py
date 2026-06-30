from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import requests
import pandas as pd

from .config import ANTHROPIC_ECON_INDEX_RELEASE_URL, ANTHROPIC_JOB_EXPOSURE_URL, PipelineConfig


REQUIRED_COLUMNS = {"occ_code", "title", "observed_exposure"}
ECON_INDEX_CLAUDE_AI_MEMBER = "aei_claude_ai_2026-06-26.csv"
COUNTRY_JOB_USAGE_COLUMNS = {
    "date_start",
    "date_end",
    "geo_id",
    "geo_level",
    "category_name",
    "hierarchy_level",
    "metric_id",
    "value",
    "node_name",
    "node_external_id",
}


def download_anthropic_job_exposure(config: PipelineConfig, refresh: bool = False) -> Path:
    config.ensure_dirs()
    target = config.raw_dir / "anthropic" / "job_exposure.csv"
    if target.exists() and not refresh:
        return target

    response = requests.get(ANTHROPIC_JOB_EXPOSURE_URL, timeout=60)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def load_anthropic_job_exposure(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"occ_code": "string", "title": "string"})
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Anthropic job exposure file missing columns: {sorted(missing)}")

    df = df.copy()
    df["occ_code"] = df["occ_code"].astype("string").str.strip()
    df["title"] = df["title"].astype("string").str.strip()
    df["observed_exposure"] = pd.to_numeric(df["observed_exposure"], errors="coerce")
    df = df.dropna(subset=["occ_code", "title", "observed_exposure"])
    return df


def download_anthropic_economic_index_release(config: PipelineConfig, refresh: bool = False) -> Path:
    config.ensure_dirs()
    target = config.raw_dir / "anthropic" / "release-2026-06-26.zip"
    if target.exists() and not refresh:
        return target

    response = requests.get(ANTHROPIC_ECON_INDEX_RELEASE_URL, timeout=120)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def load_country_job_usage(zip_path: Path, date_start: str | None = None) -> pd.DataFrame:
    with ZipFile(zip_path) as archive:
        with archive.open(ECON_INDEX_CLAUDE_AI_MEMBER) as source:
            df = pd.read_csv(source, dtype={"geo_id": "string", "node_external_id": "string"})

    missing = COUNTRY_JOB_USAGE_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Anthropic Economic Index file missing columns: {sorted(missing)}")

    sub = df[
        (df["geo_id"].isin(["ESP", "USA"]))
        & (df["geo_level"] == "country")
        & (df["category_name"] == "soc_occupation")
        & (df["hierarchy_level"] == 1)
        & (df["metric_id"] == "pct")
    ].copy()
    if sub.empty:
        raise ValueError("No Spain/US SOC major-group country usage rows found.")

    if date_start is None:
        date_start = str(sub["date_start"].max())
    sub = sub[sub["date_start"] == date_start].copy()

    pivot = sub.pivot_table(
        index=["node_external_id", "node_name", "date_start", "date_end"],
        columns="geo_id",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    required_geos = {"ESP", "USA"}
    if not required_geos.issubset(pivot.columns):
        raise ValueError(f"Missing country values for {sorted(required_geos.difference(pivot.columns))} on {date_start}.")

    out = pivot.rename(
        columns={
            "node_external_id": "soc_major_group",
            "node_name": "job_group",
            "ESP": "spain_usage_pct",
            "USA": "us_usage_pct",
        }
    ).copy()
    out["spain_minus_us_pct"] = out["spain_usage_pct"] - out["us_usage_pct"]
    out["spain_usage_pct"] = out["spain_usage_pct"].round(2)
    out["us_usage_pct"] = out["us_usage_pct"].round(2)
    out["spain_minus_us_pct"] = out["spain_minus_us_pct"].round(2)
    return out[
        [
            "soc_major_group",
            "job_group",
            "date_start",
            "date_end",
            "spain_usage_pct",
            "us_usage_pct",
            "spain_minus_us_pct",
        ]
    ].sort_values("soc_major_group", kind="stable")
