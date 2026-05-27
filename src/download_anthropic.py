from __future__ import annotations

from pathlib import Path
import requests
import pandas as pd

from .config import ANTHROPIC_JOB_EXPOSURE_URL, PipelineConfig


REQUIRED_COLUMNS = {"occ_code", "title", "observed_exposure"}


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
