from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin
import csv
import re
import zipfile

import pandas as pd
import requests

from .config import INE_EPA_MICRODATA_PAGE, PipelineConfig


@dataclass(frozen=True)
class IneFile:
    quarter: str
    microdata_path: Path
    metadata_path: Path | None = None


def load_ine_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"quarter", "microdata_url"}
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"INE manifest missing columns: {sorted(missing)}")
    return rows


def discover_candidate_links(page_url: str = INE_EPA_MICRODATA_PAGE) -> list[str]:
    response = requests.get(page_url, timeout=60)
    response.raise_for_status()
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', response.text, flags=re.IGNORECASE)
    candidates = []
    for href in hrefs:
        absolute = urljoin(page_url, href)
        lower = absolute.lower()
        if any(ext in lower for ext in [".zip", ".csv", ".txt", ".xlsx"]):
            candidates.append(absolute)
    return sorted(set(candidates))


def _download(url: str, target: Path, refresh: bool) -> Path:
    if target.exists() and not refresh:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def download_ine_from_manifest(
    config: PipelineConfig,
    manifest_path: Path,
    refresh: bool = False,
    max_quarters: int | None = None,
) -> list[IneFile]:
    rows = load_ine_manifest(manifest_path)
    if max_quarters:
        rows = rows[:max_quarters]

    files: list[IneFile] = []
    for row in rows:
        quarter = row["quarter"].strip()
        micro_url = row["microdata_url"].strip()
        meta_url = row.get("metadata_url", "").strip()
        micro_name = row.get("microdata_filename", "").strip() or Path(micro_url).name or f"{quarter}.zip"
        meta_name = row.get("metadata_filename", "").strip() or (Path(meta_url).name if meta_url else "")
        micro_path = _download(micro_url, config.raw_dir / "ine" / quarter / micro_name, refresh)
        metadata_path = None
        if meta_url:
            metadata_path = _download(meta_url, config.raw_dir / "ine" / quarter / meta_name, refresh)
        files.append(IneFile(quarter=quarter, microdata_path=micro_path, metadata_path=metadata_path))
    return files


def find_first_tabular_file(path: Path) -> Path:
    if path.suffix.lower() != ".zip":
        return path
    out_dir = path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = (out_dir / member.filename).resolve()
            if not str(target).startswith(str(out_dir.resolve())):
                raise ValueError(f"Unsafe path in zip archive {path}: {member.filename}")
        archive.extractall(out_dir)
    candidates = [
        p for p in out_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".dat", ".tab"}
    ]
    if not candidates:
        raise ValueError(f"No CSV/TXT/DAT file found inside {path}")
    preferred = [
        item for item in candidates
        if item.suffix.lower() in {".csv", ".tab"} or "csv" in {part.lower() for part in item.parts}
    ]
    pool = preferred or candidates
    return sorted(pool, key=lambda item: item.stat().st_size, reverse=True)[0]


def read_ine_microdata(path: Path, quarter: str) -> pd.DataFrame:
    tabular = find_first_tabular_file(path)
    read_kwargs = {"dtype": "string", "encoding_errors": "replace"}
    try:
        df = pd.read_csv(tabular, sep=None, engine="python", **read_kwargs)
    except Exception:
        df = pd.read_csv(tabular, sep=";", engine="python", **read_kwargs)

    df.columns = [str(col).strip().upper() for col in df.columns]
    missing = {"OCUP1", "ACT1"}.difference(df.columns)
    if missing:
        raise ValueError(
            f"INE microdata {tabular} missing {sorted(missing)}. "
            "Check file format or update parser for fixed-width layout."
        )

    df["quarter"] = quarter
    for col in ["OCUP1", "ACT1"]:
        df[col] = df[col].astype("string").str.strip()
    df = df[df["OCUP1"].notna() & df["ACT1"].notna() & (df["OCUP1"] != "") & (df["ACT1"] != "")]
    keep_columns = ["quarter", "OCUP1", "ACT1"]
    weight_column = detect_weight_column(df)
    if weight_column:
        keep_columns.append(weight_column)
    return df[keep_columns].copy()


def detect_weight_column(df: pd.DataFrame) -> str | None:
    for candidate in ["FACTOREL", "FACTOR", "PESO", "WEIGHT"]:
        if candidate in df.columns:
            return candidate
    return None
