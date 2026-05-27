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
    period: str
    microdata_path: Path
    metadata_path: Path | None = None

    @property
    def quarter(self) -> str:
        return self.period


@dataclass(frozen=True)
class IneSourceSpec:
    source: str
    occupation_column: str
    industry_column: str
    default_period: str
    valid_code_pattern: str
    raw_subdir: str


EPA_SOURCE = IneSourceSpec(
    source="epa",
    occupation_column="OCUP1",
    industry_column="ACT1",
    default_period="quarter",
    valid_code_pattern=r"\d{1,2}",
    raw_subdir="ine/epa",
)

CENSUS_SOURCE = IneSourceSpec(
    source="census",
    occupation_column="OCU63",
    industry_column="ACT89",
    default_period="2021",
    valid_code_pattern=r"\d{2}",
    raw_subdir="ine/census",
)


def get_ine_source_spec(source: str) -> IneSourceSpec:
    normalized = source.lower().strip()
    if normalized == "epa":
        return EPA_SOURCE
    if normalized == "census":
        return CENSUS_SOURCE
    raise ValueError(f"Unsupported INE source: {source}")


def load_ine_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"microdata_url"}
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
    source_spec: IneSourceSpec = EPA_SOURCE,
    refresh: bool = False,
    max_quarters: int | None = None,
) -> list[IneFile]:
    rows = load_ine_manifest(manifest_path)
    if max_quarters:
        rows = rows[:max_quarters]

    files: list[IneFile] = []
    for row in rows:
        period = (
            row.get("period", "").strip()
            or row.get("quarter", "").strip()
            or row.get("year", "").strip()
            or source_spec.default_period
        )
        micro_url = row["microdata_url"].strip()
        meta_url = row.get("metadata_url", "").strip()
        micro_name = row.get("microdata_filename", "").strip() or Path(micro_url).name or f"{period}.zip"
        meta_name = row.get("metadata_filename", "").strip() or (Path(meta_url).name if meta_url else "")
        base_dir = config.raw_dir / source_spec.raw_subdir / period
        micro_path = _download(micro_url, base_dir / micro_name, refresh)
        metadata_path = None
        if meta_url:
            metadata_path = _download(meta_url, base_dir / meta_name, refresh)
        files.append(IneFile(period=period, microdata_path=micro_path, metadata_path=metadata_path))
    return files


def _extract_preferred_tabular_file(path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        for member in members:
            target = (out_dir / member.filename).resolve()
            if not str(target).startswith(str(out_dir.resolve())):
                raise ValueError(f"Unsafe path in zip archive {path}: {member.filename}")
        tabular_members = [
            member
            for member in members
            if not member.is_dir() and Path(member.filename).suffix.lower() in {".csv", ".txt", ".dat", ".tab"}
        ]
        if not tabular_members:
            raise ValueError(f"No CSV/TXT/DAT file found inside {path}")
        preferred = [
            item
            for item in tabular_members
            if Path(item.filename).suffix.lower() in {".csv", ".tab"}
            or "csv" in {part.lower() for part in Path(item.filename).parts}
        ]
        member = sorted(preferred or tabular_members, key=lambda item: item.file_size, reverse=True)[0]
        target = out_dir / member.filename
        if not target.exists():
            archive.extract(member, out_dir)


def find_first_tabular_file(path: Path) -> Path:
    if path.suffix.lower() != ".zip":
        return path
    out_dir = path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    _extract_preferred_tabular_file(path, out_dir)
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


def read_ine_microdata(
    path: Path,
    period: str,
    source_spec: IneSourceSpec = EPA_SOURCE,
) -> pd.DataFrame:
    tabular = find_first_tabular_file(path)
    if source_spec.source == "census":
        return _read_census_microdata(tabular, period, source_spec)

    read_kwargs = {"dtype": "string", "encoding_errors": "replace"}
    try:
        df = pd.read_csv(tabular, sep=None, engine="python", **read_kwargs)
    except Exception:
        df = pd.read_csv(tabular, sep=";", engine="python", **read_kwargs)

    df.columns = [str(col).strip().upper() for col in df.columns]
    occupation_column = source_spec.occupation_column
    industry_column = source_spec.industry_column
    missing = {occupation_column, industry_column}.difference(df.columns)
    if missing and source_spec.source == "census":
        df = pd.read_fwf(
            tabular,
            colspecs=[(101, 103), (103, 105)],
            names=[occupation_column, industry_column],
            dtype="string",
            encoding_errors="replace",
        )
        missing = set()
    if missing:
        raise ValueError(
            f"INE microdata {tabular} missing {sorted(missing)}. "
            "Check file format or update parser for fixed-width layout."
        )

    for col in [occupation_column, industry_column]:
        df[col] = df[col].astype("string").str.strip()
    keep = (
        df[occupation_column].notna()
        & df[industry_column].notna()
        & df[occupation_column].str.fullmatch(source_spec.valid_code_pattern)
        & df[industry_column].str.fullmatch(source_spec.valid_code_pattern)
    )
    df = df[keep].copy()
    df["period"] = period
    df["quarter"] = period
    df["OCUP1"] = df[occupation_column]
    df["ACT1"] = df[industry_column]

    keep_columns = ["period", "quarter", "OCUP1", "ACT1"]
    weight_column = detect_weight_column(df)
    if weight_column:
        keep_columns.append(weight_column)
    return df[keep_columns].copy()


def _read_census_microdata(tabular: Path, period: str, source_spec: IneSourceSpec) -> pd.DataFrame:
    occupation_column = source_spec.occupation_column
    industry_column = source_spec.industry_column
    df = None
    for sep in ["\t", ";", ","]:
        try:
            df = pd.read_csv(
                tabular,
                sep=sep,
                dtype="string",
                encoding_errors="replace",
                usecols=[occupation_column, industry_column],
            )
            df.columns = [str(col).strip().upper().strip('"') for col in df.columns]
            break
        except Exception:
            continue
    if df is None:
        df = pd.read_fwf(
            tabular,
            colspecs=[(101, 103), (103, 105)],
            names=[occupation_column, industry_column],
            dtype="string",
            encoding_errors="replace",
        )

    for col in [occupation_column, industry_column]:
        df[col] = df[col].astype("string").str.strip()
    keep = (
        df[occupation_column].notna()
        & df[industry_column].notna()
        & df[occupation_column].str.fullmatch(source_spec.valid_code_pattern)
        & df[industry_column].str.fullmatch(source_spec.valid_code_pattern)
    )
    df = df[keep].copy()
    df["period"] = period
    df["quarter"] = period
    df["OCUP1"] = df[occupation_column]
    df["ACT1"] = df[industry_column]
    return df[["period", "quarter", "OCUP1", "ACT1"]].copy()


def detect_weight_column(df: pd.DataFrame) -> str | None:
    for candidate in ["FACTOREL", "FACTOR", "PESO", "WEIGHT"]:
        if candidate in df.columns:
            return candidate
    return None
