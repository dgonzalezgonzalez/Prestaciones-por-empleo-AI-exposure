from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import requests

from .config import PipelineConfig
from .utils import clean_occupation_title, normalize_space


ONET_OCCUPATION_DATA_URL = "https://www.onetcenter.org/dl_files/database/db_30_3_excel/Occupation%20Data.xlsx"
CNO11_NOTES_URL = "https://ine.es/daco/daco42/clasificaciones/cno11_notas.pdf"
EPA_CNO2_WEIGHTS_URL = "https://www.ine.es/jaxiT3/files/t/csv_bdsc/65134.csv"

FOUR_DIGIT_HEADING_RE = re.compile(r"^(\d{4})\s+(.+)$")
LOWER_HEADING_RE = re.compile(r"^(\d{1,3}|[A-Z])\s+")
SECTION_MARKERS = [
    "Entre sus tareas se incluyen:",
    "Ejemplos de ocupaciones incluidas en este grupo primario",
    "Ocupaciones afines no incluidas en este grupo primario",
]


def ensure_reference_file(url: str, target: Path, refresh: bool = False) -> Path:
    if target.exists() and not refresh:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def load_onet_occupation_descriptions(config: PipelineConfig, refresh: bool = False) -> pd.DataFrame:
    path = ensure_reference_file(
        ONET_OCCUPATION_DATA_URL,
        config.raw_dir / "anthropic" / "Occupation_Data_30_3.xlsx",
        refresh,
    )
    df = pd.read_excel(path, dtype="string")
    df = df.rename(
        columns={
            "O*NET-SOC Code": "occ_code",
            "Title": "onet_title",
            "Description": "onet_description",
        }
    )
    df["occ_code"] = df["occ_code"].astype("string").str.replace(r"\.00$", "", regex=True)
    keep = ["occ_code", "onet_title", "onet_description"]
    return df[keep].dropna(subset=["occ_code"]).copy()


def add_onet_descriptions(anthropic: pd.DataFrame, config: PipelineConfig, refresh: bool = False) -> pd.DataFrame:
    descriptions = load_onet_occupation_descriptions(config, refresh)
    out = anthropic.copy()
    out["occ_code"] = out["occ_code"].astype("string").str.replace(r"\.00$", "", regex=True)
    out = out.merge(descriptions, on="occ_code", how="left")
    out["embedding_text"] = out.apply(_anthropic_embedding_text, axis=1)
    return out


def _anthropic_embedding_text(row: pd.Series) -> str:
    title = clean_occupation_title(row["title"])
    raw_description = row.get("onet_description", "")
    description = "" if pd.isna(raw_description) else normalize_space(raw_description)
    if description:
        return f"O*NET occupation: {title}. Description: {description}"
    return title


def load_cno4_records(config: PipelineConfig, refresh: bool = False) -> pd.DataFrame:
    path = ensure_reference_file(CNO11_NOTES_URL, config.raw_dir / "ine" / "cno11_notas.pdf", refresh)
    return parse_cno4_pdf(path)


def parse_cno4_pdf(path: Path) -> pd.DataFrame:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    lines: list[str] = []
    for page in reader.pages[2:]:
        text = page.extract_text() or ""
        for line in text.splitlines():
            cleaned = normalize_space(line.replace("\xad", ""))
            if cleaned:
                lines.append(cleaned)

    records: list[dict[str, str]] = []
    idx = 0
    while idx < len(lines):
        match = FOUR_DIGIT_HEADING_RE.match(lines[idx])
        if not match:
            idx += 1
            continue
        code = match.group(1)
        title_parts = [match.group(2)]
        idx += 1
        while idx < len(lines) and not _starts_definition(lines[idx]) and not FOUR_DIGIT_HEADING_RE.match(lines[idx]):
            if LOWER_HEADING_RE.match(lines[idx]):
                break
            title_parts.append(lines[idx])
            idx += 1
        body = []
        while idx < len(lines) and not FOUR_DIGIT_HEADING_RE.match(lines[idx]):
            body.append(lines[idx])
            idx += 1
        title = _repair_pdf_text(" ".join(title_parts))
        structured = build_cno4_structured_text(code, title, body)
        records.append(
            {
                "CNO4": code,
                "CNO2": code[:2],
                "OCUP1": code[:1],
                "occupation_title": title,
                "cno_structured_text_es": structured,
                "embedding_text": structured,
            }
        )
    if not records:
        raise ValueError(f"No 4-digit CNO records parsed from {path}")
    return pd.DataFrame(records).drop_duplicates(subset=["CNO4"]).sort_values("CNO4").reset_index(drop=True)


def build_cno4_structured_text(code: str, title: str, body_lines: list[str]) -> str:
    buckets = {
        "definition": [],
        "tasks": [],
        "examples": [],
        "excluded": [],
    }
    current = "definition"
    for raw in body_lines:
        line = _repair_pdf_text(raw)
        if not line:
            continue
        if line == SECTION_MARKERS[0]:
            current = "tasks"
            continue
        if line == SECTION_MARKERS[1]:
            current = "examples"
            continue
        if line == SECTION_MARKERS[2]:
            current = "excluded"
            continue
        if LOWER_HEADING_RE.match(line) and not line.startswith("-"):
            continue
        buckets[current].append(line)

    definition = _limit_text(" ".join(buckets["definition"]), 1200)
    tasks = _limit_text(" ".join(_clean_bullets(buckets["tasks"])), 1200)
    examples = _limit_text("; ".join(_clean_bullets(buckets["examples"])), 700)
    excluded = _limit_text("; ".join(_clean_bullets(buckets["excluded"])), 700)
    parts = [
        f"CNO occupation: {code} {title}.",
        f"Definition: {definition}" if definition else "",
        f"Typical tasks: {tasks}" if tasks else "",
        f"Examples included: {examples}" if examples else "",
        f"Related or excluded occupations: {excluded}" if excluded else "",
    ]
    return normalize_space(" ".join(part for part in parts if part))


def aggregate_cno4_predictions(
    cno4_predictions: pd.DataFrame,
    level: str,
    cno2_weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    exposure_columns = [column for column in cno4_predictions.columns if column.startswith("observed_exposure_")]
    if level == "ocup1":
        df = cno4_predictions.copy()
        if cno2_weights is not None and not cno2_weights.empty:
            cno2 = _aggregate_equal(df, "CNO2", exposure_columns)
            weighted = cno2.merge(cno2_weights[["CNO2", "employment_weight"]], on="CNO2", how="left")
            weighted["OCUP1"] = weighted["CNO2"].astype(str).str[:1]
            weighted["employment_weight"] = weighted["employment_weight"].fillna(0.0)
            if weighted["employment_weight"].sum() == 0:
                return _aggregate_equal(df, "OCUP1", exposure_columns)
            rows = []
            for ocup1, group in weighted.groupby("OCUP1", dropna=False):
                total = group["employment_weight"].sum()
                if total <= 0:
                    fallback = df[df["OCUP1"] == ocup1]
                    row = {
                        "OCUP1": ocup1,
                        "occupation_title": f"OCUP1 {ocup1}",
                        "cno4_count": int(fallback["CNO4"].nunique()),
                        "cno2_count": int(fallback["CNO2"].nunique()),
                        "aggregation_weight_source": "equal CNO4 weights; no matching INE EPA CNO2 weight",
                    }
                    for column in exposure_columns:
                        row[column] = float(fallback[column].mean())
                    rows.append(row)
                    continue
                row = {
                    "OCUP1": ocup1,
                    "occupation_title": f"OCUP1 {ocup1}",
                    "cno4_count": int(group["cno4_count"].sum()),
                    "cno2_count": int(group["CNO2"].nunique()),
                    "aggregation_weight_source": "INE EPA table 65134 CNO2 employment weights",
                }
                for column in exposure_columns:
                    row[column] = float((group[column] * group["employment_weight"]).sum() / total)
                rows.append(row)
            return pd.DataFrame(rows).sort_values("OCUP1").reset_index(drop=True)
        return _aggregate_equal(df, "OCUP1", exposure_columns)
    if level == "cno2":
        return _aggregate_equal(cno4_predictions, "CNO2", exposure_columns)
    raise ValueError("level must be 'ocup1' or 'cno2'")


def load_epa_cno2_weights(config: PipelineConfig, refresh: bool = False) -> pd.DataFrame:
    path = ensure_reference_file(EPA_CNO2_WEIGHTS_URL, config.raw_dir / "ine" / "epa_65134_cno2_weights.csv", refresh)
    raw = pd.read_csv(path, sep=";", dtype="string")
    df = raw[
        (raw["Sexo"] == "Ambos sexos")
        & (raw["Unidad"] == "Valor absoluto")
    ].copy()
    latest = sorted(df["Periodo"].dropna().unique())[-1]
    df = df[df["Periodo"] == latest].copy()
    extracted = df["Ocupación CNO-11"].astype(str).str.extract(r"^(\d{2})\s+(.+)$")
    df["CNO2"] = extracted[0]
    df["cno2_title"] = extracted[1]
    df = df.dropna(subset=["CNO2"]).copy()
    df["employment_thousands"] = (
        df["Total"].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False).astype(float)
    )
    df["OCUP1"] = df["CNO2"].str[:1]
    totals = df.groupby("OCUP1")["employment_thousands"].transform("sum")
    df["employment_weight"] = df["employment_thousands"] / totals
    return df[["CNO2", "OCUP1", "cno2_title", "employment_thousands", "employment_weight", "Periodo"]].reset_index(drop=True)


def _aggregate_equal(df: pd.DataFrame, level_column: str, exposure_columns: list[str]) -> pd.DataFrame:
    group = df.groupby(level_column, dropna=False)
    out = group[exposure_columns].mean().reset_index()
    out["cno4_count"] = group["CNO4"].nunique().to_numpy()
    if level_column == "CNO2":
        out["OCUP1"] = out[level_column].astype(str)
        out["CNO1"] = out[level_column].astype(str).str[:1]
    elif level_column != "OCUP1":
        out["OCUP1"] = out[level_column].astype(str).str[:1]
    out["occupation_title"] = level_column + " " + out[level_column].astype(str)
    out["aggregation_weight_source"] = "equal CNO4 weights"
    sort_column = "OCUP1" if level_column == "OCUP1" else level_column
    return out.sort_values(sort_column).reset_index(drop=True)


def _starts_definition(line: str) -> bool:
    return bool(re.match(r"^(El|La|Los|Las|Este|Esta|Estos|Estas|En este|Se incluyen|Nota:)", line))


def _clean_bullets(lines: list[str]) -> list[str]:
    return [normalize_space(line.removeprefix("-").strip()) for line in lines if normalize_space(line)]


def _limit_text(text: str, limit: int) -> str:
    text = normalize_space(text)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,;") + "."


def _repair_pdf_text(text: str) -> str:
    text = normalize_space(text)
    text = text.replace("ma rco", "marco").replace("provi ncial", "provincial")
    text = re.sub(r"(\w)- (\w)", r"\1\2", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return normalize_space(text)
