from __future__ import annotations

from pathlib import Path
import re
import zipfile

import pandas as pd

from .download_ine import EPA_SOURCE, IneSourceSpec
from .utils import clean_occupation_title, normalize_space


def parse_occupation_mapping_from_excel(
    path: Path,
    source_spec: IneSourceSpec = EPA_SOURCE,
) -> pd.DataFrame:
    """Parse occupation code labels from INE metadata workbooks."""
    path = find_first_excel_file(path)
    if source_spec.source == "census":
        return parse_code_mapping_from_excel(
            path,
            dictionary_name="T_CNO",
            variable_name=source_spec.occupation_column,
            code_column="OCUP1",
            label_column="occupation_title",
        )
    return parse_code_mapping_from_excel(
        path,
        dictionary_name="TOCUP",
        variable_name=source_spec.occupation_column,
        code_column="OCUP1",
        label_column="occupation_title",
    )


def find_first_excel_file(path: Path) -> Path:
    if path.suffix.lower() != ".zip":
        return path
    out_dir = path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        for member in members:
            target = (out_dir / member.filename).resolve()
            if not str(target).startswith(str(out_dir.resolve())):
                raise ValueError(f"Unsafe path in zip archive {path}: {member.filename}")
        excel_members = [
            member
            for member in members
            if not member.is_dir() and Path(member.filename).suffix.lower() in {".xlsx", ".xls"}
        ]
        if not excel_members:
            raise ValueError(f"No Excel file found inside {path}")
        preferred = [
            item
            for item in excel_members
            if Path(item.filename).suffix.lower() == ".xlsx"
        ]
        member = sorted(preferred or excel_members, key=lambda item: item.file_size)[0]
        target = out_dir / member.filename
        if not target.exists():
            archive.extract(member, out_dir)
        return target


def parse_code_mapping_from_excel(
    path: Path,
    dictionary_name: str,
    variable_name: str,
    code_column: str,
    label_column: str,
) -> pd.DataFrame:
    """Scan INE dictionary sheets for code-description blocks."""
    sheets = pd.read_excel(path, sheet_name=None, dtype="string", header=None)
    frames: list[pd.DataFrame] = []
    for _, sheet in sheets.items():
        rows = sheet.fillna("").astype(str)
        for idx, row in rows.iterrows():
            marker = " ".join(normalize_space(v) for v in row.tolist())
            if dictionary_name not in marker or variable_name not in marker:
                continue
            for _, value_row in rows.iloc[idx + 1 :].iterrows():
                code = normalize_space(value_row.iloc[0])
                desc = normalize_space(value_row.iloc[1]) if len(value_row) > 1 else ""
                if code.lower() in {"codigo", "código"}:
                    continue
                if not code and not desc:
                    break
                if not re.fullmatch(r"\d{1,2}", code) or not desc:
                    continue
                frames.append(
                    pd.DataFrame(
                        {code_column: [code], label_column: [clean_occupation_title(desc)]}
                    )
                )

    if not frames:
        raise ValueError(f"Could not parse {variable_name} mapping from {path}")
    mapping = pd.concat(frames, ignore_index=True)
    mapping = mapping.drop_duplicates(subset=[code_column], keep="first")
    return mapping.sort_values(code_column).reset_index(drop=True)


def build_occupation_table(
    microdata: pd.DataFrame,
    mapping: pd.DataFrame | None,
    allow_code_labels: bool = False,
) -> pd.DataFrame:
    occupations = (
        microdata[["OCUP1"]]
        .dropna()
        .drop_duplicates()
        .sort_values("OCUP1")
        .reset_index(drop=True)
    )
    if mapping is not None and not mapping.empty:
        out = occupations.merge(mapping, on="OCUP1", how="left")
    else:
        out = occupations.copy()
        out["occupation_title"] = pd.NA

    missing = out["occupation_title"].isna() | (out["occupation_title"].astype("string").str.strip() == "")
    if missing.any():
        if not allow_code_labels:
            missing_codes = out.loc[missing, "OCUP1"].head(10).tolist()
            raise ValueError(
                "Missing Spanish occupation labels for OCUP1 codes "
                f"{missing_codes}. Provide INE metadata or pass --allow-code-labels."
            )
        out.loc[missing, "occupation_title"] = "OCUP1 " + out.loc[missing, "OCUP1"].astype(str)

    out["occupation_title"] = out["occupation_title"].map(clean_occupation_title)
    return out
