from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from .utils import clean_occupation_title, normalize_space


def parse_occupation_mapping_from_excel(path: Path) -> pd.DataFrame:
    """Parse OCUP1 code labels from INE metadata workbook.

    INE layouts vary across periods, so this scans sheets for rows mentioning OCUP1
    or CNO and keeps rows that look like code-description pairs.
    """
    sheets = pd.read_excel(path, sheet_name=None, dtype="string", header=None)
    frames: list[pd.DataFrame] = []
    for _, sheet in sheets.items():
        rows = sheet.fillna("").astype(str)
        for idx, row in rows.iterrows():
            marker = " ".join(normalize_space(v) for v in row.tolist())
            if "TOCUP" not in marker or "OCUP1" not in marker:
                continue
            for _, value_row in rows.iloc[idx + 1 :].iterrows():
                code = normalize_space(value_row.iloc[0])
                desc = normalize_space(value_row.iloc[1]) if len(value_row) > 1 else ""
                if code.lower() == "código":
                    continue
                if not code and not desc:
                    break
                if not re.fullmatch(r"\d{1,2}", code) or not desc:
                    continue
                frames.append(
                    pd.DataFrame(
                        {"OCUP1": [code], "occupation_title": [clean_occupation_title(desc)]}
                    )
                )

    if not frames:
        raise ValueError(f"Could not parse OCUP1 occupation mapping from {path}")
    mapping = pd.concat(frames, ignore_index=True)
    mapping = mapping.drop_duplicates(subset=["OCUP1"], keep="first")
    return mapping.sort_values("OCUP1").reset_index(drop=True)


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
