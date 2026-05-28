from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd


OCCUPATION_EXPOSURE_COLUMNS = [
    "observed_exposure_rf",
    "observed_exposure_ridge",
    "observed_exposure_cosine_weighted",
    "observed_exposure_cosine_nearest",
    "observed_exposure_ensemble",
]

INDUSTRY_EXPOSURE_COLUMNS = [
    "observed_exposure_cnae_rf",
    "observed_exposure_cnae_ridge",
    "observed_exposure_cnae_cosine_weighted",
    "observed_exposure_cnae_cosine_nearest",
    "observed_exposure_cnae_ensemble",
]


SCHEMA = """
CREATE TABLE IF NOT EXISTS occupation_exposure (
    ocup1 TEXT NOT NULL,
    occupation_title TEXT NOT NULL,
    occupation_title_en TEXT,
    embedding_model TEXT NOT NULL,
    translation_model TEXT,
    model_sha256 TEXT NOT NULL,
    observed_exposure REAL NOT NULL,
    observed_exposure_rf REAL,
    observed_exposure_ridge REAL,
    observed_exposure_cosine_weighted REAL,
    observed_exposure_cosine_nearest REAL,
    observed_exposure_ensemble REAL,
    generated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ocup1, embedding_model, model_sha256)
);

CREATE TABLE IF NOT EXISTS industry_quarter_exposure (
    cnae TEXT NOT NULL,
    quarter TEXT NOT NULL,
    observed_exposure_cnae REAL NOT NULL,
    total_weight REAL NOT NULL,
    covered_weight REAL NOT NULL,
    coverage_share REAL NOT NULL,
    occupation_count INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    translation_model TEXT,
    model_sha256 TEXT NOT NULL,
    observed_exposure_cnae_rf REAL,
    observed_exposure_cnae_ridge REAL,
    observed_exposure_cnae_cosine_weighted REAL,
    observed_exposure_cnae_cosine_nearest REAL,
    observed_exposure_cnae_ensemble REAL,
    generated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (cnae, quarter, embedding_model, model_sha256)
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(occupation_exposure)").fetchall()}
    if "occupation_title_en" not in columns:
        conn.execute("ALTER TABLE occupation_exposure ADD COLUMN occupation_title_en TEXT")
    if "translation_model" not in columns:
        conn.execute("ALTER TABLE occupation_exposure ADD COLUMN translation_model TEXT")
    for column in OCCUPATION_EXPOSURE_COLUMNS:
        if column not in columns:
            conn.execute(f"ALTER TABLE occupation_exposure ADD COLUMN {column} REAL")
    panel_columns = {row[1] for row in conn.execute("PRAGMA table_info(industry_quarter_exposure)").fetchall()}
    if "translation_model" not in panel_columns:
        conn.execute("ALTER TABLE industry_quarter_exposure ADD COLUMN translation_model TEXT")
    for column in INDUSTRY_EXPOSURE_COLUMNS:
        if column not in panel_columns:
            conn.execute(f"ALTER TABLE industry_quarter_exposure ADD COLUMN {column} REAL")
    conn.commit()
    return conn


def upsert_occupation_exposure(
    conn: sqlite3.Connection,
    predictions: pd.DataFrame,
    embedding_model: str,
    translation_model: str,
    model_sha256: str,
) -> None:
    extra_columns = [
        column
        for column in OCCUPATION_EXPOSURE_COLUMNS
        if column in predictions.columns
    ]
    rows = [
        (
            str(row["OCUP1"]),
            str(row["occupation_title"]),
            str(row.get("occupation_title_en", "")),
            embedding_model,
            translation_model,
            model_sha256,
            float(row["observed_exposure"]),
            *[float(row[column]) for column in extra_columns],
        )
        for _, row in predictions.iterrows()
    ]
    columns = [
        "ocup1",
        "occupation_title",
        "occupation_title_en",
        "embedding_model",
        "translation_model",
        "model_sha256",
        "observed_exposure",
        *extra_columns,
    ]
    placeholders = ", ".join(["?"] * len(columns))
    conn.executemany(
        f"""
        INSERT OR REPLACE INTO occupation_exposure
        ({", ".join(columns)})
        VALUES ({placeholders})
        """,
        rows,
    )
    conn.commit()


def upsert_industry_quarter_exposure(
    conn: sqlite3.Connection,
    panel: pd.DataFrame,
    embedding_model: str,
    translation_model: str,
    model_sha256: str,
) -> None:
    extra_columns = [column for column in INDUSTRY_EXPOSURE_COLUMNS if column in panel.columns]
    rows = [
        (
            str(row["cnae"]),
            str(row["quarter"]),
            float(row["observed_exposure_cnae"]),
            float(row["total_weight"]),
            float(row["covered_weight"]),
            float(row["coverage_share"]),
            int(row["occupation_count"]),
            embedding_model,
            translation_model,
            model_sha256,
            *[float(row[column]) for column in extra_columns],
        )
        for _, row in panel.iterrows()
    ]
    columns = [
        "cnae",
        "quarter",
        "observed_exposure_cnae",
        "total_weight",
        "covered_weight",
        "coverage_share",
        "occupation_count",
        "embedding_model",
        "translation_model",
        "model_sha256",
        *extra_columns,
    ]
    placeholders = ", ".join(["?"] * len(columns))
    conn.executemany(
        f"""
        INSERT OR REPLACE INTO industry_quarter_exposure
        ({", ".join(columns)})
        VALUES ({placeholders})
        """,
        rows,
    )
    conn.commit()
