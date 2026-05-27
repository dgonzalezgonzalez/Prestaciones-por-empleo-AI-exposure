from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS occupation_exposure (
    ocup1 TEXT NOT NULL,
    occupation_title TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    model_sha256 TEXT NOT NULL,
    observed_exposure REAL NOT NULL,
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
    model_sha256 TEXT NOT NULL,
    generated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (cnae, quarter, embedding_model, model_sha256)
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def upsert_occupation_exposure(
    conn: sqlite3.Connection,
    predictions: pd.DataFrame,
    embedding_model: str,
    model_sha256: str,
) -> None:
    rows = [
        (
            str(row["OCUP1"]),
            str(row["occupation_title"]),
            embedding_model,
            model_sha256,
            float(row["observed_exposure"]),
        )
        for _, row in predictions.iterrows()
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO occupation_exposure
        (ocup1, occupation_title, embedding_model, model_sha256, observed_exposure)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def upsert_industry_quarter_exposure(
    conn: sqlite3.Connection,
    panel: pd.DataFrame,
    embedding_model: str,
    model_sha256: str,
) -> None:
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
            model_sha256,
        )
        for _, row in panel.iterrows()
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO industry_quarter_exposure
        (cnae, quarter, observed_exposure_cnae, total_weight, covered_weight,
         coverage_share, occupation_count, embedding_model, model_sha256)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
