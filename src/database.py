from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd


OCCUPATION_EXPOSURE_COLUMNS = [
    "observed_exposure_rf",
    "observed_exposure_cosine_weighted",
    "observed_exposure_cosine_nearest",
]

INDUSTRY_EXPOSURE_COLUMNS = [
    "observed_exposure_cnae_rf",
    "observed_exposure_cnae_cosine_weighted",
    "observed_exposure_cnae_cosine_nearest",
]


SCHEMA = """
CREATE TABLE IF NOT EXISTS occupation_exposure (
    ocup1 TEXT NOT NULL,
    occupation_title TEXT NOT NULL,
    occupation_title_en TEXT,
    embedding_model TEXT NOT NULL,
    translation_model TEXT,
    model_sha256 TEXT NOT NULL,
    observed_exposure_rf REAL,
    observed_exposure_cosine_weighted REAL,
    observed_exposure_cosine_nearest REAL,
    generated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ocup1, embedding_model, model_sha256)
);

CREATE TABLE IF NOT EXISTS industry_quarter_exposure (
    cnae TEXT NOT NULL,
    industry_name TEXT,
    quarter TEXT NOT NULL,
    total_weight REAL NOT NULL,
    covered_weight REAL NOT NULL,
    coverage_share REAL NOT NULL,
    occupation_count INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    translation_model TEXT,
    model_sha256 TEXT NOT NULL,
    observed_exposure_cnae_rf REAL,
    observed_exposure_cnae_cosine_weighted REAL,
    observed_exposure_cnae_cosine_nearest REAL,
    generated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (cnae, quarter, embedding_model, model_sha256)
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    _relax_not_null_columns(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(occupation_exposure)").fetchall()}
    if "occupation_title_en" not in columns:
        conn.execute("ALTER TABLE occupation_exposure ADD COLUMN occupation_title_en TEXT")
    if "translation_model" not in columns:
        conn.execute("ALTER TABLE occupation_exposure ADD COLUMN translation_model TEXT")
    for column in OCCUPATION_EXPOSURE_COLUMNS:
        if column not in columns:
            conn.execute(f"ALTER TABLE occupation_exposure ADD COLUMN {column} REAL")
    _drop_legacy_column(conn, "occupation_exposure", "observed_exposure")
    _drop_legacy_column(conn, "occupation_exposure", "observed_exposure_ridge")
    _drop_legacy_column(conn, "occupation_exposure", "observed_exposure_ensemble")
    panel_columns = {row[1] for row in conn.execute("PRAGMA table_info(industry_quarter_exposure)").fetchall()}
    if "industry_name" not in panel_columns:
        conn.execute("ALTER TABLE industry_quarter_exposure ADD COLUMN industry_name TEXT")
    if "translation_model" not in panel_columns:
        conn.execute("ALTER TABLE industry_quarter_exposure ADD COLUMN translation_model TEXT")
    for column in INDUSTRY_EXPOSURE_COLUMNS:
        if column not in panel_columns:
            conn.execute(f"ALTER TABLE industry_quarter_exposure ADD COLUMN {column} REAL")
    _drop_legacy_column(conn, "industry_quarter_exposure", "observed_exposure_cnae")
    _drop_legacy_column(conn, "industry_quarter_exposure", "observed_exposure_cnae_ridge")
    _drop_legacy_column(conn, "industry_quarter_exposure", "observed_exposure_cnae_ensemble")
    conn.commit()
    return conn


def _relax_not_null_columns(conn: sqlite3.Connection) -> None:
    occupation_info = conn.execute("PRAGMA table_info(occupation_exposure)").fetchall()
    panel_info = conn.execute("PRAGMA table_info(industry_quarter_exposure)").fetchall()
    if any(row[1] == "observed_exposure_rf" and row[3] for row in occupation_info):
        _rebuild_table(conn, "occupation_exposure")
    if any(row[1] == "observed_exposure_cnae_rf" and row[3] for row in panel_info):
        _rebuild_table(conn, "industry_quarter_exposure")


def _rebuild_table(conn: sqlite3.Connection, table: str) -> None:
    backup = f"{table}_legacy_not_null"
    conn.execute(f"ALTER TABLE {table} RENAME TO {backup}")
    conn.executescript(SCHEMA)
    old_columns = [row[1] for row in conn.execute(f"PRAGMA table_info({backup})").fetchall()]
    new_columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    common = [column for column in old_columns if column in new_columns]
    conn.execute(
        f"""
        INSERT OR IGNORE INTO {table} ({", ".join(common)})
        SELECT {", ".join(common)} FROM {backup}
        """
    )
    conn.execute(f"DROP TABLE {backup}")


def _drop_legacy_column(conn: sqlite3.Connection, table: str, column: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column in columns:
        conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")


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
            str(row.get("industry_name", "")),
            str(row["quarter"]),
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
        "industry_name",
        "quarter",
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
