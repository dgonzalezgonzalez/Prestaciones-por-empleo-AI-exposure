from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .utils import clean_occupation_title, file_sha256


def vectors_to_matrix(vectors: list[list[float]]) -> np.ndarray:
    if not vectors:
        raise ValueError("No embeddings supplied.")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError(f"Embedding dimension mismatch: {sorted(dimensions)}")
    return np.asarray(vectors, dtype=float)


def train_exposure_model(
    anthropic_df: pd.DataFrame,
    embedding_lookup: dict[str, list[float]],
    model_path: Path,
    metrics_path: Path,
    random_seed: int,
    n_estimators: int,
) -> RandomForestRegressor:
    rows = []
    vectors = []
    for _, row in anthropic_df.iterrows():
        title = clean_occupation_title(row["title"])
        vector = embedding_lookup.get(title)
        if vector is None:
            continue
        rows.append(row)
        vectors.append(vector)
    if not rows:
        raise ValueError("No Anthropic rows had embeddings.")

    train_df = pd.DataFrame(rows)
    y = pd.to_numeric(train_df["observed_exposure"], errors="coerce")
    keep = y.notna()
    x = vectors_to_matrix([vector for vector, ok in zip(vectors, keep) if ok])
    y = y[keep].to_numpy(dtype=float)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_seed,
        min_samples_leaf=2,
        n_jobs=1,
    )
    metrics: dict[str, float | int] = {"n_rows": int(len(y)), "embedding_dim": int(x.shape[1])}
    if len(y) >= 10:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=random_seed
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        metrics["test_mae"] = float(mean_absolute_error(y_test, pred))
        metrics["test_r2"] = float(r2_score(y_test, pred))
    else:
        model.fit(x, y)
        metrics["test_mae"] = float("nan")
        metrics["test_r2"] = float("nan")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics["model_sha256"] = file_sha256(model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model


def load_model(path: Path) -> RandomForestRegressor:
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found: {path}")
    return joblib.load(path)


def predict_occupation_exposure(
    occupation_df: pd.DataFrame,
    embedding_lookup: dict[str, list[float]],
    model: RandomForestRegressor,
) -> pd.DataFrame:
    vectors = []
    rows = []
    for _, row in occupation_df.iterrows():
        title_source = row["embedding_text"] if "embedding_text" in occupation_df.columns else row["occupation_title"]
        title = clean_occupation_title(title_source)
        vector = embedding_lookup.get(title)
        if vector is None:
            raise ValueError(f"Missing embedding for Spanish occupation title: {title}")
        rows.append(row)
        vectors.append(vector)
    x = vectors_to_matrix(vectors)
    out = pd.DataFrame(rows).copy()
    out["observed_exposure"] = model.predict(x)
    return out
