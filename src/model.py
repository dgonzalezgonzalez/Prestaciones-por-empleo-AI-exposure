from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .utils import clean_occupation_title, file_sha256


EXPOSURE_COLUMNS = [
    "observed_exposure_rf",
    "observed_exposure_ridge",
    "observed_exposure_cosine_weighted",
    "observed_exposure_cosine_nearest",
    "observed_exposure_ensemble",
]


@dataclass
class ExposureModelBundle:
    rf_model: RandomForestRegressor
    ridge_model: object
    anthropic_x: np.ndarray
    anthropic_y: np.ndarray
    anthropic_titles: list[str]
    metrics: dict


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
) -> ExposureModelBundle:
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

    train_df = pd.DataFrame(rows).reset_index(drop=True)
    y = pd.to_numeric(train_df["observed_exposure"], errors="coerce")
    keep = y.notna()
    x = vectors_to_matrix([vector for vector, ok in zip(vectors, keep) if ok])
    anthropic_titles = [
        clean_occupation_title(title)
        for title, ok in zip(train_df["title"].astype(str).tolist(), keep)
        if ok
    ]
    y = y[keep].to_numpy(dtype=float)

    rf_model = _new_rf(random_seed, n_estimators)
    ridge_model = _new_ridge()
    metrics: dict[str, object] = {
        "n_rows": int(len(y)),
        "embedding_dim": int(x.shape[1]),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
        "target_min": float(np.min(y)),
        "target_max": float(np.max(y)),
        "final_fit_rows": int(len(y)),
        "final_fit_uses_all_rows": True,
    }
    if len(y) >= 10:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=random_seed
        )
        diagnostic_rf = _new_rf(random_seed, n_estimators)
        diagnostic_rf.fit(x_train, y_train)
        pred = diagnostic_rf.predict(x_test)
        metrics["holdout"] = _metric_block(y_test, pred)
    else:
        metrics["holdout"] = {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    if len(y) >= 5:
        metrics["cross_validation"] = _cross_validate_methods(x, y, random_seed, n_estimators)

    rf_model.fit(x, y)
    ridge_model.fit(x, y)
    bundle = ExposureModelBundle(
        rf_model=rf_model,
        ridge_model=ridge_model,
        anthropic_x=x,
        anthropic_y=y,
        anthropic_titles=anthropic_titles,
        metrics=metrics,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metrics["model_sha256"] = file_sha256(model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    return bundle


def _new_rf(random_seed: int, n_estimators: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_seed,
        min_samples_leaf=2,
        n_jobs=1,
    )


def _new_ridge():
    return make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-3, 3, 13)),
    )


def _metric_block(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": float(r2_score(y_true, pred)) if len(y_true) > 1 else float("nan"),
    }


def _cross_validate_methods(
    x: np.ndarray,
    y: np.ndarray,
    random_seed: int,
    n_estimators: int,
) -> dict[str, dict[str, float]]:
    splits = min(5, len(y))
    kfold = KFold(n_splits=splits, shuffle=True, random_state=random_seed)
    predictions = {
        "random_forest": np.empty_like(y, dtype=float),
        "ridge": np.empty_like(y, dtype=float),
        "cosine_weighted": np.empty_like(y, dtype=float),
        "cosine_nearest": np.empty_like(y, dtype=float),
        "global_mean": np.empty_like(y, dtype=float),
    }
    for train_idx, test_idx in kfold.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train = y[train_idx]
        rf_model = _new_rf(random_seed, n_estimators)
        rf_model.fit(x_train, y_train)
        ridge_model = _new_ridge()
        ridge_model.fit(x_train, y_train)
        predictions["random_forest"][test_idx] = rf_model.predict(x_test)
        predictions["ridge"][test_idx] = ridge_model.predict(x_test)
        predictions["cosine_weighted"][test_idx] = _cosine_weighted_predict(x_train, y_train, x_test)
        predictions["cosine_nearest"][test_idx] = _cosine_nearest_predict(x_train, y_train, x_test)
        predictions["global_mean"][test_idx] = float(np.mean(y_train))

    return {name: _metric_block(y, pred) for name, pred in predictions.items()}


def _cosine_similarity(train_x: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    train_norm = np.linalg.norm(train_x, axis=1)
    target_norm = np.linalg.norm(target_x, axis=1)
    denom = np.outer(target_norm, train_norm)
    sims = target_x @ train_x.T
    return np.divide(sims, denom, out=np.zeros_like(sims, dtype=float), where=denom != 0)


def _cosine_weighted_predict(train_x: np.ndarray, y: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    sims = np.maximum(_cosine_similarity(train_x, target_x), 0.0)
    totals = sims.sum(axis=1)
    fallback = float(np.mean(y))
    return np.divide(sims @ y, totals, out=np.full(len(target_x), fallback), where=totals != 0)


def _cosine_nearest_predict(train_x: np.ndarray, y: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    sims = _cosine_similarity(train_x, target_x)
    return y[np.argmax(sims, axis=1)]


def load_model(path: Path) -> ExposureModelBundle:
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found: {path}")
    return joblib.load(path)


def predict_occupation_exposure(
    occupation_df: pd.DataFrame,
    embedding_lookup: dict[str, list[float]],
    model: ExposureModelBundle | RandomForestRegressor,
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
    if isinstance(model, ExposureModelBundle):
        rf_pred = model.rf_model.predict(x)
        ridge_pred = model.ridge_model.predict(x)
        cosine_weighted = _cosine_weighted_predict(model.anthropic_x, model.anthropic_y, x)
        cosine_nearest = _cosine_nearest_predict(model.anthropic_x, model.anthropic_y, x)
        out["observed_exposure_rf"] = rf_pred
        out["observed_exposure_ridge"] = ridge_pred
        out["observed_exposure_cosine_weighted"] = cosine_weighted
        out["observed_exposure_cosine_nearest"] = cosine_nearest
        out["observed_exposure_ensemble"] = np.mean(
            np.vstack([rf_pred, ridge_pred, cosine_weighted, cosine_nearest]),
            axis=0,
        )
    else:
        out["observed_exposure_rf"] = model.predict(x)
    return out
