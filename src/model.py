from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from .utils import clean_occupation_title, file_sha256


EXPOSURE_COLUMNS = [
    "observed_exposure_rf",
    "observed_exposure_cosine_weighted",
    "observed_exposure_cosine_nearest",
]
VALID_METHODS = ("rf", "cosine_weighted", "cosine_nearest")


@dataclass
class ExposureModelBundle:
    rf_model: RandomForestRegressor | None
    anthropic_x: np.ndarray
    anthropic_y: np.ndarray
    anthropic_titles: list[str]
    anthropic_codes: list[str]
    metrics: dict


def normalize_methods(methods: list[str] | tuple[str, ...] | str) -> tuple[str, ...]:
    if isinstance(methods, str):
        values = [item.strip() for item in methods.split(",")]
    else:
        values = [str(item).strip() for item in methods]
    values = [item for item in values if item]
    unknown = sorted(set(values).difference(VALID_METHODS))
    if unknown:
        raise ValueError(f"Unknown exposure methods: {unknown}. Valid methods: {list(VALID_METHODS)}")
    if not values:
        raise ValueError("At least one exposure method is required.")
    return tuple(dict.fromkeys(values))


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
    methods: list[str] | tuple[str, ...] = VALID_METHODS,
) -> ExposureModelBundle:
    methods = normalize_methods(methods)
    rows = []
    vectors = []
    for _, row in anthropic_df.iterrows():
        text = row["embedding_text"] if "embedding_text" in anthropic_df.columns else row["title"]
        title = clean_occupation_title(text)
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
    anthropic_codes = [
        str(code)
        for code, ok in zip(train_df["occ_code"].astype(str).tolist(), keep)
        if ok
    ]
    y = y[keep].to_numpy(dtype=float)

    rf_model = _new_rf(random_seed, n_estimators) if "rf" in methods else None
    metrics: dict[str, object] = {
        "n_rows": int(len(y)),
        "embedding_dim": int(x.shape[1]),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
        "target_min": float(np.min(y)),
        "target_max": float(np.max(y)),
        "methods": list(methods),
    }
    if "rf" in methods:
        metrics["final_fit_rows"] = int(len(y))
        metrics["final_fit_uses_all_rows"] = True
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
        metrics["cross_validation"] = _cross_validate_methods(x, y, random_seed, n_estimators, methods)

    if rf_model is not None:
        rf_model.fit(x, y)
    bundle = ExposureModelBundle(
        rf_model=rf_model,
        anthropic_x=x,
        anthropic_y=y,
        anthropic_titles=anthropic_titles,
        anthropic_codes=anthropic_codes,
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
    methods: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    splits = min(5, len(y))
    kfold = KFold(n_splits=splits, shuffle=True, random_state=random_seed)
    predictions = {"global_mean": np.empty_like(y, dtype=float)}
    if "rf" in methods:
        predictions["random_forest"] = np.empty_like(y, dtype=float)
    if "cosine_weighted" in methods:
        predictions["cosine_weighted"] = np.empty_like(y, dtype=float)
    if "cosine_nearest" in methods:
        predictions["cosine_nearest"] = np.empty_like(y, dtype=float)

    for train_idx, test_idx in kfold.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train = y[train_idx]
        if "rf" in methods:
            rf_model = _new_rf(random_seed, n_estimators)
            rf_model.fit(x_train, y_train)
            predictions["random_forest"][test_idx] = rf_model.predict(x_test)
        if "cosine_weighted" in methods:
            predictions["cosine_weighted"][test_idx] = _cosine_assignment_weighted_predict(x_train, y_train, x_test)
        if "cosine_nearest" in methods:
            predictions["cosine_nearest"][test_idx] = _cosine_nearest_predict(x_train, y_train, x_test)
        predictions["global_mean"][test_idx] = float(np.mean(y_train))

    return {name: _metric_block(y, pred) for name, pred in predictions.items()}


def _cosine_similarity(source_x: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    source_norm = np.linalg.norm(source_x, axis=1)
    target_norm = np.linalg.norm(target_x, axis=1)
    denom = np.outer(target_norm, source_norm)
    sims = target_x @ source_x.T
    return np.divide(sims, denom, out=np.zeros_like(sims, dtype=float), where=denom != 0)


def _cosine_assignment_weighted_predict(source_x: np.ndarray, y: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    sims = _cosine_similarity(source_x, target_x)
    nearest_target = np.argmax(sims, axis=0)
    predictions = np.empty(len(target_x), dtype=float)
    nearest_source_fallback = _cosine_nearest_predict(source_x, y, target_x)
    for target_idx in range(len(target_x)):
        assigned = nearest_target == target_idx
        if not np.any(assigned):
            predictions[target_idx] = nearest_source_fallback[target_idx]
            continue
        weights = sims[target_idx, assigned]
        total_weight = weights.sum()
        if total_weight <= 0:
            predictions[target_idx] = nearest_source_fallback[target_idx]
            continue
        predictions[target_idx] = float(np.average(y[assigned], weights=weights))
    return predictions


def _cosine_nearest_predict(source_x: np.ndarray, y: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    sims = _cosine_similarity(source_x, target_x)
    return y[np.argmax(sims, axis=1)]


def load_model(path: Path) -> ExposureModelBundle:
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found: {path}")
    return joblib.load(path)


def predict_occupation_exposure(
    occupation_df: pd.DataFrame,
    embedding_lookup: dict[str, list[float]],
    model: ExposureModelBundle | RandomForestRegressor,
    methods: list[str] | tuple[str, ...] = VALID_METHODS,
) -> pd.DataFrame:
    methods = normalize_methods(methods)
    vectors = []
    rows = []
    for _, row in occupation_df.iterrows():
        title_source = row["embedding_text"] if "embedding_text" in occupation_df.columns else row["occupation_title"]
        title = clean_occupation_title(title_source)
        vector = embedding_lookup.get(title)
        if vector is None:
            raise ValueError(f"Missing Spanish occupation title embedding: {title}")
        rows.append(row)
        vectors.append(vector)
    x = vectors_to_matrix(vectors)
    out = pd.DataFrame(rows).copy()
    if isinstance(model, ExposureModelBundle):
        if "rf" in methods:
            if model.rf_model is None:
                raise ValueError("RF requested but model bundle does not contain a fitted Random Forest.")
            out["observed_exposure_rf"] = model.rf_model.predict(x)
        if "cosine_weighted" in methods:
            out["observed_exposure_cosine_weighted"] = _cosine_assignment_weighted_predict(model.anthropic_x, model.anthropic_y, x)
        if "cosine_nearest" in methods:
            out["observed_exposure_cosine_nearest"] = _cosine_nearest_predict(model.anthropic_x, model.anthropic_y, x)
    else:
        if "rf" not in methods:
            raise ValueError("Legacy sklearn model only supports the rf method.")
        out["observed_exposure_rf"] = model.predict(x)
    return out


def cosine_match_details(
    occupation_df: pd.DataFrame,
    embedding_lookup: dict[str, list[float]],
    model: ExposureModelBundle,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vectors = []
    rows = []
    for _, row in occupation_df.iterrows():
        title_source = row["embedding_text"] if "embedding_text" in occupation_df.columns else row["occupation_title"]
        title = clean_occupation_title(title_source)
        vector = embedding_lookup.get(title)
        if vector is None:
            raise ValueError(f"Missing Spanish title embedding: {title}")
        rows.append(row)
        vectors.append(vector)

    target_x = vectors_to_matrix(vectors)
    sims = _cosine_similarity(model.anthropic_x, target_x)
    nearest_target_by_source = np.argmax(sims, axis=0)
    weighted_rows = []
    for target_idx, row in enumerate(rows):
        assigned_idx = np.flatnonzero(nearest_target_by_source == target_idx)
        if len(assigned_idx) == 0:
            assigned_idx = np.array([int(np.argmax(sims[target_idx]))])
        for source_idx in assigned_idx:
            weighted_rows.append(_match_row(row, model, sims[target_idx, source_idx], int(source_idx), "cosine_weighted"))

    nearest_rows = []
    nearest_source_by_target = np.argmax(sims, axis=1)
    for target_idx, row in enumerate(rows):
        source_idx = int(nearest_source_by_target[target_idx])
        nearest_rows.append(_match_row(row, model, sims[target_idx, source_idx], source_idx, "cosine_nearest"))
    return pd.DataFrame(weighted_rows), pd.DataFrame(nearest_rows)


def _match_row(row: pd.Series, model: ExposureModelBundle, similarity: float, source_idx: int, method: str) -> dict[str, object]:
    out = {
        "method": method,
        "spanish_code": str(row.get("OCUP1", row.get("CNO4", ""))),
        "spanish_title": str(row.get("occupation_title", "")),
        "spanish_embedding_text": str(row.get("embedding_text", "")),
        "anthropic_occ_code": model.anthropic_codes[source_idx],
        "anthropic_title": model.anthropic_titles[source_idx],
        "anthropic_observed_exposure": float(model.anthropic_y[source_idx]),
        "cosine_similarity": float(similarity),
    }
    for key in ["CNO4", "CNO2", "OCUP1"]:
        if key in row:
            out[key] = str(row[key])
    return out
