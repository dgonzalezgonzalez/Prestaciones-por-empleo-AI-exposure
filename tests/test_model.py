from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd

from src.download_anthropic import load_anthropic_job_exposure
from src.model import EXPOSURE_COLUMNS, predict_occupation_exposure, train_exposure_model, vectors_to_matrix


class ModelTests(TestCase):
    def _tmp_path(self, name: str) -> Path:
        root = Path.cwd() / "test_artifacts"
        root.mkdir(parents=True, exist_ok=True)
        path = root / name
        if path.exists():
            path.unlink()
        return path

    def test_load_anthropic_requires_observed_exposure(self):
        path = self._tmp_path("bad.csv")
        path.write_text("occ_code,title\n11-1011,Chief Executives\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "observed_exposure"):
            load_anthropic_job_exposure(path)

    def test_vectors_to_matrix_rejects_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            vectors_to_matrix([[1.0], [1.0, 2.0]])

    def test_train_save_predict_small_model(self):
        df = pd.DataFrame(
            {
                "occ_code": [str(i) for i in range(12)],
                "title": [f"Job {i}" for i in range(12)],
                "observed_exposure": np.linspace(0, 1, 12),
            }
        )
        embeddings = {f"Job {i}": [float(i), float(i % 3), 1.0] for i in range(12)}
        model = train_exposure_model(
            df,
            embeddings,
            self._tmp_path("rf.joblib"),
            metrics_path := self._tmp_path("metrics.json"),
            random_seed=1,
            n_estimators=10,
        )
        occupations = pd.DataFrame({"OCUP1": ["1"], "occupation_title": ["Job 1"]})
        pred = predict_occupation_exposure(occupations, embeddings, model)
        self.assertEqual(len(pred), 1)
        self.assertTrue(np.isfinite(pred.loc[0, "observed_exposure"]))
        self.assertTrue(set(EXPOSURE_COLUMNS).issubset(pred.columns))
        self.assertEqual(model.metrics["final_fit_uses_all_rows"], True)
        self.assertIn("cross_validation", metrics_path.read_text(encoding="utf-8"))

    def test_cosine_predictions_use_anthropic_neighbors(self):
        df = pd.DataFrame(
            {
                "occ_code": ["a", "b", "c", "d", "e"],
                "title": ["Low", "High", "Mid", "Other 1", "Other 2"],
                "observed_exposure": [0.0, 1.0, 0.5, 0.2, 0.8],
            }
        )
        embeddings = {
            "Low": [1.0, 0.0],
            "High": [0.0, 1.0],
            "Mid": [0.7, 0.7],
            "Other 1": [0.9, 0.1],
            "Other 2": [0.1, 0.9],
            "Spanish high": [0.0, 0.99],
        }
        model = train_exposure_model(
            df,
            embeddings,
            self._tmp_path("cosine.joblib"),
            self._tmp_path("cosine_metrics.json"),
            random_seed=1,
            n_estimators=10,
        )
        occupations = pd.DataFrame(
            {"OCUP1": ["1"], "occupation_title": ["Spanish high"], "embedding_text": ["Spanish high"]}
        )
        pred = predict_occupation_exposure(occupations, embeddings, model)
        self.assertAlmostEqual(pred.loc[0, "observed_exposure_cosine_nearest"], 1.0)
        self.assertTrue(0.0 <= pred.loc[0, "observed_exposure_cosine_weighted"] <= 1.0)

    def test_prediction_uses_embedding_text_when_present(self):
        class DummyModel:
            def predict(self, x):
                return np.array([0.5])

        occupations = pd.DataFrame(
            {
                "OCUP1": ["1"],
                "occupation_title": ["Directores y gerentes"],
                "embedding_text": ["Managers and executives"],
            }
        )
        pred = predict_occupation_exposure(
            occupations,
            {"Managers and executives": [1.0, 2.0]},
            DummyModel(),
        )
        self.assertAlmostEqual(pred.loc[0, "observed_exposure"], 0.5)
