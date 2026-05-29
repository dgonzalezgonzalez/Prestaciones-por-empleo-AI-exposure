from unittest import TestCase

import pandas as pd

from src.taxonomy import aggregate_cno4_predictions, build_cno4_structured_text


class TaxonomyTests(TestCase):
    def test_build_cno4_structured_text_keeps_sections(self):
        text = build_cno4_structured_text(
            "1211",
            "Directores financieros",
            [
                "Los directores financieros planifican y coordinan actividades financieras.",
                "Entre sus tareas se incluyen:",
                "- evaluar la situación financiera;",
                "- preparar presupuestos;",
                "Ejemplos de ocupaciones incluidas en este grupo primario",
                "- Directores financieros",
                "Ocupaciones afines no incluidas en este grupo primario",
                "- Especialistas en contabilidad, 2611",
            ],
        )

        self.assertIn("CNO occupation: 1211 Directores financieros.", text)
        self.assertIn("Typical tasks: evaluar la situación financiera", text)
        self.assertIn("Examples included: Directores financieros", text)
        self.assertIn("Related or excluded occupations: Especialistas", text)

    def test_aggregate_cno4_predictions_uses_cno2_weights(self):
        predictions = pd.DataFrame(
            {
                "CNO4": ["1111", "1112", "1211"],
                "CNO2": ["11", "11", "12"],
                "OCUP1": ["1", "1", "1"],
                "observed_exposure_cosine_weighted": [0.1, 0.3, 0.9],
                "observed_exposure_cosine_nearest": [0.2, 0.4, 0.8],
            }
        )
        weights = pd.DataFrame({"CNO2": ["11", "12"], "employment_weight": [0.25, 0.75]})

        out = aggregate_cno4_predictions(predictions, "ocup1", weights)

        row = out.iloc[0]
        self.assertAlmostEqual(row["observed_exposure_cosine_weighted"], 0.25 * 0.2 + 0.75 * 0.9)
        self.assertAlmostEqual(row["observed_exposure_cosine_nearest"], 0.25 * 0.3 + 0.75 * 0.8)
        self.assertEqual(row["cno4_count"], 3)
