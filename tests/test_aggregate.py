from unittest import TestCase

import pandas as pd

from src.aggregate import aggregate_industry_quarter_exposure


class AggregateTests(TestCase):
    def test_weighted_average_by_industry_quarter(self):
        micro = pd.DataFrame(
            {
                "quarter": ["2025Q1", "2025Q1", "2025Q1"],
                "ACT1": ["A", "A", "B"],
                "OCUP1": ["1", "2", "1"],
                "FACTOREL": [2, 1, 4],
            }
        )
        exposure = pd.DataFrame(
            {
                "OCUP1": ["1", "2"],
                "observed_exposure_rf": [0.2, 0.8],
                "observed_exposure_cosine_weighted": [0.3, 0.9],
            }
        )

        out = aggregate_industry_quarter_exposure(micro, exposure)

        a = out[out["cnae"] == "A"].iloc[0]
        self.assertAlmostEqual(a["observed_exposure_cnae_rf"], (2 * 0.2 + 1 * 0.8) / 3)
        self.assertAlmostEqual(a["observed_exposure_cnae_cosine_weighted"], (2 * 0.3 + 1 * 0.9) / 3)
        self.assertAlmostEqual(a["coverage_share"], 1.0)

    def test_missing_exposure_reports_partial_coverage(self):
        micro = pd.DataFrame(
            {"quarter": ["2025Q1", "2025Q1"], "ACT1": ["A", "A"], "OCUP1": ["1", "2"]}
        )
        exposure = pd.DataFrame({"OCUP1": ["1"], "observed_exposure_rf": [0.4]})

        out = aggregate_industry_quarter_exposure(micro, exposure)

        self.assertAlmostEqual(out.loc[0, "observed_exposure_cnae_rf"], 0.4)
        self.assertAlmostEqual(out.loc[0, "coverage_share"], 0.5)
