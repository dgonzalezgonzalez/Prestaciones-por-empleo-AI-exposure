from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from scripts.compare_epa_sepe_unemployment import build_comparison, read_sepe_quarterly_unemployment


class UnemploymentSourceCheckTests(TestCase):
    def test_sepe_monthly_stock_is_quarterly_average(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sepe.csv"
            pd.DataFrame(
                {
                    "period": ["2025-01", "2025-02", "2025-03", "2025-03"],
                    "dimension": ["total", "total", "total", "age"],
                    "category": ["Total", "Total", "Total", "18-24"],
                    "gender": ["Total", "Total", "Total", "Total"],
                    "parados": [90, 120, 150, 999],
                }
            ).to_csv(path, index=False)

            out = read_sepe_quarterly_unemployment(path)

        self.assertEqual(out.loc[0, "quarter"], "2025Q1")
        self.assertEqual(out.loc[0, "sepe_months"], 3)
        self.assertAlmostEqual(out.loc[0, "sepe_registered_unemployed"], 120.0)

    def test_comparison_keeps_overlapping_quarters(self):
        epa = pd.DataFrame({"quarter": ["2025Q1", "2025Q2"], "epa_unemployed": [10, 12]})
        sepe = pd.DataFrame({"quarter": ["2025Q2", "2025Q3"], "sepe_registered_unemployed": [11, 13]})

        out = build_comparison(epa, sepe)

        self.assertEqual(out["quarter"].tolist(), ["2025Q2"])
        self.assertEqual(out.loc[0, "epa_unemployed"], 12)
