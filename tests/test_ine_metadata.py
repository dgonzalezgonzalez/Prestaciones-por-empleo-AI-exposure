from unittest import TestCase

import pandas as pd

from src.ine_metadata import build_occupation_table


class IneMetadataTests(TestCase):
    def test_build_occupation_table_uses_and_cleans_mapping(self):
        micro = pd.DataFrame({"OCUP1": ["1", "2", "1"], "ACT1": ["A", "A", "B"], "quarter": ["2025Q1"] * 3})
        mapping = pd.DataFrame(
            {
                "OCUP1": ["1", "2"],
                "occupation_title": ["Directores (códigos CNO-2011)", "Técnicos sanitarios"],
            }
        )

        out = build_occupation_table(micro, mapping)

        self.assertEqual(out.loc[out["OCUP1"] == "1", "occupation_title"].item(), "Directores")
        self.assertEqual(set(out["OCUP1"]), {"1", "2"})

    def test_build_occupation_table_requires_labels_by_default(self):
        micro = pd.DataFrame({"OCUP1": ["1"], "ACT1": ["A"], "quarter": ["2025Q1"]})
        with self.assertRaisesRegex(ValueError, "Missing Spanish occupation labels"):
            build_occupation_table(micro, None)
