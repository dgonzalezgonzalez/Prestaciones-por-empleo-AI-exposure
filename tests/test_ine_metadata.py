from unittest import TestCase
from pathlib import Path

import pandas as pd

from src.download_ine import read_ine_microdata
from src.ine_metadata import build_occupation_table, parse_occupation_mapping_from_excel


class IneMetadataTests(TestCase):
    def _tmp_path(self, name: str) -> Path:
        root = Path.cwd() / "test_artifacts"
        root.mkdir(parents=True, exist_ok=True)
        path = root / name
        if path.exists():
            path.unlink()
        return path

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

    def test_parse_occupation_mapping_from_excel_uses_tocup_block_only(self):
        path = self._tmp_path("metadata.xlsx")
        rows = pd.DataFrame(
            [
                ["OTHER", None, "OCUP1"],
                ["Código", "Descripción", None],
                ["1", "Not an occupation variable description", None],
                [None, None, None],
                ["TOCUP", None, "OCUP1 *** (2 veces más)"],
                ["Código", "Descripción", None],
                ["0", "Ocupaciones militares (códigos CNO-2011). Fuerzas armadas (códigos CNO-1994)", None],
                ["1", "Directores y gerentes (códigos CNO-2011)", None],
                [None, None, None],
            ]
        )
        with pd.ExcelWriter(path) as writer:
            rows.to_excel(writer, sheet_name="Tablas1", header=False, index=False)

        out = parse_occupation_mapping_from_excel(path)

        self.assertEqual(out["OCUP1"].tolist(), ["0", "1"])
        self.assertEqual(out.loc[0, "occupation_title"], "Ocupaciones militares. Fuerzas armadas")

    def test_read_ine_microdata_keeps_only_pipeline_columns(self):
        path = self._tmp_path("microdata.tab")
        path.write_text(
            '"OCUP1"\t"ACT1"\t"FACTOREL"\t"EXTRA"\n'
            '1\t5\t10.5\tignored\n',
            encoding="utf-8",
        )

        out = read_ine_microdata(path, "2020Q1")

        self.assertEqual(out.columns.tolist(), ["quarter", "OCUP1", "ACT1", "FACTOREL"])
        self.assertEqual(out.loc[0, "quarter"], "2020Q1")
