from unittest import TestCase
from pathlib import Path

import pandas as pd

from src.download_ine import CENSUS_SOURCE, read_ine_microdata
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

    def test_parse_census_occupation_mapping_from_excel_uses_cno_block(self):
        path = self._tmp_path("census_metadata.xlsx")
        rows = pd.DataFrame(
            [
                ["T_CNAE", None, "ACT89"],
                ["Código", "Descripción", None],
                ["11", "Wrong block", None],
                [None, None, None],
                ["T_CNO", None, "OCU63"],
                ["Código", "Descripción", None],
                ["11", "Miembros del poder ejecutivo", None],
                ["27", "Profesionales de las tecnologías de la información", None],
                ["XX", "No consta", None],
                [None, None, None],
            ]
        )
        with pd.ExcelWriter(path) as writer:
            rows.to_excel(writer, sheet_name="Tablas7", header=False, index=False)

        out = parse_occupation_mapping_from_excel(path, CENSUS_SOURCE)

        self.assertEqual(out["OCUP1"].tolist(), ["11", "27"])
        self.assertEqual(
            out.loc[out["OCUP1"] == "27", "occupation_title"].item(),
            "Profesionales de las tecnologías de la información",
        )

    def test_read_census_microdata_maps_two_digit_columns(self):
        path = self._tmp_path("census_microdata.csv")
        path.write_text(
            "OCU63;ACT89;EXTRA\n"
            "27;62;kept\n"
            "XX;62;bad occupation\n"
            "27;XX;bad industry\n",
            encoding="utf-8",
        )

        out = read_ine_microdata(path, "2021", CENSUS_SOURCE)

        self.assertEqual(
            out[["period", "quarter", "OCUP1", "ACT1"]].to_dict("records"),
            [{"period": "2021", "quarter": "2021", "OCUP1": "27", "ACT1": "62"}],
        )

    def test_read_census_microdata_falls_back_to_fixed_width_positions(self):
        path = self._tmp_path("census_microdata.txt")
        row = (" " * 101) + "27" + "62" + "rest\n"
        path.write_text(row, encoding="utf-8")

        out = read_ine_microdata(path, "2021", CENSUS_SOURCE)

        self.assertEqual(out.loc[0, "OCUP1"], "27")
        self.assertEqual(out.loc[0, "ACT1"], "62")
