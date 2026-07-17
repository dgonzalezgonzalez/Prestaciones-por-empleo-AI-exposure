import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.build_paper_tables import build_tables


ROOT = Path(__file__).resolve().parents[1]


class PaperTableTests(TestCase):
    def test_rendered_tables_keep_paper_values_and_requested_headers(self):
        with TemporaryDirectory(dir=ROOT) as tmp:
            outputs = build_tables(ROOT / "docs" / "paper_table_values.json", Path(tmp))
            summary = outputs["summary_statistics.tex"].read_text(encoding="utf-8-sig")
            twfe = outputs["twfe_results.tex"].read_text(encoding="utf-8-sig")

        self.assertIn("Mean & SD & Min & Max & Obs.", summary)
        self.assertIn("5,693 & 25,338 & 0 & 467,292", summary)
        self.assertIn("2,776 & 11,081 & 0 & 216,727", summary)
        self.assertIn("6.718 & 1.944 & 0.000 & 13.055", summary)
        self.assertIn("0.089", twfe)
        self.assertIn("(0.021)", twfe)
        self.assertIn("-0.003", twfe)

    def test_snapshot_has_all_current_result_rows(self):
        data = json.loads((ROOT / "docs" / "paper_table_values.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(len(data["continuous_did"]), 5)
        self.assertEqual(len(data["sdid"]), 3)
        self.assertEqual(len(data["twfe"]), 8)

