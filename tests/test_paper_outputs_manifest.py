from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "paper_outputs_manifest.json"


class PaperOutputsManifestTests(unittest.TestCase):
    def test_manifest_is_exact_and_unique(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["figures"]), 15)
        self.assertEqual(len(manifest["tables"]), 2)
        self.assertEqual(len(manifest["inline_tables"]), 3)
        figure_names = [entry["file"] for entry in manifest["figures"]]
        table_names = [entry["file"] for entry in manifest["tables"]]
        labels = [entry["label"] for entry in manifest["inline_tables"]]
        self.assertEqual(len(figure_names), len(set(figure_names)))
        self.assertEqual(len(table_names), len(set(table_names)))
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(figure_names[0], "figure_anthropic_country_soc_major_group_spain_us_may2026.png")
        self.assertIn("summary_statistics.tex", table_names)
        self.assertEqual(set(labels), {"tab:contdid", "tab:sdid", "tab:twfe"})

    def test_retained_plot_generators_do_not_set_figure_titles(self) -> None:
        source_files = [
            ROOT / "scripts" / "build_claude_country_job_usage_figure.py",
            ROOT / "scripts" / "compare_epa_sepe_unemployment.py",
            ROOT / "scripts" / "run_ai_exposure_econometrics.py",
            ROOT / "scripts" / "run_contdid_analysis.R",
            ROOT / "scripts" / "run_sdid_estimates.do",
        ]
        title_patterns = [
            re.compile(r"\.set_title\s*\("),
            re.compile(r"\bggtitle\s*\("),
            re.compile(r"\btitle\s*\("),
        ]
        for path in source_files:
            text = path.read_text(encoding="utf-8-sig")
            for pattern in title_patterns:
                self.assertIsNone(pattern.search(text), f"Figure title found in {path}: {pattern.pattern}")


if __name__ == "__main__":
    unittest.main()
