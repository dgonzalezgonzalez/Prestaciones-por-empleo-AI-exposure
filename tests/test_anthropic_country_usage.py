from pathlib import Path
from unittest import TestCase
from zipfile import ZipFile

from src.download_anthropic import ECON_INDEX_CLAUDE_AI_MEMBER, load_country_job_usage


class AnthropicCountryUsageTests(TestCase):
    def _tmp_zip(self) -> Path:
        root = Path.cwd() / "test_artifacts"
        root.mkdir(parents=True, exist_ok=True)
        path = root / "anthropic_release.zip"
        if path.exists():
            path.unlink()
        return path

    def test_load_country_job_usage_uses_latest_soc_major_group_rows(self):
        path = self._tmp_zip()
        csv = "\n".join(
            [
                "date_start,date_end,geo_id,geo_level,category_name,hierarchy_level,metric_id,value,node_name,node_external_id",
                "2026-04-01,2026-05-01,ESP,country,soc_occupation,1,pct,1.00,Management,11",
                "2026-04-01,2026-05-01,USA,country,soc_occupation,1,pct,2.00,Management,11",
                "2026-05-01,2026-06-01,ESP,country,soc_occupation,1,pct,5.91,Management,11",
                "2026-05-01,2026-06-01,USA,country,soc_occupation,1,pct,5.85,Management,11",
                "2026-05-01,2026-06-01,ESP,country,soc_occupation,1,pct,23.92,Computer and Mathematical,15",
                "2026-05-01,2026-06-01,USA,country,soc_occupation,1,pct,21.13,Computer and Mathematical,15",
                "2026-05-01,2026-06-01,ESP,country,soc_occupation,0,pct,99.00,Software Developers,15-1252",
                "2026-05-01,2026-06-01,ESP,country,request,2,pct,99.00,Writing,x",
            ]
        )
        with ZipFile(path, "w") as archive:
            archive.writestr(ECON_INDEX_CLAUDE_AI_MEMBER, csv)

        out = load_country_job_usage(path)

        self.assertEqual(out["soc_major_group"].tolist(), ["11", "15"])
        self.assertEqual(out.loc[0, "date_start"], "2026-05-01")
        self.assertAlmostEqual(out.loc[out["soc_major_group"] == "11", "spain_minus_us_pct"].iloc[0], 0.06)
        self.assertAlmostEqual(out.loc[out["soc_major_group"] == "15", "spain_minus_us_pct"].iloc[0], 2.79)
