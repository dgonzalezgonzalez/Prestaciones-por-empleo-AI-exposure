"""Run SEPE synthetic DiD analysis through Stata's sdid/sdid_event commands."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATA_PATHS = [
    Path(r"C:\Program Files\StataNow19\StataMP-64.exe"),
    Path(r"C:\Program Files\Stata19\StataMP-64.exe"),
    Path(r"C:\Program Files\Stata18\StataMP-64.exe"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stata SDID estimates for the SEPE CNO4 panel.")
    parser.add_argument("--stata-exe", default=None, help="Path to StataMP executable. Defaults to STATA_EXE or PATH.")
    parser.add_argument("--reps", type=int, default=100, help="Bootstrap replications for sdid and sdid_event.")
    return parser.parse_args()


def resolve_stata(stata_exe: str | None) -> Path:
    candidates = []
    if stata_exe:
        candidates.append(Path(stata_exe))
    if os.environ.get("STATA_EXE"):
        candidates.append(Path(os.environ["STATA_EXE"]))
    for name in ["StataMP-64.exe", "StataMP.exe", "StataSE-64.exe", "StataBE-64.exe"]:
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    candidates.extend(DEFAULT_STATA_PATHS)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError("StataMP not found. Pass --stata-exe or set STATA_EXE.")


def main() -> int:
    args = parse_args()
    stata = resolve_stata(args.stata_exe)
    do_file = ROOT / "scripts" / "run_sdid_estimates.do"
    env = os.environ.copy()
    env["PATH"] = f"{stata.parent}{os.pathsep}{env.get('PATH', '')}"
    command = [str(stata), "/e", "do", str(do_file), str(args.reps)]
    print(f"Running Stata SDID: {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    log_path = ROOT / "analysis" / "econometrics_outputs" / "sdid" / "stata_sdid.log"
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "\nr(" in log_text or "\nr;" in log_text:
            raise RuntimeError(f"Stata SDID failed; inspect {log_path}")

    expected = [
        ROOT / "analysis" / "econometrics_outputs" / "sdid" / "sdid_estimates.csv",
        ROOT / "analysis" / "econometrics_outputs" / "sdid" / "sdid_eventstudy_paths.csv",
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"Stata SDID run finished but missing outputs: {missing}")
    for scratch in [
        ROOT / "analysis" / "econometrics_outputs" / "sdid" / "sdid_estimates.dta",
        log_path,
        ROOT / "run_sdid_estimates.log",
    ]:
        if scratch.exists():
            scratch.unlink()
    print("Stata SDID outputs written to analysis/econometrics_outputs/sdid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
