"""P0 run ledger: append one CSV row per experiment invocation.

Usage (from repo root /code/docs/GDRL):
  python -m disaster_va_dag_ho.scripts.run_log \\
      --cmd "python -m disaster_va_dag_ho.scripts.run_sens_v2 --scope grid-lite" \\
      --out "results/v2_p1_sens_raw.csv results/v2_p1_sens_summary.csv" \\
      --notes grid-lite --wall 1800.0
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # /code/docs/GDRL
PROJ = ROOT / "disaster_va_dag_ho"
RESULTS = PROJ / "results"
MAIN_V2 = PROJ / "configs" / "main_v2.yaml"
FIELDS = ["ts", "wall_s", "argv", "git_rev", "cfg_sha", "out_files", "notes"]


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "n/a"


def cfg_sha() -> str:
    try:
        return hashlib.sha256(MAIN_V2.read_bytes()).hexdigest()[:12]
    except Exception:
        return "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--wall", type=float, default=0.0)
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / "run_log.csv"
    new = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "ts": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wall_s": f"{a.wall:.1f}",
            "argv": a.cmd,
            "git_rev": git_rev(),
            "cfg_sha": cfg_sha(),
            "out_files": a.out,
            "notes": a.notes,
        })
    print(f"run_log: {p} +1 row")


if __name__ == "__main__":
    main()
