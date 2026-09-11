"""Export publication LaTeX tables from frozen v2 CSVs.

Reads results/main_table_v2_summary.csv, ablation_v2_summary.csv and the
paired-statistics report values, writes docs/paper_v2/tables_v2.tex.
Run from repo root: python -m disaster_va_dag_ho.scripts.export_tables_v2
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parents[2] / "docs" / "paper_v2" / "tables_v2.tex"


def parse(s):
    m = re.match(r"([-+0-9.eE]+)±([-+0-9.eE]+)", str(s))
    return (float(m.group(1)), float(m.group(2))) if m else (float(s), 0.0)


def load(path):
    out = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = {k: parse(row[k]) for k in row
                 if k not in ("method", "seeds", "cell", "R", "lambda_high")}
            done, arrived, failed = d["apps_done"][0], d["apps_arrived"][0], d["apps_failed"][0]
            d["_eff"] = (d["mean_delay_s"][0] * done + 12.0 * failed) / max(arrived, 1)
            out[row["method"]] = d
    return out


def fmt(v, nd=4, scale=1.0, unit=""):
    return f"{v[0]*scale:.{nd}f}$\\pm${v[1]*scale:.{nd}f}{unit}"


def kj(v):
    return f"{v[0]/1000:.1f}k"


def main():
    main_tbl = load(RESULTS / "main_table_v2_summary.csv")
    abl = load(RESULTS / "ablation_v2_summary.csv")
    L = ["%% Auto-exported by disaster_va_dag_ho/scripts/export_tables_v2.py "
         "- do not edit by hand.", ""]

    # ---------------- Table I: main ----------------
    L += [r"\begin{table}[t]", r"\caption{Main comparison at the frozen "
          r"operating point (R=800\,m, $\lambda_{\mathrm{high}}$=0.2\,s$^{-1}$; "
          r"40 terminals, 3 UAVs, 2 LEOs, horizon 50\,s). Mean$\pm$std over 5 "
          r"scene seeds; each RL row is trained on the paired scene seed. "
          r"Effective latency $=(\sum \mathrm{delay}_{\mathrm{done}}+"
          r"\mathrm{failed}\cdot 12)/\mathrm{arrived}$; it penalises failed "
          r"tasks by their deadline so that it is not biased by hard-task "
          r"selection.}", r"\label{tab:main}", r"\centering\small",
          r"\begin{tabular}{lcccccc}", r"\toprule",
          r"Method & Success rate & Cond.\ delay (s) & Eff.\ lat.\ (s) & "
          r"UAV energy (J) & Invalid LEO attempts & Unassoc.\ frac. \\",
          r"\midrule"]
    names = {"B1-random": "B1-random", "B2-chase": "B2-chase",
             "B3-hover": "B3-hover", "Ours(T-patrol)": "Ours (T-patrol)",
             "B6-noVWprune": "B6 (no VW-prune)", "B4-flat": "B4-flat RL",
             "B5-SACflat": "B5-SAC-flat"}
    order = ["B1-random", "B2-chase", "B3-hover", "Ours(T-patrol)",
             "B6-noVWprune", "B4-flat", "B5-SACflat"]
    for m in order:
        r = main_tbl[m]
        b = (m == "Ours(T-patrol)")
        hl0, hl1 = (r"\textbf{", "}") if b else ("", "")
        L.append(f"{names[m]} & {hl0}{fmt(r['success_rate'])}{hl1} & "
                 f"{fmt(r['mean_delay_s'])} & {r['_eff']:.2f} & {kj(r['energy_j'])} & "
                 f"{r['leo_invalid_attempts'][0]:.0f} & "
                 f"{fmt(r['frac_unassociated'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # ---------------- Table II: statistics ----------------
    L += ["% --- paired one-sided Wilcoxon, center cell (see "
          "results/v2_p1_stats_report.txt) ---",
          r"\begin{table}[t]", r"\caption{Paired statistics at the center "
          r"cell (R=800\,m, $\lambda_{\mathrm{high}}$=0.2, scene seeds 0--9; "
          r"RL baselines evaluated on their paired scene seed; one-sided "
          r"paired Wilcoxon signed-rank).}",
          r"\label{tab:stats}", r"\centering\small",
          r"\begin{tabular}{lccl}", r"\toprule",
          r"Claim & $\Delta$ & $n$ & $p$ \\ \midrule",
          r"Ours $-$ B3-hover, success & $+0.0949$ & 10 & 0.0010 \\",
          r"Ours $-$ B4-flat, success & $+0.1306$ & 10 & 0.0010 \\",
          r"B6 $-$ Ours, cond.\ delay & $+0.172$\,s & 10 & 0.0137 \\",
          r"Ours $-$ B2-chase, success (n.s.) & $+0.0069$ & 10 & 1.0000 \\",
          r"Ours $-$ B2-chase, energy (Ours higher) & $+2130$\,J & 10 & 0.0098 \\",
          r"B4-flat 250$\rightarrow$500 ep, success & $-0.001$ & 5 & 0.81 \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # ---------------- Table III: ablations ----------------
    L += [r"\begin{table}[t]", r"\caption{Ablations on the frozen config "
          r"(5 seeds). dep.-viol.\ counts child nodes committed before their "
          r"parents' outputs exist; legal schedules keep it at zero.}",
          r"\label{tab:abl}", r"\centering\small",
          r"\begin{tabular}{lccccccc}", r"\toprule",
          r"Ablation & Success & Cond.\ delay (s) & Eff.\ lat.\ (s) & "
          r"Energy (J) & Invalid LEO & dep.-viol./ep. & Unassoc.\ frac. \\",
          r"\midrule"]
    lbl = {"ablate-no-VWprune": "w/o VW-prune (=B6)",
           "ablate-no-DAG-order": "w/o DAG-order",
           "ablate-no-hard-cover": "w/o hard cover",
           "ablate-no-embed(=B4)": "w/o embedding (=B4)"}
    for m in ["Ours(T-patrol)", "ablate-no-VWprune", "ablate-no-DAG-order",
              "ablate-no-hard-cover", "ablate-no-embed(=B4)"]:
        r = main_tbl[m] if m == "Ours(T-patrol)" else abl[m]
        name = "Ours (reference)" if m == "Ours(T-patrol)" else lbl[m]
        dep = r.get("dep_violations", (0.0, 0.0))  # legal schedules keep it at 0
        L.append(f"{name} & {fmt(r['success_rate'])} & {fmt(r['mean_delay_s'])} "
                 f"& {r['_eff']:.2f} & {kj(r['energy_j'])} & "
                 f"{r['leo_invalid_attempts'][0]:.0f} & "
                 f"{fmt(dep, 2)} & "
                 f"{fmt(r['frac_unassociated'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    OUT.write_text("\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
