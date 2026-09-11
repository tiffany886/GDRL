#!/usr/bin/env bash
# P0 one-shot: pytest gate + one sensitivity scope + run_log row.
#   bash disaster_va_dag_ho/scripts/p1_all.sh <scope> [note]
set -euo pipefail
PY=/root/miniconda3/envs/asr_env/bin/python
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
scope="${1:-grid-lite}"
notes="${2:-p0-run}"
t0=$(date +%s)
echo "[p1_all] scope=$scope cwd=$ROOT"
"$PY" -m pytest disaster_va_dag_ho/tests -q | tail -1
"$PY" -m disaster_va_dag_ho.scripts.run_sens_v2 --scope "$scope"
t1=$(date +%s)
outs=""
for f in disaster_va_dag_ho/results/v2_p1_sens_*; do [ -e "$f" ] && outs="$outs $f"; done
"$PY" -m disaster_va_dag_ho.scripts.run_log \
  --cmd "p1_all.sh scope=$scope" --out "$outs" --notes "$notes" --wall "$((t1-t0))"
echo "[p1_all] done wall=$((t1-t0))s"
