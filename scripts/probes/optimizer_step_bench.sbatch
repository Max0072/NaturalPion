#!/bin/bash
#SBATCH -A p330
#SBATCH -J optbench
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# scripts/probes/optimizer_step_bench.py: s/step of the optimizer alone, Pion / Fast variants / Turbo.
#   sbatch -x rtx6004 scripts/probes/optimizer_step_bench.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
# the method's scale at 3000 steps: the best point of the k-series grid (scripts/analysis/kseries.py), unless NP_LR is set
export NP_LR="${NP_LR:-$(cd "$REPO" && python3 -c 'import sys,os; sys.path.insert(0,"scripts/analysis"); import kseries; from pathlib import Path; t=kseries.scale(Path(os.environ["DATA_p330"],"runs","npv2"),3000); print(t or "")')}"
[ -n "$NP_LR" ] || { echo "the k-m3k grid is not finished"; exit 1; }
apptainer exec --env PYTHONUNBUFFERED=1 --env NP_LR="$NP_LR" --env PYTHONPATH="$REPO" --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/optimizer_step_bench.py
