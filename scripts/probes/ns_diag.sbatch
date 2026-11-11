#!/bin/bash
#SBATCH -A p330
#SBATCH -J nsdiag
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# scripts/probes/ns_diag.py for ns, then cayley and trunc as controls, 200 steps each.
#   sbatch -x rtx6004 scripts/probes/ns_diag.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"; DATA="${DATA:-$DATA_p330/c4}"
for RET in ${RETS:-ns cayley trunc}; do
  OUT="$DATA_p330/runs/nsdiag/$RET"; rm -rf "$OUT"; mkdir -p "$OUT"
  apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" python scripts/probes/ns_diag.py "$RET" 200 "$DATA" "$OUT"
done
