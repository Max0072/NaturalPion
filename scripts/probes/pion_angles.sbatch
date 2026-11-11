#!/bin/bash
#SBATCH -A p330
#SBATCH -J pionang
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Pion's rotation angle per step (scripts/probes/pion_angles.py): published Pion, 3840 steps, lr 1e-3, seed 0,
# the same flags as stepnormfinal/pion3840, whose val it should reproduce (the probe only reads).
#   sbatch -x rtx6004 scripts/probes/pion_angles.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"; DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/pionangles"; mkdir -p "$OUT"; rm -f "$OUT/pion_angles.jsonl"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env PION_ANGLES="$OUT/pion_angles.jsonl" --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/pion_angles.py --optimizer pion --lr 1e-3 --train-steps 3840 --batch-sequences 512 \
      --micro-batch 256 --rho-micro 128 --precision bf16 --seed 0 --device cuda --log-every 23 --eval-every 500 \
      --adamw-lr 8e-3 --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT" --no-resume
