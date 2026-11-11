#!/bin/bash
#SBATCH -A p330
#SBATCH -J pion
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Every Pion run the paper reads, repeated exactly from its own manifest (published Pion: lr 1e-3 unless the row is
# part of the learning-rate check, AdamW 8e-3, Q per head). One row per run: the manifest under runs-record/.
# These runs exist; this script is how to reproduce them, replacing the launchers they were first started from
# (step_norm_*.sbatch, adamw2d, seeds_head, ladder15k; removed 2026-11-05, in git history).
#   sbatch -x rtx6004 -t <limit> --array=<rows> scripts/paper/pion.sbatch     (OUTROOT: where to write)
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"; DATA="${DATA:-$DATA_p330/c4}"
OUTROOT="${OUTROOT:-$DATA_p330/runs/pion-repro}"
RUNS=(
  runs-record/adamw2d/pion-lr0.001-s0-9e6e6d32f6/manifest.json   # 3000 steps, lr 0.001, seed 0
  runs-record/ladder15k/pion-lr0.001-s0-f67fdad3e9/manifest.json # 15000 steps, lr 0.001, seed 0
  runs-record/seedshead/pion-lr0.001-s1-c087b8ffb2/manifest.json # 3000 steps, lr 0.001, seed 1
  runs-record/seedshead/pion-lr0.001-s2-dc941e4784/manifest.json # 3000 steps, lr 0.001, seed 2
  runs-record/seedshead/pion-lr0.001-s3-98a6307212/manifest.json # 3000 steps, lr 0.001, seed 3
  runs-record/stepnormclock/pion-lr0.001-s0-dbe147d5df/manifest.json # 5300 steps, lr 0.001, seed 0
  runs-record/stepnormfinal/pion19200/pion-lr0.001-s0-c204471d77/manifest.json # 19200 steps, lr 0.001, seed 0
  runs-record/stepnormfinal/pion3840/pion-lr0.001-s0-99af31913e/manifest.json # 3840 steps, lr 0.001, seed 0
  runs-record/stepnormfinal/pion4500/pion-lr0.001-s0-a0620db963/manifest.json # 4500 steps, lr 0.001, seed 0
  runs-record/stepnorminitlong/pion15000-lr1e-3-pl3/pion-lr0.001-s0-3fbaf1efa6/manifest.json # 15000 steps, lr 0.001, seed 0
  runs-record/stepnorminitlong/pion19200-lr1e-3-pl3/pion-lr0.001-s0-ff82c233de/manifest.json # 19200 steps, lr 0.001, seed 0
  runs-record/stepnorminitlong/pion3000-lr1e-3-pl3/pion-lr0.001-s0-cf5ddb296e/manifest.json # 3000 steps, lr 0.001, seed 0
  runs-record/stepnorminitlong/pion3840-lr1e-3-pl3/pion-lr0.001-s0-9839cfc20f/manifest.json # 3840 steps, lr 0.001, seed 0
  runs-record/stepnorminitlong/pion73242-lr1e-3-pl3/pion-lr0.001-s0-2628ead405/manifest.json # 73242 steps, lr 0.001, seed 0
  runs-record/stepnormlong/pion19200-lr1e-3/pion-lr0.001-s1-d4613bde59/manifest.json # 19200 steps, lr 0.001, seed 1
  runs-record/stepnormlong/pion19200-lr1e-3/pion-lr0.001-s2-8bb6788604/manifest.json # 19200 steps, lr 0.001, seed 2
  runs-record/stepnormlong/pion19200-lr1e-3/pion-lr0.001-s3-c3faf1dd1a/manifest.json # 19200 steps, lr 0.001, seed 3
  runs-record/stepnormlong/pion19200-lr2e-3/pion-lr0.002-s0-ce20b741e5/manifest.json # 19200 steps, lr 0.002, seed 0
  runs-record/stepnormlong/pion19200-lr4e-3/pion-lr0.004-s0-e4a8fb1edf/manifest.json # 19200 steps, lr 0.004, seed 0
  runs-record/stepnormlong/pion30000-lr1e-3/pion-lr0.001-s0-41c97c70b8/manifest.json # 30000 steps, lr 0.001, seed 0
  runs-record/stepnormlong/pion3840-lr1e-3/pion-lr0.001-s1-11c2d4b7d3/manifest.json # 3840 steps, lr 0.001, seed 1
  runs-record/stepnormlong/pion3840-lr1e-3/pion-lr0.001-s2-cfc6c1938d/manifest.json # 3840 steps, lr 0.001, seed 2
  runs-record/stepnormlong/pion3840-lr1e-3/pion-lr0.001-s3-dfb14e874b/manifest.json # 3840 steps, lr 0.001, seed 3
  runs-record/stepnormlong/pion3840-lr2e-3/pion-lr0.002-s0-120c31c49a/manifest.json # 3840 steps, lr 0.002, seed 0
  runs-record/stepnormlong/pion3840-lr4e-3/pion-lr0.004-s0-7903a58ed3/manifest.json # 3840 steps, lr 0.004, seed 0
  runs-record/stepnormlong/pion3840-lr5e-4/pion-lr0.0005-s0-b2391a6fcb/manifest.json # 3840 steps, lr 0.0005, seed 0
  runs-record/stepnormlong/pion45000-lr1e-3/pion-lr0.001-s0-928f42db13/manifest.json # 45000 steps, lr 0.001, seed 0
  runs-record/stepnormlong/pion73242-lr1e-3/pion-lr0.001-s0-f3db83e6fb/manifest.json # 73242 steps, lr 0.001, seed 0
)
M="${RUNS[$SLURM_ARRAY_TASK_ID]}"
OUT="$OUTROOT/$(basename "$(dirname "$(dirname "$M")")")"; mkdir -p "$OUT"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run --from-manifest "$M" --device cuda --out-dir "$OUT" \
      --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin"
