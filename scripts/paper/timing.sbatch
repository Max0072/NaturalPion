#!/bin/bash
#SBATCH -A p330
#SBATCH -J timing
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Every step time of the paper from one job on one GPU, one configuration after another (Pion first and last, to see
# the node is steady): 600 steps each of the 3000-step schedule, no evaluation, a log row every 100 steps; seconds per
# step = (wall at step 599 - wall at step 100) / 499. Run last, once the method's form is settled.
#   sbatch -x rtx6004 scripts/paper/timing.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"; DATA="${DATA:-$DATA_p330/c4}"
OUTROOT="$DATA_p330/runs/timing"
F=natural-pion-fast
# the method's scale at 3000 steps: the best point of the k-series grid (scripts/analysis/kseries.py), unless NP_LR is set
NP_LR="${NP_LR:-$(cd "$REPO" && python3 -c 'import sys,os; sys.path.insert(0,"scripts/analysis"); import kseries; from pathlib import Path; t=kseries.scale(Path(os.environ["DATA_p330"],"runs","npv2"),3000); print(t or "")')}"
[ -n "$NP_LR" ] || { echo "the k-m3k grid is not finished"; exit 1; }
echo "method scale $NP_LR"
OFF='"stats_window":null,"beta_a":0.95,"beta_d":0.5,"share_inputs":false,"batch_sigma":false,"share_root":false'
RMS='"length":"rms"'
CONFIGS=(
  "pion-a|pion|1e-3|||"
  "alg1|$F|3e-4|cayley|fp32|{\"t_fac\":1,$OFF,$RMS}"
  "tf25|$F|$NP_LR|cayley|fp32|{\"t_fac\":25,$OFF,$RMS}"
  "windows|$F|$NP_LR|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,$RMS}"
  "batched|$F|$NP_LR|cayley|fp32|{\"share_inputs\":false,\"batch_sigma\":false,$RMS}"
  "fast|$F|$NP_LR|||{$RMS}"
  "final|$F|$NP_LR|||{$RMS,\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}"
  "pion-b|pion|1e-3|||"
)
for row in "${CONFIGS[@]}"; do
  IFS='|' read -r TAG OPT LR RET STATS KW <<< "$row"
  ARGS=(--optimizer "$OPT" --lr "$LR")
  [ -n "$RET" ] && ARGS+=(--np-retraction "$RET")
  [ -n "$STATS" ] && ARGS+=(--np-stats-dtype "$STATS")
  [ -n "$KW" ] && ARGS+=(--np-kwargs "$KW")
  [ "$OPT" = pion ] || ARGS+=(--ngd-beta1 0.9 --ngd-eps 1e-3)
  echo "== $TAG ${ARGS[*]}"
  apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
      python -m harness.run "${ARGS[@]}" --train-steps 3000 --max-steps 600 --no-resume --batch-sequences 512 \
        --micro-batch 256 --rho-micro 128 --precision bf16 --seed 0 --device cuda --log-every 100 --eval-every 100000 \
        --adamw-lr 8e-3 --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUTROOT/$TAG" \
      > "logs/timing-$SLURM_JOB_ID-$TAG.log" 2>&1 || echo "   failed: see logs/timing-$SLURM_JOB_ID-$TAG.log"
done
