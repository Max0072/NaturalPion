#!/bin/bash
#SBATCH -A p330
#SBATCH -J adamwlong
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 16:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# AdamW's lr at full length: a full-length run repeated from its own manifest with only adamw_lr changed (8e-3 was
# tuned for Pion at 3000 and 15000 steps only). Row 0: Pion (runs-record/stepnormlong/pion73242-lr1e-3); row 1: the
# method (npv2/d1pc-73242, decay depth 1%). Both at depth 1% of every peak, as the runs they repeat.
#   sbatch -x rtx6003,rtx6004 --array=<rows> scripts/paper/adamw_long.sbatch      (ADAMW: the lr, default 4e-3)
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"; DATA="${DATA:-$DATA_p330/c4}"
ADAMW="${ADAMW:-4e-3}"
# row 2 (2026-11-10): Pion at its other rotation lr, 2e-3 (and depth 1%), so that Table 10b can take Pion's better lr
# at every AdamW lr
ROWS=(
  "pion|$(ls $REPO/runs-record/stepnormlong/pion73242-lr1e-3/*/manifest.json | head -1)|"
  "np|$(ls $DATA_p330/runs/npv2/d1pc-73242/*/manifest.json | head -1)|"
  "pion-lr2e-3|$(ls $REPO/runs-record/stepnormlong/pion73242-lr1e-3/*/manifest.json | head -1)|2e-3"
)
IFS='|' read -r NAME MAN LROVR <<< "${ROWS[$SLURM_ARRAY_TASK_ID]}"
[ -f "$MAN" ] || { echo "no manifest for $NAME"; exit 1; }
OUT="$DATA_p330/runs/adamwlong/$NAME-adamw$ADAMW"
echo "row $SLURM_ARRAY_TASK_ID: $NAME from $MAN, adamw_lr $ADAMW -> $OUT"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" python - "$MAN" "$ADAMW" "$OUT" "$DATA" "$LROVR" <<'PY'
import json, sys
from dataclasses import replace
from harness.config import RunConfig
from harness.train import train
man, adamw, out, data, lr = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
ref = RunConfig.from_manifest(json.load(open(man))["config"])
cfg = replace(ref, adamw_lr=adamw, out_dir=out, data_path=f"{data}/c4_train.bin", val_path=f"{data}/c4_val.bin")
if lr:                                    # the other rotation lr, at the same 1% decay depth as the rest of the paper
    cfg = replace(cfg, lr=float(lr), lr_min_ratio=0.01)
diff = {k: (getattr(ref, k), getattr(cfg, k)) for k in ref.__dataclass_fields__ if getattr(ref, k) != getattr(cfg, k)}
print("differs from the reference in:", diff, flush=True)
print("wrote", train(cfg, device="cuda"))
PY
