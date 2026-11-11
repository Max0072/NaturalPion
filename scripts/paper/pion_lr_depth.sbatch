#!/bin/bash
#SBATCH -A p330
#SBATCH -J pionlrd
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00                                        # per row: pass -t (73242 steps ~10 h)
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Table 11 (Pion's lr) at one decay depth: the rows at lr != 1e-3 repeated from their own manifests with only
# lr_min_ratio = 0.01 added (they ran with the absolute lr_min 1e-5: depth 2% at 5e-4, 0.5% at 2e-3, 0.25% at 4e-3;
# the 1e-3 rows already have 1%). Output under runs/pion-lrdepth/<row>/.
#   sbatch -x rtx6003,rtx6004 --array=0-4 scripts/paper/pion_lr_depth.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"; DATA="${DATA:-$DATA_p330/c4}"
# rows 0-4: Table 11 at lr != 1e-3, from their own manifests (lr as recorded); rows 5-8 (2026-11-10): Pion at lr 2e-3
# at the budgets of Table 4, from the lr 1e-3 runs' manifests with lr overridden, so that Pion's lr is chosen per
# budget from {1e-3, 2e-3} as Natural Pion's scale is
ROWS=("pion3840-lr5e-4|stepnormlong/pion3840-lr5e-4/*|" "pion3840-lr2e-3|stepnormlong/pion3840-lr2e-3/*|"
      "pion3840-lr4e-3|stepnormlong/pion3840-lr4e-3/*|" "pion19200-lr2e-3|stepnormlong/pion19200-lr2e-3/*|"
      "pion19200-lr4e-3|stepnormlong/pion19200-lr4e-3/*|"
      "pion15000-lr2e-3|ladder15k/pion-lr0.001-s0-f67fdad3e9|2e-3" "pion30000-lr2e-3|stepnormlong/pion30000-lr1e-3/*|2e-3"
      "pion45000-lr2e-3|stepnormlong/pion45000-lr1e-3/*|2e-3" "pion73242-lr2e-3|stepnormlong/pion73242-lr1e-3/*|2e-3"
      "pion19200-lr2e-3|stepnormlong/pion19200-lr1e-3/pion-lr0.001-s1-*|2e-3"
      "pion19200-lr2e-3|stepnormlong/pion19200-lr1e-3/pion-lr0.001-s2-*|2e-3"
      "pion19200-lr2e-3|stepnormlong/pion19200-lr1e-3/pion-lr0.001-s3-*|2e-3")
# rows 9-11 (2026-11-10): seeds 1-3 of lr 2e-3 at 19 200 steps (from the lr 1e-3 seed runs), so that the two lrs are
# compared on four seeds each at the one budget where 2e-3 won on seed 0 alone; written next to seed 0
IFS='|' read -r NAME SRC LROVR <<< "${ROWS[$SLURM_ARRAY_TASK_ID]}"
MAN=$(ls $REPO/runs-record/$SRC/manifest.json 2>/dev/null | head -1)
OUT="$DATA_p330/runs/pion-lrdepth/$NAME"
echo "row $SLURM_ARRAY_TASK_ID: $NAME from $MAN -> $OUT"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" python - "$MAN" "$OUT" "$DATA" "$LROVR" <<'PY'
import json, sys
from dataclasses import replace
from harness.config import RunConfig
from harness.train import train
man, out, data, lr = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
ref = RunConfig.from_manifest(json.load(open(man))["config"])
cfg = replace(ref, lr_min_ratio=0.01, out_dir=out, data_path=f"{data}/c4_train.bin", val_path=f"{data}/c4_val.bin")
if lr:
    cfg = replace(cfg, lr=float(lr))
print("differs from the reference in:", {k: (getattr(ref, k), getattr(cfg, k)) for k in ref.__dataclass_fields__
                                          if getattr(ref, k) != getattr(cfg, k)}, flush=True)
print("wrote", train(cfg, device="cuda"))
PY
