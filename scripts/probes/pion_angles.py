"""Published Pion, with the rotation angle it takes recorded; the trajectory is untouched.

Pion's step on a block is `T(-c g)`, a truncated exponential of the scaled generator, so the largest rotation
angle of that block at that step is `c * max(||g_in||_2, ||g_out||_2)`. Every 23rd step of each parameter this
probe writes that angle for every rotated block to $PION_ANGLES (jsonl: step, name index, block, angle).
Only reads; the update is computed by the unchanged baseline method.

    PION_ANGLES=out.jsonl python scripts/probes/pion_angles.py --optimizer pion --lr 1e-3 ...   (harness flags)
"""
import json
import os

import torch

import ngd_pion.pion_baseline as pb
from harness.run import main as run_main
from ngd_pion.direction import generators
from ngd_pion.linalg import skew

PATH = os.environ["PION_ANGLES"]
EVERY = 23
_orig = pb.Pion._update_block
_index = {}


def _update_block(self, W, G, state, group, prefix, side):
    out = _orig(self, W, G, state, group, prefix, side)
    if state.get("step", 0) % EVERY == 0:
        # recompute the smoothed generators exactly as the baseline just used them, from its stored state
        key = id(state)
        _index.setdefault(key, len(_index))
        mi, mo = state.get(prefix + "m_in"), state.get(prefix + "m_out")
        if mi is None or mo is None:
            g_in, g_out = generators(W, G)
        else:
            g_in, g_out = mi, mo
        g_in, g_out = skew(g_in), skew(g_out)
        c = self._scale(W, g_in, g_out, group, side)
        ang = c * max(float(torch.linalg.matrix_norm(g_in.float(), 2)), float(torch.linalg.matrix_norm(g_out.float(), 2)))
        with open(PATH, "a") as f:
            f.write(json.dumps({"step": state.get("step", 0), "param": _index[key], "block": prefix, "angle": ang}) + "\n")
    return out


pb.Pion._update_block = _update_block

if __name__ == "__main__":
    run_main()
