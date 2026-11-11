# Job scripts behind the paper

Every number in `docs/paper/preprint.md` is computed by `scripts/analysis/paper_numbers.py` from the runs in
`runs-record/` (manifest and log of every run) and from the probe outputs in `runs-record/probes/`. Submit from the
repository root (`logs/` must exist).

| script | what it runs |
|---|---|
| `v2.sbatch` | every Natural Pion run of the paper's second version, one row per run (Sections 3-5: retractions, statistics precision, regularisation, true Fisher, Algorithm 1 and the speed-up chain, the 5.4 ablations, the budget curve, full length) |
| `pion.sbatch` | every Pion run the paper reads, repeated exactly from its manifest (`harness.run --from-manifest`) |
| `collect_npv2.sh` | copies the logs of `v2.sbatch` runs into `runs-record/npv2` and re-renders the paper |
| `pion_lr_depth.sbatch` | Pion at lr 5e-4, 2e-3 and 4e-3 at a decay depth of 1% (Table 9; Pion's lr per budget from {1e-3, 2e-3}) |
| `adamw_long.sbatch` | the AdamW lr at full length (Table 10b) |
| `timing.sbatch` | wall-clock step times on one card (Section 4.5) |

The method is `--optimizer natural-pion-fast --lr <peak step scale>` with the class defaults (see `ngd_pion/README.md`);
Algorithm 1 is the same class with `t_fac` 1, `stats_window` None and retraction "cayley" (v2.sbatch, the speed-up
chain). The `#SBATCH` lines and `$DATA_p330` paths are those of the Aphrodite cluster where the runs were made.
