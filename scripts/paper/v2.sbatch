#!/bin/bash
#SBATCH -A p330
#SBATCH -J npv2
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# NOTE (2026-11-11): these rows were run before NaturalPionFast's defaults became the method (git tag
# pre-defaults-2026-11-11); rows that do not set length/tf32/fisher_dtype/eps relied on the old defaults. To repeat a
# run exactly, use its manifest (python -m harness.run --from-manifest ...), which restores them, or that tag.
# Every run of the paper's second version (user, 2026-11-05): the query projection rotated per head (always), Natural
# Pion Fast in its final form unless a row varies something. One row per run:
#   TAG | OPTIMIZER | STEPS | LR (the step scale) | SEED | BETA1 | RETRACTION | STATS | NP_KWARGS | EXTRA
# An empty RETRACTION / STATS is the optimizer class's own default (Fast: ns_delta, bf16; natural-pion = Algorithm 1:
# cayley, fp32). Runs land in $DATA_p330/runs/npv2/<TAG>/; scripts/paper/collect_npv2.sh copies the logs to
# runs-record/npv2 and refreshes the paper. Rows 0-98 were first submitted from v2_rest.sbatch (output under
# runs/npv2x), rows 99-110 from v2_perhead.sbatch (runs/npv2/s<steps>); collect_npv2.sh merges both.
#   sbatch -x rtx6004 -t <limit> --array=<rows>%4 scripts/paper/v2.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"; DATA="${DATA:-$DATA_p330/c4}"
F=natural-pion-fast; A=natural-pion
ROWS=(
# 0-5  retraction (3.4, 4.8) and statistics precision (4.7); the ns_delta/bf16 reference is npv2/s3000 seed 0
"ret-cayley|$F|3000|0.2|0|0.9|cayley|||"
"ret-ns|$F|3000|0.2|0|0.9|ns|||"
"ret-trunc|$F|3000|0.2|0|0.9|trunc|||"
"ret-trunc-15k|$F|15000|0.05|0|0.9|trunc|||"
"stats-fp32|$F|3000|0.2|0|0.9||fp32||"
"form46|$F|3000|0.2|0|0.9|cayley|fp32|{\"batch_sigma\":false,\"share_inputs\":false}|"
# 6-13 regularisation (5.4)
"reg-eps1e-6|$F|3000|0.2|0|0.9||||--ngd-eps 1e-6"
"reg-eps1e-5|$F|3000|0.2|0|0.9||||--ngd-eps 1e-5"
"reg-eps1e-3|$F|3000|0.2|0|0.9||||--ngd-eps 1e-3"
"reg-eps1e-2|$F|3000|0.2|0|0.9||||--ngd-eps 1e-2"
"reg-damp1e-4|$F|3000|0.2|0|0.9||||--ngd-eps 1e-6 --np-damping 1e-4"
"reg-damp1e-3|$F|3000|0.2|0|0.9||||--ngd-eps 1e-6 --np-damping 1e-3"
"reg-damp1e-2|$F|3000|0.2|0|0.9||||--ngd-eps 1e-6 --np-damping 1e-2"
"reg-damp1e-1|$F|3000|0.2|0|0.9||||--ngd-eps 1e-6 --np-damping 1e-1"
# 14-18 true vs empirical Fisher (3.1 remark): sampled targets at three angles, empirical at the two others
"fisher-mc-0.1|$F|3000|0.1|0|0.9||||--ngd-fisher-mc-every 1"
"fisher-mc-0.2|$F|3000|0.2|0|0.9||||--ngd-fisher-mc-every 1"
"fisher-mc-0.4|$F|3000|0.4|0|0.9||||--ngd-fisher-mc-every 1"
"fisher-emp-0.1|$F|3000|0.1|0|0.9||||"
"fisher-emp-0.4|$F|3000|0.4|0|0.9||||"
# 19-24 Algorithm 1 (4.1, 4.2): momentum grid, no-momentum grid
"alg1-mom-0.05|$A|3000|0.05|0|0.9||||"
"alg1-mom-0.1|$A|3000|0.1|0|0.9||||"
"alg1-mom-0.2|$A|3000|0.2|0|0.9||||"
"alg1-nomom-0.2|$A|3000|0.2|0|0||||"
"alg1-nomom-0.4|$A|3000|0.4|0|0||||"
"alg1-nomom-0.8|$A|3000|0.8|0|0||||"
# 25   Algorithm 1 at 15 000 steps
"alg1-mom-15k|$A|15000|0.1|0|0.9||||"
# 26-31 amortised refresh (4.2): T_fac 25, without and with momentum
"tf25-nomom-0.4|$A|3000|0.4|0|0|||{\"t_fac\":25}|"
"tf25-nomom-0.8|$A|3000|0.8|0|0|||{\"t_fac\":25}|"
"tf25-nomom-1.6|$A|3000|1.6|0|0|||{\"t_fac\":25}|"
"tf25-mom-0.1|$A|3000|0.1|0|0.9|||{\"t_fac\":25}|"
"tf25-mom-0.2|$A|3000|0.2|0|0.9|||{\"t_fac\":25}|"
"tf25-mom-0.4|$A|3000|0.4|0|0.9|||{\"t_fac\":25}|"
# 32-35 T_fac 10, seeds 0-3
"tf10-s0|$A|3000|0.2|0|0.9|||{\"t_fac\":10}|"
"tf10-s1|$A|3000|0.2|1|0.9|||{\"t_fac\":10}|"
"tf10-s2|$A|3000|0.2|2|0.9|||{\"t_fac\":10}|"
"tf10-s3|$A|3000|0.2|3|0.9|||{\"t_fac\":10}|"
# 36-38 statistics windows (4.3)
"win-adlast6|$A|3000|0.2|0|0.9|||{\"t_fac\":25,\"stats_window\":6,\"beta_a\":0.5}|"
"win-betad09|$A|3000|0.2|0|0.9|||{\"t_fac\":25,\"beta_d\":0.9}|"
"win-betad095|$A|3000|0.2|0|0.9|||{\"t_fac\":25,\"beta_d\":0.95}|"
# 39-49 direction (5.4): X = F^-1 G against X = G, without and with momentum; angle grid (0.2 with momentum = npv2)
"dir-nat-nomom-0.1|$F|3000|0.1|0|0||||"
"dir-nat-nomom-0.2|$F|3000|0.2|0|0||||"
"dir-nat-nomom-0.4|$F|3000|0.4|0|0||||"
"dir-nat-mom-0.1|$F|3000|0.1|0|0.9||||"
"dir-nat-mom-0.4|$F|3000|0.4|0|0.9||||"
"dir-raw-nomom-0.1|$F|3000|0.1|0|0|||{\"direction\":\"raw\"}|"
"dir-raw-nomom-0.2|$F|3000|0.2|0|0|||{\"direction\":\"raw\"}|"
"dir-raw-nomom-0.4|$F|3000|0.4|0|0|||{\"direction\":\"raw\"}|"
"dir-raw-mom-0.1|$F|3000|0.1|0|0.9|||{\"direction\":\"raw\"}|"
"dir-raw-mom-0.2|$F|3000|0.2|0|0.9|||{\"direction\":\"raw\"}|"
"dir-raw-mom-0.4|$F|3000|0.4|0|0.9|||{\"direction\":\"raw\"}|"
# 50-74 length rules (5.4): five-point angle grid each, x2 apart, centred on the old optimum
"len-before-3e-6|$F|3000|3e-6|0|0.9|||{\"length\":\"before\"}|"
"len-before-1e-5|$F|3000|1e-5|0|0.9|||{\"length\":\"before\"}|"
"len-before-3e-5|$F|3000|3e-5|0|0.9|||{\"length\":\"before\"}|"
"len-before-1e-4|$F|3000|1e-4|0|0.9|||{\"length\":\"before\"}|"
"len-before-3e-4|$F|3000|3e-4|0|0.9|||{\"length\":\"before\"}|"
"len-rms-6e-4|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\"}|"
"len-rms-1.2e-3|$F|3000|1.2e-3|0|0.9|||{\"length\":\"rms\"}|"
"len-rms-2.4e-3|$F|3000|2.4e-3|0|0.9|||{\"length\":\"rms\"}|"
"len-rms-4.8e-3|$F|3000|4.8e-3|0|0.9|||{\"length\":\"rms\"}|"
"len-rms-9.6e-3|$F|3000|9.6e-3|0|0.9|||{\"length\":\"rms\"}|"
"len-fisher-2.4e-3|$F|3000|2.4e-3|0|0.9|||{\"length\":\"fisher\"}|"
"len-fisher-4.8e-3|$F|3000|4.8e-3|0|0.9|||{\"length\":\"fisher\"}|"
"len-fisher-9.6e-3|$F|3000|9.6e-3|0|0.9|||{\"length\":\"fisher\"}|"
"len-fisher-1.9e-2|$F|3000|1.9e-2|0|0.9|||{\"length\":\"fisher\"}|"
"len-fisher-3.8e-2|$F|3000|3.8e-2|0|0.9|||{\"length\":\"fisher\"}|"
"len-quad-1e-3|$F|3000|1e-3|0|0.9|||{\"length\":\"quad\"}|"
"len-quad-2e-3|$F|3000|2e-3|0|0.9|||{\"length\":\"quad\"}|"
"len-quad-4e-3|$F|3000|4e-3|0|0.9|||{\"length\":\"quad\"}|"
"len-quad-8e-3|$F|3000|8e-3|0|0.9|||{\"length\":\"quad\"}|"
"len-quad-1.6e-2|$F|3000|1.6e-2|0|0.9|||{\"length\":\"quad\"}|"
"len-fixed-5e-4|$F|3000|5e-4|0|0.9|||{\"length\":\"fixed\"}|"
"len-fixed-1e-3|$F|3000|1e-3|0|0.9|||{\"length\":\"fixed\"}|"
"len-fixed-2e-3|$F|3000|2e-3|0|0.9|||{\"length\":\"fixed\"}|"
"len-fixed-4e-3|$F|3000|4e-3|0|0.9|||{\"length\":\"fixed\"}|"
"len-fixed-8e-3|$F|3000|8e-3|0|0.9|||{\"length\":\"fixed\"}|"
# 75-80 each side by its own sigma; attention angle multiplier
"sides-each-0.1|$F|3000|0.1|0|0.9|||{\"sides\":\"each\"}|"
"sides-each-0.2|$F|3000|0.2|0|0.9|||{\"sides\":\"each\"}|"
"attn-x0.25|$F|3000|0.2|0|0.9|||{\"attn_mult\":0.25}|"
"attn-x0.5|$F|3000|0.2|0|0.9|||{\"attn_mult\":0.5}|"
"attn-x2|$F|3000|0.2|0|0.9|||{\"attn_mult\":2}|"
"attn-x4|$F|3000|0.2|0|0.9|||{\"attn_mult\":4}|"
# 81-91 schedule of the angle (AdamW keeps its cosine): best of three angles; warmup-stable-decay
"sched-constant-0.05|$F|3000|0.05|0|0.9||||--schedule constant"
"sched-constant-0.1|$F|3000|0.1|0|0.9||||--schedule constant"
"sched-constant-0.2|$F|3000|0.2|0|0.9||||--schedule constant"
"sched-linear-0.1|$F|3000|0.1|0|0.9||||--schedule linear"
"sched-linear-0.2|$F|3000|0.2|0|0.9||||--schedule linear"
"sched-linear-0.4|$F|3000|0.4|0|0.9||||--schedule linear"
"sched-warmcos-0.1|$F|3000|0.1|0|0.9||||--schedule warmcos"
"sched-warmcos-0.2|$F|3000|0.2|0|0.9||||--schedule warmcos"
"sched-warmcos-0.4|$F|3000|0.4|0|0.9||||--schedule warmcos"
"sched-wsd-0.2|$F|3000|0.2|0|0.9||||--schedule wsd --wsd-decay 0.2"
"sched-wsd-0.1|$F|3000|0.2|0|0.9||||--schedule wsd --wsd-decay 0.1"
# 92-93 every plane of X turned by the same angle (polar step)
"planes-equal-0.1|$F|3000|0.1|0|0.9|||{\"planes\":\"equal\"}|"
"planes-equal-0.2|$F|3000|0.2|0|0.9|||{\"planes\":\"equal\"}|"
# 94-96 statistics windows (4.3) on the T_fac = 25 form of the chain (statistics every step, EMA 0.95 / 0.5)
"win-dlast6|$F|3000|0.2|0|0.9|cayley|fp32|{\"t_fac\":25,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"d_window\":6}|"
"win-alast6|$F|3000|0.2|0|0.9|cayley|fp32|{\"t_fac\":25,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"a_window\":6}|"
"win-aevery2|$F|3000|0.2|0|0.9|cayley|fp32|{\"t_fac\":25,\"stats_window\":null,\"beta_a\":0.9025,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"a_every\":2}|"
# 97-98 profile of the optimizer step by part (Table 13): the exact form and its T_fac = 25 form, 400 steps
"prof-alg1|$A|3000|0.2|0|0.9||||--max-steps 400 --np-profile true"
"prof-tf25|$A|3000|0.2|0|0.9|||{\"t_fac\":25}|--max-steps 400 --np-profile true"
# 99-110 Section 4: 3000 and 15000 steps over seeds 0-3, the budget curve, full length
"s3000|$F|3000|0.2|0|0.9|||"
"s3000|$F|3000|0.2|1|0.9|||"
"s3000|$F|3000|0.2|2|0.9|||"
"s3000|$F|3000|0.2|3|0.9|||"
"s15000|$F|15000|0.05|0|0.9|||"
"s15000|$F|15000|0.05|1|0.9|||"
"s15000|$F|15000|0.05|2|0.9|||"
"s15000|$F|15000|0.05|3|0.9|||"
"s9000|$F|9000|0.05|0|0.9|||"
"s30000|$F|30000|0.05|0|0.9|||"
"s45000|$F|45000|0.05|0|0.9|||"
"s73242|$F|73242|0.05|0|0.9|||"
# 111 statistics precision at 15000 steps (4.7)
"stats-fp32-15k|$F|15000|0.05|0|0.9||fp32||"
# 112-123 the initial spectrum (Appendix D): power-law exponent alpha, flat, orthogonal; alpha 3 over seeds, 15000, full length
"init-pl2|$F|3000|0.2|0|0.9||||--init-pl-alpha 2"
"init-pl2.5|$F|3000|0.2|0|0.9||||--init-pl-alpha 2.5"
"init-pl3|$F|3000|0.2|0|0.9||||--init-pl-alpha 3"
"init-pl4|$F|3000|0.2|0|0.9||||--init-pl-alpha 4"
"init-pl5|$F|3000|0.2|0|0.9||||--init-pl-alpha 5"
"init-flat|$F|3000|0.2|0|0.9||||--init-pl-alpha inf"
"init-orth|$F|3000|0.2|0|0.9||||--init orthogonal"
"init-pl3|$F|3000|0.2|1|0.9||||--init-pl-alpha 3"
"init-pl3|$F|3000|0.2|2|0.9||||--init-pl-alpha 3"
"init-pl3|$F|3000|0.2|3|0.9||||--init-pl-alpha 3"
"init-pl3-15k|$F|15000|0.05|0|0.9||||--init-pl-alpha 3"
"init-pl3-73242|$F|73242|0.05|0|0.9||||--init-pl-alpha 3"
# 124-129 second wave: seeds 1-3 at the best angle of a grid (submit with --dependency=afterany:<grid job>)
"alg1-mom-seeds|$A|3000|best:alg1-mom|1|0.9||||"
"alg1-mom-seeds|$A|3000|best:alg1-mom|2|0.9||||"
"alg1-mom-seeds|$A|3000|best:alg1-mom|3|0.9||||"
"len-rms-seeds|$F|3000|best:len-rms|1|0.9|||{\"length\":\"rms\"}|"
"len-rms-seeds|$F|3000|best:len-rms|2|0.9|||{\"length\":\"rms\"}|"
"len-rms-seeds|$F|3000|best:len-rms|3|0.9|||{\"length\":\"rms\"}|"
# 130-134 grid edges found on 2026-11-05: angle 0.8 where 0.4 was best; smaller angles where X = G diverged
"dir-nat-nomom-0.8|$F|3000|0.8|0|0||||"
"dir-nat-mom-0.8|$F|3000|0.8|0|0.9||||"
"tf25-mom-0.8|$A|3000|0.8|0|0.9|||{\"t_fac\":25}|"
"dir-raw-nomom-0.05|$F|3000|0.05|0|0|||{\"direction\":\"raw\"}|"
"dir-raw-nomom-0.025|$F|3000|0.025|0|0|||{\"direction\":\"raw\"}|"
# 135-136 TF32 (scripts/probes/tf32_check.py): everything but the Cayley step, and everything
"tf32-upstream|$F|3000|0.2|0|0.9|||{\"tf32\":\"upstream\"}|"
"tf32-all|$F|3000|0.2|0|0.9|||{\"tf32\":\"all\"}|"
# 137 TF32 before the Cayley step at 15000 steps: the second point of Fast's acceptance rule (within 0.010)
"tf32-upstream-15k|$F|15000|0.05|0|0.9|||{\"tf32\":\"upstream\"}|"
# 138-140 the peak angle at 15000 steps in this setting (0.05 is row s15000)
"theta15k-0.025|$F|15000|0.025|0|0.9||||"
"theta15k-0.1|$F|15000|0.1|0|0.9||||"
"theta15k-0.2|$F|15000|0.2|0|0.9||||"
# 141-142 F^-1 products in bf16 (Fast's acceptance rule: within 0.010 at 3000 and 15000)
"fisherbf16|$F|3000|0.2|0|0.9|||{\"fisher_dtype\":\"bf16\"}|"
"fisherbf16-15k|$F|15000|0.05|0|0.9|||{\"fisher_dtype\":\"bf16\"}|"
# 143-148 the RMS length rule: the 3000-step grid extended below its edge, and a 15000-step grid
"len-rms-3e-4|$F|3000|3e-4|0|0.9|||{\"length\":\"rms\"}|"
"len-rms-1.5e-4|$F|3000|1.5e-4|0|0.9|||{\"length\":\"rms\"}|"
"rms15k-1.5e-4|$F|15000|1.5e-4|0|0.9|||{\"length\":\"rms\"}|"
"rms15k-3e-4|$F|15000|3e-4|0|0.9|||{\"length\":\"rms\"}|"
"rms15k-6e-4|$F|15000|6e-4|0|0.9|||{\"length\":\"rms\"}|"
"rms15k-1.2e-3|$F|15000|1.2e-3|0|0.9|||{\"length\":\"rms\"}|"
# 149-150 which matrices want a larger angle (rms_angles.py: RMS gives v x2.3, o x1.7, q and k ~1): the angle rule
# with v and o at 2x; control, q and k at 2x
"vo-x2|$F|3000|0.2|0|0.9||||--np-angle-mult {\"wv\":2,\"wo\":2}"
"qk-x2|$F|3000|0.2|0|0.9||||--np-angle-mult {\"wq\":2,\"wk\":2}"
# ---- the method with Pion's RMS length rule (user, 2026-11-06). Wave 1: what unlocks the rest.
# 151-154 Algorithm 1 with the RMS rule (Fast with every speed-up off: bases and statistics every step, exact Cayley, fp32)
"rms-alg1-1.5e-4|$F|3000|1.5e-4|0|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|"
"rms-alg1-3e-4|$F|3000|3e-4|0|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|"
"rms-alg1-6e-4|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|"
"rms-alg1-1.2e-3|$F|3000|1.2e-3|0|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|"
# 155-157 T_fac 25 with full statistics (the chain's second step), RMS rule
"rms-tf25-3e-4|$F|3000|3e-4|0|0.9|cayley|fp32|{\"t_fac\":25,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|"
"rms-tf25-6e-4|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":25,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|"
"rms-tf25-1.2e-3|$F|3000|1.2e-3|0|0.9|cayley|fp32|{\"t_fac\":25,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|"
# 158-161 the RMS scale at 9000 and 30000 steps
"rms9k-3e-4|$F|9000|3e-4|0|0.9|||{\"length\":\"rms\"}|"
"rms9k-6e-4|$F|9000|6e-4|0|0.9|||{\"length\":\"rms\"}|"
"rms30k-1.5e-4|$F|30000|1.5e-4|0|0.9|||{\"length\":\"rms\"}|"
"rms30k-3e-4|$F|30000|3e-4|0|0.9|||{\"length\":\"rms\"}|"
# 162-163 the candidate final form: RMS + TF32 before the Cayley step + bf16 F^-1 (Fast's 0.010 rule against rows len-rms-6e-4 and rms15k-3e-4)
"rms-final|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"rms-final-15k|$F|15000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
# 164-167 the method (RMS rule, batched) at 3000 steps, seeds 0-3, on the final implementation
"rms3k|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\"}|"
"rms3k|$F|3000|6e-4|1|0.9|||{\"length\":\"rms\"}|"
"rms3k|$F|3000|6e-4|2|0.9|||{\"length\":\"rms\"}|"
"rms3k|$F|3000|6e-4|3|0.9|||{\"length\":\"rms\"}|"
# ---- wave 2, overnight 2026-11-06: the RMS method in fp32 (the parts every scope needs). Code frozen until it ends.
# 168-173 Section 4: 15000 seeds 1-3; the 30000 grid's third point; 45000 and full length at the 30000 grid's best
"rms15k-3e-4|$F|15000|3e-4|1|0.9|||{\"length\":\"rms\"}|"
"rms15k-3e-4|$F|15000|3e-4|2|0.9|||{\"length\":\"rms\"}|"
"rms15k-3e-4|$F|15000|3e-4|3|0.9|||{\"length\":\"rms\"}|"
"rms30k-6e-4|$F|30000|6e-4|0|0.9|||{\"length\":\"rms\"}|"
"rms45k|$F|45000|best:rms30k|0|0.9|||{\"length\":\"rms\"}|"
"rms73242|$F|73242|best:rms30k|0|0.9|||{\"length\":\"rms\"}|"
# 174-187 Appendix C on the RMS rule: exact-form seeds and 15000; T_fac 10; statistics windows; batching; profiles
"rms-alg1-6e-4|$F|3000|6e-4|1|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|"
"rms-alg1-6e-4|$F|3000|6e-4|2|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|"
"rms-alg1-6e-4|$F|3000|6e-4|3|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|"
"rms-alg1-15k|$F|15000|3e-4|0|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|"
"rms-tf10|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":10,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|"
"rms-win-dlast6|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"d_window\":6}|"
"rms-win-alast6|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.5,\"beta_d\":0.5,\"a_window\":6}|"
"rms-win-adlast6|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":6,\"beta_a\":0.5,\"beta_d\":0.5}|"
"rms-win-aevery2|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.9025,\"beta_d\":0.5,\"a_every\":2}|"
"rms-win-betad09|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.9}|"
"rms-win-betad095|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.95}|"
"rms-form46|$F|3000|6e-4|0|0.9|cayley|fp32|{\"share_inputs\":false,\"batch_sigma\":false,\"length\":\"rms\"}|"
"rms-prof-alg1|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--max-steps 400 --np-profile true"
"rms-prof-tf25|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":25,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--max-steps 400 --np-profile true"
# 188-193 statistics precision and retractions on the RMS method
"rms-stats-fp32|$F|3000|6e-4|0|0.9||fp32|{\"length\":\"rms\"}|"
"rms-stats-fp32-15k|$F|15000|3e-4|0|0.9||fp32|{\"length\":\"rms\"}|"
"rms-ret-cayley|$F|3000|6e-4|0|0.9|cayley||{\"length\":\"rms\"}|"
"rms-ret-ns|$F|3000|6e-4|0|0.9|ns||{\"length\":\"rms\"}|"
"rms-ret-trunc|$F|3000|6e-4|0|0.9|trunc||{\"length\":\"rms\"}|"
"rms-ret-trunc-15k|$F|15000|3e-4|0|0.9|trunc||{\"length\":\"rms\"}|"
# 194-213 Section 4.4 on the RMS method: regularisation, true Fisher, direction
"rms-reg-eps1e-6|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\"}|--ngd-eps 1e-6"
"rms-reg-eps1e-5|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\"}|--ngd-eps 1e-5"
"rms-reg-eps1e-3|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\"}|--ngd-eps 1e-3"
"rms-reg-eps1e-2|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\"}|--ngd-eps 1e-2"
"rms-reg-damp1e-4|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\"}|--ngd-eps 1e-6 --np-damping 1e-4"
"rms-reg-damp1e-3|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\"}|--ngd-eps 1e-6 --np-damping 1e-3"
"rms-reg-damp1e-2|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\"}|--ngd-eps 1e-6 --np-damping 1e-2"
"rms-reg-damp1e-1|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\"}|--ngd-eps 1e-6 --np-damping 1e-1"
"rms-fisher-mc-3e-4|$F|3000|3e-4|0|0.9|||{\"length\":\"rms\"}|--ngd-fisher-mc-every 1"
"rms-fisher-mc-6e-4|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\"}|--ngd-fisher-mc-every 1"
"rms-fisher-mc-1.2e-3|$F|3000|1.2e-3|0|0.9|||{\"length\":\"rms\"}|--ngd-fisher-mc-every 1"
"rms-dir-nat-nomom-3e-4|$F|3000|3e-4|0|0|||{\"length\":\"rms\"}|"
"rms-dir-nat-nomom-6e-4|$F|3000|6e-4|0|0|||{\"length\":\"rms\"}|"
"rms-dir-nat-nomom-1.2e-3|$F|3000|1.2e-3|0|0|||{\"length\":\"rms\"}|"
"rms-dir-raw-mom-3e-4|$F|3000|3e-4|0|0.9|||{\"length\":\"rms\",\"direction\":\"raw\"}|"
"rms-dir-raw-mom-6e-4|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"direction\":\"raw\"}|"
"rms-dir-raw-mom-1.2e-3|$F|3000|1.2e-3|0|0.9|||{\"length\":\"rms\",\"direction\":\"raw\"}|"
"rms-dir-raw-nomom-3e-4|$F|3000|3e-4|0|0|||{\"length\":\"rms\",\"direction\":\"raw\"}|"
"rms-dir-raw-nomom-6e-4|$F|3000|6e-4|0|0|||{\"length\":\"rms\",\"direction\":\"raw\"}|"
"rms-dir-raw-nomom-1.2e-3|$F|3000|1.2e-3|0|0|||{\"length\":\"rms\",\"direction\":\"raw\"}|"
# 214-215 the final form with the Cayley step by scaling and squaring (2026-11-06 night): 3000 and 15000
"rms-final-sq|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"rms-final-sq-15k|$F|15000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
# ---- 216-256: the paper's runs of the final form (RMS rule, TF32 before the Cayley step, bf16 F^-1, Cayley by scaling and
# squaring), on RTX. Launched by scripts/paper/launch_final.sh once rows 214-215 pass. Code frozen until they end.
# 216-223 Section 4: 3000 and 15000 over seeds 0-3
"f3k|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f3k|$F|3000|6e-4|1|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f3k|$F|3000|6e-4|2|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f3k|$F|3000|6e-4|3|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f15k|$F|15000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f15k|$F|15000|3e-4|1|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f15k|$F|15000|3e-4|2|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f15k|$F|15000|3e-4|3|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
# 224-230 the budget curve: 9000 and 30000 scale grids, then 45000 and full length at the 30000 grid's best
"f9k-3e-4|$F|9000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f9k-6e-4|$F|9000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f30k-1.5e-4|$F|30000|1.5e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f30k-3e-4|$F|30000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f30k-6e-4|$F|30000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f45k|$F|45000|best:f30k|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f73242|$F|73242|best:f30k|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
# 231-236 statistics precision and retractions
"f-stats-fp32|$F|3000|6e-4|0|0.9||fp32|{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f-stats-fp32-15k|$F|15000|3e-4|0|0.9||fp32|{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f-ret-cayley|$F|3000|6e-4|0|0.9|cayley||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f-ret-ns|$F|3000|6e-4|0|0.9|ns||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f-ret-trunc|$F|3000|6e-4|0|0.9|trunc||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f-ret-trunc-15k|$F|15000|3e-4|0|0.9|trunc||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
# 237-256 Section 4.4: regularisation, true Fisher, direction
"f-reg-eps1e-6|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-6"
"f-reg-eps1e-5|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-5"
"f-reg-eps1e-3|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"f-reg-eps1e-2|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-2"
"f-reg-damp1e-4|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-6 --np-damping 1e-4"
"f-reg-damp1e-3|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-6 --np-damping 1e-3"
"f-reg-damp1e-2|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-6 --np-damping 1e-2"
"f-reg-damp1e-1|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-6 --np-damping 1e-1"
"f-fisher-mc-3e-4|$F|3000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-fisher-mc-every 1"
"f-fisher-mc-6e-4|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-fisher-mc-every 1"
"f-fisher-mc-1.2e-3|$F|3000|1.2e-3|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-fisher-mc-every 1"
"f-dir-nat-nomom-3e-4|$F|3000|3e-4|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f-dir-nat-nomom-6e-4|$F|3000|6e-4|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f-dir-nat-nomom-1.2e-3|$F|3000|1.2e-3|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|"
"f-dir-raw-mom-3e-4|$F|3000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|"
"f-dir-raw-mom-6e-4|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|"
"f-dir-raw-mom-1.2e-3|$F|3000|1.2e-3|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|"
"f-dir-raw-nomom-3e-4|$F|3000|3e-4|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|"
"f-dir-raw-nomom-6e-4|$F|3000|6e-4|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|"
"f-dir-raw-nomom-1.2e-3|$F|3000|1.2e-3|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|"
# 257-261 floor x refresh cadence on the final form (decides epsilon and T_fac before the long runs):
# eps 1e-4 / 1e-3 x T_fac 25 / 10; (1e-4, 25) is f3k / f15k, (1e-3, 25) at 3000 is f-reg-eps1e-3
"m-tf10-e4|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"t_fac\":10}|"
"m-tf10-e4-15k|$F|15000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"t_fac\":10}|"
"m-tf25-e3-15k|$F|15000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"m-tf10-e3|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"t_fac\":10}|--ngd-eps 1e-3"
"m-tf10-e3-15k|$F|15000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"t_fac\":10}|--ngd-eps 1e-3"
# 262-265 the RMS scale with the floor 1e-3 (6e-4 at 3000 is f-reg-eps1e-3, 3e-4 at 15000 is m-tf25-e3-15k)
"e3-3k-3e-4|$F|3000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"e3-3k-1.2e-3|$F|3000|1.2e-3|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"e3-15k-1.5e-4|$F|15000|1.5e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"e3-15k-6e-4|$F|15000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
# ---- 266-294: the method with the floor 1e-3 (final form + eps 1e-3; user, 2026-11-07). Seed 0 at 3000 and 15000 are
# f-reg-eps1e-3 and m-tf25-e3-15k (the same configuration). Code frozen until these end.
"g3k|$F|3000|6e-4|1|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g3k|$F|3000|6e-4|2|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g3k|$F|3000|6e-4|3|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g15k|$F|15000|3e-4|1|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g15k|$F|15000|3e-4|2|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g15k|$F|15000|3e-4|3|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g9k-3e-4|$F|9000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g9k-6e-4|$F|9000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g30k|$F|30000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g45k|$F|45000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g73242|$F|73242|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g-stats-fp32|$F|3000|6e-4|0|0.9||fp32|{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g-stats-fp32-15k|$F|15000|3e-4|0|0.9||fp32|{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g-ret-cayley|$F|3000|6e-4|0|0.9|cayley||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g-ret-ns|$F|3000|6e-4|0|0.9|ns||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g-ret-trunc|$F|3000|6e-4|0|0.9|trunc||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g-ret-trunc-15k|$F|15000|3e-4|0|0.9|trunc||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g-fisher-mc-3e-4|$F|3000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --ngd-fisher-mc-every 1"
"g-fisher-mc-6e-4|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --ngd-fisher-mc-every 1"
"g-fisher-mc-1.2e-3|$F|3000|1.2e-3|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --ngd-fisher-mc-every 1"
"g-dir-nat-nomom-3e-4|$F|3000|3e-4|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g-dir-nat-nomom-6e-4|$F|3000|6e-4|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g-dir-nat-nomom-1.2e-3|$F|3000|1.2e-3|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g-dir-raw-mom-3e-4|$F|3000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3"
"g-dir-raw-mom-6e-4|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3"
"g-dir-raw-mom-1.2e-3|$F|3000|1.2e-3|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3"
"g-dir-raw-nomom-3e-4|$F|3000|3e-4|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3"
"g-dir-raw-nomom-6e-4|$F|3000|6e-4|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3"
"g-dir-raw-nomom-1.2e-3|$F|3000|1.2e-3|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3"
# 295-312: Appendix C's chain with the floor 1e-3 (the method's floor), RMS rule, stages before TF32 / bf16
"c-alg1-3e-4|$F|3000|3e-4|0|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3"
"c-alg1-6e-4|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3"
"c-alg1-1.2e-3|$F|3000|1.2e-3|0|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3"
"c-alg1-seeds|$F|3000|best:c-alg1|1|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3"
"c-alg1-seeds|$F|3000|best:c-alg1|2|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3"
"c-alg1-seeds|$F|3000|best:c-alg1|3|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3"
"c-alg1-15k|$F|15000|3e-4|0|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3"
"c-tf25|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":25,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3"
"c-tf10|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":10,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3"
"c-win-dlast6|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"d_window\":6}|--ngd-eps 1e-3"
"c-win-alast6|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.5,\"beta_d\":0.5,\"a_window\":6}|--ngd-eps 1e-3"
"c-win-adlast6|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":6,\"beta_a\":0.5,\"beta_d\":0.5}|--ngd-eps 1e-3"
"c-win-aevery2|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.9025,\"beta_d\":0.5,\"a_every\":2}|--ngd-eps 1e-3"
"c-win-betad09|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.9}|--ngd-eps 1e-3"
"c-win-betad095|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.95}|--ngd-eps 1e-3"
"c-form46|$F|3000|6e-4|0|0.9|cayley|fp32|{\"share_inputs\":false,\"batch_sigma\":false,\"length\":\"rms\"}|--ngd-eps 1e-3"
"c-prof-alg1|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --max-steps 400 --np-profile true"
"c-prof-tf25|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":25,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --max-steps 400 --np-profile true"
# 313-324 (2026-11-08): two more points beyond the edge of each ablation grid whose best was at an edge
"g-dir-raw-mom-1.5e-4|$F|3000|1.5e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3"
"g-dir-raw-nomom-1.5e-4|$F|3000|1.5e-4|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3"
"g-dir-raw-mom-7.5e-5|$F|3000|7.5e-5|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3"
"g-dir-raw-nomom-7.5e-5|$F|3000|7.5e-5|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3"
"g-dir-nat-nomom-2.4e-3|$F|3000|2.4e-3|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"g-dir-nat-nomom-4.8e-3|$F|3000|4.8e-3|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3"
"len-fisher-1.2e-3|$F|3000|1.2e-3|0|0.9|||{\"length\":\"fisher\"}|"
"len-fisher-6e-4|$F|3000|6e-4|0|0.9|||{\"length\":\"fisher\"}|"
"len-quad-5e-4|$F|3000|5e-4|0|0.9|||{\"length\":\"quad\"}|"
"len-quad-2.5e-4|$F|3000|2.5e-4|0|0.9|||{\"length\":\"quad\"}|"
"len-fixed-2.5e-4|$F|3000|2.5e-4|0|0.9|||{\"length\":\"fixed\"}|"
"len-fixed-1.25e-4|$F|3000|1.25e-4|0|0.9|||{\"length\":\"fixed\"}|"
# 325-326 (2026-11-08): the floor follows the step size, eps(t) = 1e-3 * lr(t) / lr(0); against g3k s0 / g73242
"h-floorlr-3k|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"floor_schedule\":\"lr\"}|--ngd-eps 1e-3"
"h-floorlr-73242|$F|73242|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"floor_schedule\":\"lr\"}|--ngd-eps 1e-3"
# 327-330 (2026-11-08): depth of the cosine decay; lr_min was an absolute 1e-5 (3.3% of the RMS peak at full
# length, 0.02% of the angle rule's). d1pc: 1% of the peak, as Pion; d02pm: 0.02%, as the angle rule. Against g3k s0 / g73242
"d1pc-3k|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min 6e-6"
"d1pc-73242|$F|73242|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min 3e-6"
"d02pm-3k|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min 1.2e-7"
"d02pm-73242|$F|73242|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min 6e-8"
# 331-446 (2026-11-09): the k-series. Every run the preprint reads, again at a cosine decay depth of 1% of
# each optimizer's peak (--lr-min-ratio 0.01, as Pion's 1e-5 at lr 1e-3); the method's scale re-chosen per budget
# (k-m3k, k-g9k, k-m15k, k-m30k); rows at the method's scale take it from those grids (best:k-m3k / best:k-m15k).
"k-m3k-3e-4|$F|3000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-m3k-6e-4|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-m3k-1.2e-3|$F|3000|1.2e-3|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-m3k-2.4e-3|$F|3000|2.4e-3|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g9k-1.5e-4|$F|9000|1.5e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g9k-3e-4|$F|9000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g9k-6e-4|$F|9000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g9k-1.2e-3|$F|9000|1.2e-3|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-m15k-1.5e-4|$F|15000|1.5e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-m15k-3e-4|$F|15000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-m15k-6e-4|$F|15000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-m15k-1.2e-3|$F|15000|1.2e-3|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-m30k-1.5e-4|$F|30000|1.5e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-m30k-3e-4|$F|30000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-m30k-6e-4|$F|30000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-m30k-1.2e-3|$F|30000|1.2e-3|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g45k|$F|45000|best:k-m30k|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-dir-nat-mom-0.1|$F|3000|0.1|0|0.9||||--lr-min-ratio 0.01"
"k-dir-nat-mom-0.4|$F|3000|0.4|0|0.9||||--lr-min-ratio 0.01"
"k-len-before-3e-6|$F|3000|3e-6|0|0.9|||{\"length\":\"before\"}|--lr-min-ratio 0.01"
"k-len-before-1e-5|$F|3000|1e-5|0|0.9|||{\"length\":\"before\"}|--lr-min-ratio 0.01"
"k-len-before-3e-5|$F|3000|3e-5|0|0.9|||{\"length\":\"before\"}|--lr-min-ratio 0.01"
"k-len-before-1e-4|$F|3000|1e-4|0|0.9|||{\"length\":\"before\"}|--lr-min-ratio 0.01"
"k-len-before-3e-4|$F|3000|3e-4|0|0.9|||{\"length\":\"before\"}|--lr-min-ratio 0.01"
"k-len-rms-6e-4|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\"}|--lr-min-ratio 0.01"
"k-len-rms-1.2e-3|$F|3000|1.2e-3|0|0.9|||{\"length\":\"rms\"}|--lr-min-ratio 0.01"
"k-len-rms-2.4e-3|$F|3000|2.4e-3|0|0.9|||{\"length\":\"rms\"}|--lr-min-ratio 0.01"
"k-len-rms-4.8e-3|$F|3000|4.8e-3|0|0.9|||{\"length\":\"rms\"}|--lr-min-ratio 0.01"
"k-len-rms-9.6e-3|$F|3000|9.6e-3|0|0.9|||{\"length\":\"rms\"}|--lr-min-ratio 0.01"
"k-len-fisher-2.4e-3|$F|3000|2.4e-3|0|0.9|||{\"length\":\"fisher\"}|--lr-min-ratio 0.01"
"k-len-fisher-4.8e-3|$F|3000|4.8e-3|0|0.9|||{\"length\":\"fisher\"}|--lr-min-ratio 0.01"
"k-len-fisher-9.6e-3|$F|3000|9.6e-3|0|0.9|||{\"length\":\"fisher\"}|--lr-min-ratio 0.01"
"k-len-fisher-1.9e-2|$F|3000|1.9e-2|0|0.9|||{\"length\":\"fisher\"}|--lr-min-ratio 0.01"
"k-len-fisher-3.8e-2|$F|3000|3.8e-2|0|0.9|||{\"length\":\"fisher\"}|--lr-min-ratio 0.01"
"k-len-quad-1e-3|$F|3000|1e-3|0|0.9|||{\"length\":\"quad\"}|--lr-min-ratio 0.01"
"k-len-quad-2e-3|$F|3000|2e-3|0|0.9|||{\"length\":\"quad\"}|--lr-min-ratio 0.01"
"k-len-quad-4e-3|$F|3000|4e-3|0|0.9|||{\"length\":\"quad\"}|--lr-min-ratio 0.01"
"k-len-quad-8e-3|$F|3000|8e-3|0|0.9|||{\"length\":\"quad\"}|--lr-min-ratio 0.01"
"k-len-quad-1.6e-2|$F|3000|1.6e-2|0|0.9|||{\"length\":\"quad\"}|--lr-min-ratio 0.01"
"k-len-fixed-5e-4|$F|3000|5e-4|0|0.9|||{\"length\":\"fixed\"}|--lr-min-ratio 0.01"
"k-len-fixed-1e-3|$F|3000|1e-3|0|0.9|||{\"length\":\"fixed\"}|--lr-min-ratio 0.01"
"k-len-fixed-2e-3|$F|3000|2e-3|0|0.9|||{\"length\":\"fixed\"}|--lr-min-ratio 0.01"
"k-len-fixed-4e-3|$F|3000|4e-3|0|0.9|||{\"length\":\"fixed\"}|--lr-min-ratio 0.01"
"k-len-fixed-8e-3|$F|3000|8e-3|0|0.9|||{\"length\":\"fixed\"}|--lr-min-ratio 0.01"
"k-s3000|$F|3000|0.2|0|0.9||||--lr-min-ratio 0.01"
"k-s3000|$F|3000|0.2|1|0.9||||--lr-min-ratio 0.01"
"k-s3000|$F|3000|0.2|2|0.9||||--lr-min-ratio 0.01"
"k-s3000|$F|3000|0.2|3|0.9||||--lr-min-ratio 0.01"
"k-len-rms-seeds|$F|3000|best:k-len-rms|1|0.9|||{\"length\":\"rms\"}|--lr-min-ratio 0.01"
"k-len-rms-seeds|$F|3000|best:k-len-rms|2|0.9|||{\"length\":\"rms\"}|--lr-min-ratio 0.01"
"k-len-rms-seeds|$F|3000|best:k-len-rms|3|0.9|||{\"length\":\"rms\"}|--lr-min-ratio 0.01"
"k-dir-nat-mom-0.8|$F|3000|0.8|0|0.9||||--lr-min-ratio 0.01"
"k-len-rms-3e-4|$F|3000|3e-4|0|0.9|||{\"length\":\"rms\"}|--lr-min-ratio 0.01"
"k-len-rms-1.5e-4|$F|3000|1.5e-4|0|0.9|||{\"length\":\"rms\"}|--lr-min-ratio 0.01"
"k-rms15k-3e-4|$F|15000|best:k-m15k|0|0.9|||{\"length\":\"rms\"}|--lr-min-ratio 0.01"
"k-rms3k|$F|3000|best:k-m3k|0|0.9|||{\"length\":\"rms\"}|--lr-min-ratio 0.01"
"k-rms-final-sq|$F|3000|best:k-m3k|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--lr-min-ratio 0.01"
"k-rms-final-sq-15k|$F|15000|best:k-m15k|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--lr-min-ratio 0.01"
"k-f3k|$F|3000|best:k-m3k|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--lr-min-ratio 0.01"
"k-f-reg-eps1e-6|$F|3000|best:k-m3k|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-6 --lr-min-ratio 0.01"
"k-f-reg-eps1e-5|$F|3000|best:k-m3k|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-5 --lr-min-ratio 0.01"
"k-f-reg-eps1e-2|$F|3000|best:k-m3k|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-2 --lr-min-ratio 0.01"
"k-f-reg-damp1e-4|$F|3000|best:k-m3k|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-6 --np-damping 1e-4 --lr-min-ratio 0.01"
"k-f-reg-damp1e-3|$F|3000|best:k-m3k|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-6 --np-damping 1e-3 --lr-min-ratio 0.01"
"k-f-reg-damp1e-2|$F|3000|best:k-m3k|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-6 --np-damping 1e-2 --lr-min-ratio 0.01"
"k-f-reg-damp1e-1|$F|3000|best:k-m3k|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-6 --np-damping 1e-1 --lr-min-ratio 0.01"
"k-g-stats-fp32|$F|3000|best:k-m3k|0|0.9||fp32|{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g-stats-fp32-15k|$F|15000|best:k-m15k|0|0.9||fp32|{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g-ret-cayley|$F|3000|best:k-m3k|0|0.9|cayley||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g-ret-ns|$F|3000|best:k-m3k|0|0.9|ns||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g-ret-trunc|$F|3000|best:k-m3k|0|0.9|trunc||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g-ret-trunc-15k|$F|15000|best:k-m15k|0|0.9|trunc||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g-fisher-mc-3e-4|$F|3000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --ngd-fisher-mc-every 1 --lr-min-ratio 0.01"
"k-g-fisher-mc-6e-4|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --ngd-fisher-mc-every 1 --lr-min-ratio 0.01"
"k-g-fisher-mc-1.2e-3|$F|3000|1.2e-3|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --ngd-fisher-mc-every 1 --lr-min-ratio 0.01"
"k-g-fisher-mc-2.4e-3|$F|3000|2.4e-3|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --ngd-fisher-mc-every 1 --lr-min-ratio 0.01"
"k-g-dir-nat-nomom-3e-4|$F|3000|3e-4|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g-dir-nat-nomom-6e-4|$F|3000|6e-4|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g-dir-nat-nomom-1.2e-3|$F|3000|1.2e-3|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g-dir-raw-mom-3e-4|$F|3000|3e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g-dir-raw-mom-6e-4|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g-dir-raw-mom-1.2e-3|$F|3000|1.2e-3|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g-dir-raw-nomom-3e-4|$F|3000|3e-4|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g-dir-raw-nomom-6e-4|$F|3000|6e-4|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g-dir-raw-nomom-1.2e-3|$F|3000|1.2e-3|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-alg1-3e-4|$F|3000|3e-4|0|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-alg1-6e-4|$F|3000|6e-4|0|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-alg1-1.2e-3|$F|3000|1.2e-3|0|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-alg1-2.4e-3|$F|3000|2.4e-3|0|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-alg1-seeds|$F|3000|best:k-c-alg1|1|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-alg1-seeds|$F|3000|best:k-c-alg1|2|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-alg1-seeds|$F|3000|best:k-c-alg1|3|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-alg1-15k|$F|15000|best:k-m15k|0|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-tf25|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"t_fac\":25,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-tf10|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"t_fac\":10,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-win-dlast6|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"d_window\":6}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-win-alast6|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.5,\"beta_d\":0.5,\"a_window\":6}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-win-adlast6|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":6,\"beta_a\":0.5,\"beta_d\":0.5}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-win-aevery2|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.9025,\"beta_d\":0.5,\"a_every\":2}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-win-betad09|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.9}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-win-betad095|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.95}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-form46|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"share_inputs\":false,\"batch_sigma\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-prof-alg1|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --max-steps 400 --np-profile true --lr-min-ratio 0.01"
"k-c-prof-tf25|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"t_fac\":25,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --max-steps 400 --np-profile true --lr-min-ratio 0.01"
"k-g-dir-raw-mom-1.5e-4|$F|3000|1.5e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g-dir-raw-nomom-1.5e-4|$F|3000|1.5e-4|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g-dir-raw-mom-7.5e-5|$F|3000|7.5e-5|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g-dir-raw-nomom-7.5e-5|$F|3000|7.5e-5|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\",\"direction\":\"raw\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g-dir-nat-nomom-2.4e-3|$F|3000|2.4e-3|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g-dir-nat-nomom-4.8e-3|$F|3000|4.8e-3|0|0|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-len-fisher-1.2e-3|$F|3000|1.2e-3|0|0.9|||{\"length\":\"fisher\"}|--lr-min-ratio 0.01"
"k-len-fisher-6e-4|$F|3000|6e-4|0|0.9|||{\"length\":\"fisher\"}|--lr-min-ratio 0.01"
"k-len-quad-5e-4|$F|3000|5e-4|0|0.9|||{\"length\":\"quad\"}|--lr-min-ratio 0.01"
"k-len-quad-2.5e-4|$F|3000|2.5e-4|0|0.9|||{\"length\":\"quad\"}|--lr-min-ratio 0.01"
"k-len-fixed-2.5e-4|$F|3000|2.5e-4|0|0.9|||{\"length\":\"fixed\"}|--lr-min-ratio 0.01"
"k-len-fixed-1.25e-4|$F|3000|1.25e-4|0|0.9|||{\"length\":\"fixed\"}|--lr-min-ratio 0.01"
# 447-452: the method's seeds 1-3 at 3000 and 15000 (seed 0 is the best point of k-m3k / k-m15k)
"k-g3k|$F|3000|best:k-m3k|1|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g3k|$F|3000|best:k-m3k|2|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g3k|$F|3000|best:k-m3k|3|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g15k|$F|15000|best:k-m15k|1|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g15k|$F|15000|best:k-m15k|2|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-g15k|$F|15000|best:k-m15k|3|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
# 453-480 (2026-11-09): Table 8 and Appendix C on the method's configuration (floor 1e-3, TF32 before the rotation, bf16 F^-1,
# depth 1%); only the length rule changes. The RMS row is the method's own grid k-m3k and seeds k-g3k.
"k-t13-angle-0.05|$F|3000|0.05|0|0.9|||{\"length\":\"spectral\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-angle-0.1|$F|3000|0.1|0|0.9|||{\"length\":\"spectral\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-angle-0.2|$F|3000|0.2|0|0.9|||{\"length\":\"spectral\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-angle-0.4|$F|3000|0.4|0|0.9|||{\"length\":\"spectral\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-angle-0.8|$F|3000|0.8|0|0.9|||{\"length\":\"spectral\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-before-1e-6|$F|3000|1e-6|0|0.9|||{\"length\":\"before\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-before-3e-6|$F|3000|3e-6|0|0.9|||{\"length\":\"before\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-before-1e-5|$F|3000|1e-5|0|0.9|||{\"length\":\"before\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-before-3e-5|$F|3000|3e-5|0|0.9|||{\"length\":\"before\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-before-1e-4|$F|3000|1e-4|0|0.9|||{\"length\":\"before\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-fisher-3e-4|$F|3000|3e-4|0|0.9|||{\"length\":\"fisher\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-fisher-6e-4|$F|3000|6e-4|0|0.9|||{\"length\":\"fisher\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-fisher-1.2e-3|$F|3000|1.2e-3|0|0.9|||{\"length\":\"fisher\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-fisher-2.4e-3|$F|3000|2.4e-3|0|0.9|||{\"length\":\"fisher\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-fisher-4.8e-3|$F|3000|4.8e-3|0|0.9|||{\"length\":\"fisher\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-quad-2.5e-4|$F|3000|2.5e-4|0|0.9|||{\"length\":\"quad\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-quad-5e-4|$F|3000|5e-4|0|0.9|||{\"length\":\"quad\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-quad-1e-3|$F|3000|1e-3|0|0.9|||{\"length\":\"quad\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-quad-2e-3|$F|3000|2e-3|0|0.9|||{\"length\":\"quad\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-quad-4e-3|$F|3000|4e-3|0|0.9|||{\"length\":\"quad\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-fixed-6.25e-5|$F|3000|6.25e-5|0|0.9|||{\"length\":\"fixed\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-fixed-1.25e-4|$F|3000|1.25e-4|0|0.9|||{\"length\":\"fixed\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-fixed-2.5e-4|$F|3000|2.5e-4|0|0.9|||{\"length\":\"fixed\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-fixed-5e-4|$F|3000|5e-4|0|0.9|||{\"length\":\"fixed\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-fixed-1e-3|$F|3000|1e-3|0|0.9|||{\"length\":\"fixed\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-angle-seeds|$F|3000|best:k-t13-angle|1|0.9|||{\"length\":\"spectral\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-angle-seeds|$F|3000|best:k-t13-angle|2|0.9|||{\"length\":\"spectral\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-t13-angle-seeds|$F|3000|best:k-t13-angle|3|0.9|||{\"length\":\"spectral\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
# 481-482 (2026-11-09): the method with AdamW lr 4e-3 / 1.6e-2 around the shared 8e-3 (tuned for Pion); against k-m3k-6e-4
"k-adamw-4e-3|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01 --adamw-lr 4e-3"
"k-adamw-1.6e-2|$F|3000|6e-4|0|0.9|||{\"length\":\"rms\",\"tf32\":\"upstream\",\"fisher_dtype\":\"bf16\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01 --adamw-lr 1.6e-2"
# 483-484 (2026-11-09): the precision check on the method's configuration: the method with every product before the
# rotation in fp32 (no TF32, F^-1 in fp32), against the method's own runs at 3000 / 15000 steps
"k-prec-fp32-3k|$F|3000|best:k-m3k|0|0.9|||{\"length\":\"rms\",\"tf32\":\"off\",\"fisher_dtype\":\"fp32\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-prec-fp32-15k|$F|15000|best:k-m15k|0|0.9|||{\"length\":\"rms\",\"tf32\":\"off\",\"fisher_dtype\":\"fp32\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
# 485-486 (2026-11-10): Algorithm 1's step time and profile without the sigma its exact solve does not need (the code
# skips it since 2026-11-10); same configuration as k-c-alg1-<best> and k-c-prof-alg1
"k-c-alg1-speed|$F|3000|best:k-c-alg1|0|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-prof-alg1-nosig|$F|3000|best:k-c-alg1|0|0.9|cayley|fp32|{\"t_fac\":1,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --max-steps 400 --np-profile true --lr-min-ratio 0.01"
# 487-496 (2026-11-10): the chain's step times without the unneeded sigma (same rows as k-c-*, new code)
"k-c-tf10-speed|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"t_fac\":10,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-tf25-speed|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"t_fac\":25,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-win-dlast6-speed|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"d_window\":6}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-win-alast6-speed|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.5,\"beta_d\":0.5,\"a_window\":6}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-win-adlast6-speed|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":6,\"beta_a\":0.5,\"beta_d\":0.5}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-win-aevery2-speed|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.9025,\"beta_d\":0.5,\"a_every\":2}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-win-betad09-speed|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.9}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-win-betad095-speed|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"t_fac\":25,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\",\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.95}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-form46-speed|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"share_inputs\":false,\"batch_sigma\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --lr-min-ratio 0.01"
"k-c-prof-tf25-nosig|$F|3000|best:k-m3k|0|0.9|cayley|fp32|{\"t_fac\":25,\"stats_window\":null,\"beta_a\":0.95,\"beta_d\":0.5,\"share_inputs\":false,\"batch_sigma\":false,\"share_root\":false,\"length\":\"rms\"}|--ngd-eps 1e-3 --max-steps 400 --np-profile true --lr-min-ratio 0.01"
# 497-499 (2026-11-10): their pion_msign (reference/pion/.../pion_msign.py): msign of the generator by Muon's
# Newton-Schulz, raw-gradient EMA 0.95, alternating sides, truncated exponential; rotation lr grid around their 1e-2
"k-pmsign-5e-3|pion|3000|5e-3|0|0.9||||--pion-scaling msign --pion-momentum ambient --pion-beta1 0.95 --pion-alternate true --lr-min-ratio 0.01"
"k-pmsign-1e-2|pion|3000|1e-2|0|0.9||||--pion-scaling msign --pion-momentum ambient --pion-beta1 0.95 --pion-alternate true --lr-min-ratio 0.01"
"k-pmsign-2e-2|pion|3000|2e-2|0|0.9||||--pion-scaling msign --pion-momentum ambient --pion-beta1 0.95 --pion-alternate true --lr-min-ratio 0.01"
# 500-501 (2026-11-11): pion_msign's grid past its edge (2e-2 was best)
"k-pmsign-4e-2|pion|3000|4e-2|0|0.9||||--pion-scaling msign --pion-momentum ambient --pion-beta1 0.95 --pion-alternate true --lr-min-ratio 0.01"
"k-pmsign-8e-2|pion|3000|8e-2|0|0.9||||--pion-scaling msign --pion-momentum ambient --pion-beta1 0.95 --pion-alternate true --lr-min-ratio 0.01"
)
IFS='|' read -r TAG OPT STEPS LR SEED B1 RET STATS KW EXTRA <<< "${ROWS[$SLURM_ARRAY_TASK_ID]}"
case "$RET" in ""|cayley|ns|ns_delta|trunc) ;; *) echo "row $SLURM_ARRAY_TASK_ID: bad RETRACTION '$RET'"; exit 1 ;; esac
case "$STATS" in ""|fp32|bf16) ;; *) echo "row $SLURM_ARRAY_TASK_ID: bad STATS '$STATS'"; exit 1 ;; esac
case "$KW" in ""|"{"*"}") ;; *) echo "row $SLURM_ARRAY_TASK_ID: bad NP_KWARGS '$KW'"; exit 1 ;; esac
# LR "best:<prefix>": the scale (lr) of the best finished run among the tags <prefix>-<angle> (seed 0), read from their
# logs at start; refuses if any of them is unfinished. Submit such rows with --dependency=afterany:<the grid's job>.
if [ "${LR#best:}" != "$LR" ]; then
    LR=$(python3 - "${LR#best:}" <<'PY'
import glob, json, sys
prefix, best, todo = sys.argv[1], None, []
for m in glob.glob(f"/onyx/data/p330/runs/npv2*/{prefix}-*/*/manifest.json"):
    try:                                     # only the grid itself: <prefix>-<angle>, not e.g. <prefix>-15k
        float(m.split("/")[-3][len(prefix) + 1:])
    except ValueError:
        continue
    c = json.load(open(m))["config"]
    if c["seed"] != 0:
        continue
    vals = [json.loads(l) for l in open(m.replace("manifest.json", "log.jsonl")) if '"val_loss"' in l]
    if not vals or vals[-1]["step"] < c["train_steps"] - 1:
        todo.append(m.split("/")[-3]); continue
    v = vals[-1]["val_loss"]
    if not isinstance(v, (int, float)) or v != v:   # a diverged point is never the best
        continue
    if best is None or v < best[0]:
        best = (v, c["lr"])
if todo or best is None:
    sys.exit(f"grid {prefix}: unfinished {todo or 'all'}")
print(best[1])
PY
) || { echo "cannot resolve the angle of $TAG"; exit 1; }
fi
EV=500; [ "$STEPS" -gt 3000 ] && EV=1000; [ "$STEPS" -gt 15000 ] && EV=2000
OPT_ARGS=()
[ -n "$RET" ] && OPT_ARGS+=(--np-retraction "$RET")
[ -n "$STATS" ] && OPT_ARGS+=(--np-stats-dtype "$STATS")
[ -n "$KW" ] && OPT_ARGS+=(--np-kwargs "$KW")
OUT="$DATA_p330/runs/npv2/$TAG"; mkdir -p "$OUT"
ADAMW="--adamw-lr 8e-3"; case "$EXTRA" in *--adamw-lr*) ADAMW="" ;; esac   # a row may set its own AdamW lr
echo "row $SLURM_ARRAY_TASK_ID: $TAG $OPT steps $STEPS angle $LR seed $SEED beta1 $B1 $RET $STATS kw=$KW $EXTRA"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run --optimizer "$OPT" "${OPT_ARGS[@]}" $EXTRA \
      --lr "$LR" --ngd-beta1 "$B1" --train-steps "$STEPS" --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
      --precision bf16 --seed "$SEED" --device cuda --log-every 23 --eval-every "$EV" $ADAMW \
      --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
