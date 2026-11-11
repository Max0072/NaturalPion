# Natural Pion

Code, run records and source of the preprint *Natural Pion: Spectrum-Preserving Rotations along the Natural Gradient*
(Maxim Shikunov, November 2026): `docs/paper/latex/main.pdf`.

Natural Pion trains each weight matrix W by rotating it, W <- exp(X_out) W exp(X_in), like Pion, so W's singular
values never change. The rotation generators X are preconditioned by a K-FAC Fisher on the generators, inverted in the
generalized eigenbasis of A (the second moment of a layer's inputs) and D (the second moment of its output gradients).
On LLaMA-60M trained on C4 it reaches the losses Pion reaches at 1.90-2.12x fewer steps; at full length (73 242 steps)
the validation loss is 3.321 against Pion's 3.371.

## Use the optimizer

```python
import torch
from ngd_pion import NaturalPionFast
from ngd_pion.hooks import attach, attach_backward

linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]   # the matrices to rotate
weights = [m.weight for m in linears]
rest = [p for p in model.parameters() if all(p is not w for w in weights)]

opt = NaturalPionFast(weights, lr=3e-4)                    # lr: the peak step scale (Pion's RMS rule)
adamw = torch.optim.AdamW(rest, lr=8e-3, weight_decay=0.1) # embeddings, norms, output head
attach(linears, opt); attach_backward(linears, opt)        # record A (inputs) and D (output gradients)

for batch in data:
    loss_fn(model, batch).backward()
    opt.step(); adamw.step()
    opt.zero_grad(); adamw.zero_grad()
```

The class defaults are the method of the paper (Algorithm 1). The paper's schedule is cosine decay to 1% of the peak,
with peak 6e-4 at 3000 steps and 3e-4 from 9000 steps on. `harness/train.py` (`build_optimizers`) is the exact wiring
of the paper's runs, including the query projection rotated per attention head.

## Layout

| path | what it is |
|---|---|
| `ngd_pion/` | the optimizer: `natural_pion.py` (the method), `pion_baseline.py` (published Pion, the baseline), primitives, `reference.py` (a numpy oracle for the tests); see `ngd_pion/README.md` |
| `harness/` | the training harness: LLaMA-60M, the C4 token corpus, configuration (`config.py`), training loop |
| `tests/` | `pytest tests` |
| `runs-record/` | the manifest (full configuration) and log of every run the paper reads, and the probe outputs |
| `scripts/analysis/` | `paper_numbers.py` computes every number of the paper from `runs-record/` into `docs/paper/numbers.json`; `render_tex.py` writes them into `docs/paper/latex/numbers.tex`; `arxiv_bundle.py` packs the arXiv source; `budget_fit.py`, `figures.py`, `flops.py`, `check_fisher.py`; `render_paper.py` fills the markdown draft |
| `scripts/probes/` | the measurements the paper cites (retraction drift, Newton-Schulz rounding, step-time benchmark, ...) |
| `scripts/paper/` | the job scripts of the runs (SLURM, see `scripts/paper/README.md`) |
| `scripts/prepare_data.py` | tokenises C4 into the binary corpus the harness reads |
| `docs/paper/` | `latex/` (the paper: `main.tex`, every number a `\pn{key}{format}` filled from `numbers.json`; `main.pdf`), `numbers.json`, `numbers.md` (every number with its source), figures; `preprint.md` is the markdown draft the LaTeX was converted from |

## Reproduce

Regenerate the paper's numbers and the PDF from the run records (CPU, seconds; the PDF needs
[tectonic](https://tectonic-typesetting.github.io)):

```bash
pip install -e ".[test,paper]"
python scripts/analysis/paper_numbers.py     # -> docs/paper/numbers.json
python scripts/analysis/render_tex.py        # -> docs/paper/latex/numbers.tex
(cd docs/paper/latex && tectonic main.tex)   # -> docs/paper/latex/main.pdf
pytest tests
```

Three numbers (`mc_err_in`, `mc_err_out`, `pullback_err`, Monte Carlo and round-off checks of Appendix A) come out
slightly different on another machine (the pull-back residual is pure round-off: 3.6e-15 on the cluster, 5.3e-15 on
an Apple laptop); every other number is reproduced exactly.

Re-run a training run from its record on one GPU, after `scripts/prepare_data.py` (`pip install -e ".[data]"`):

```bash
python -m harness.run --from-manifest runs-record/<...>/manifest.json \
    --data-path data/c4_train.bin --val-path data/c4_val.bin --out-dir runs/<name>
```

(the manifests record the cluster's paths). The job scripts in `scripts/paper/` carry the `#SBATCH` lines and `$DATA_p330` paths of the Aphrodite cluster
of The Cyprus Institute, where the runs were made (PyTorch 2.13, NVIDIA RTX-class cards).

Runs recorded before 2026-11-11 were made when `NaturalPionFast`'s defaults were older; their manifests carry no
`defaults_version`, and `RunConfig.from_manifest` restores the defaults they ran with.
