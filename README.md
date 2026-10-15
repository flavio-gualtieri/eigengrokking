# eigengrok

Grokking is when a network memorizes its training set early and then, thousands of steps later, suddenly starts to
generalize. I wanted to know whether the curvature of the loss gives any warning before that jump.

I trained 240 small transformers on modular addition (a + b mod 91), varying weight decay, initialization scale and
seed, and tracked the Hessian and its Gauss-Newton part throughout training. Then I tried to predict each run's
grokking step from the first few hundred steps after it memorized the training set.

**Short answer: a little.** Curvature improves the prediction only slightly over a baseline that already knows the
hyperparameters and a few cheap training statistics. The clearest curvature signal, a sharp drop in effective rank,
happens at the transition rather than before it.

![Curvature around the grokking transition](reports/figures/forecast_trajectory.png)

## Results

Each run's grokking step is predicted by a model that never saw its hyperparameter setting. The score is R² on
seed-to-seed differences within a setting, the part the hyperparameters can't explain (30% of the variance).
Brackets are 95% bootstrap intervals.

| model | baseline (hyperparameters + training stats) | gain from adding curvature |
|---|---|---|
| ridge regression | 0.17 [0.08, 0.27] | +0.106 [+0.022, +0.180] |
| gradient-boosted trees | 0.48 [0.38, 0.58] | +0.120 [+0.021, +0.216] |

So curvature helps, but not by much: seed-level R² goes from 0.17 to 0.28 with ridge, and from 0.48 to 0.60 with
trees. The more surprising part is how much the cheap statistics (loss, weight norms, gradient norm) already explain.

## A detour: the training was broken

In the first version of the sweep, every single run lost its solution after grokking, a median of 39 times in 150k
steps. It turned out to be AdamW's default beta2 = 0.999, not numerical precision: rerunning in float64 made no
clear difference, while beta2 = 0.98 left about a tenth as many collapses and delayed grokking by about 10% or less.
All results above come from the rerun with beta2 = 0.98.

![Collapses under each optimizer setting](reports/figures/collapse_pilot_test_acc.png)

## How it works

The Hessian of a 220k-parameter network is never formed. Its spectrum is estimated with stochastic Lanczos
quadrature, using Hessian-vector products on the full training set. Each run reuses the same random probes at every
checkpoint, so the change between checkpoints is measured much more precisely than the value itself. The estimator is
tested against exact Hessians on small models, and costs about 2.3 s per checkpoint on an A100. The rerun, the
optimizer experiments and the curvature took about 131 A100-hours.

## Caveats

- One task and one small model. This says nothing yet about larger models or real data.
- The gain is small and depends on analysis choices. Without clipping held-out features to the training range (a
  step I added after the first full run), the ridge gain shrinks to +0.03 and is no longer distinguishable from zero.
  Across the other robustness checks, some show a gain and some don't.
- Nothing transfers across initialization scales: a model trained on one fails on the other.

## Reproduce

```bash
pip install -e ".[dev]"     # Python 3.11+, versions pinned in requirements.txt
make test
EIGENGROK_REPO=$PWD bash scripts/run_audit_pipeline.sh   # every table and figure from the data in reports/ (~30 min on CPU)
```

Training and curvature need GPUs; the SLURM scripts are in `scripts/`. [`RESULTS.md`](RESULTS.md) lists every
number with the script and data behind it, and `reports/` has the full tables, including all the robustness checks.
