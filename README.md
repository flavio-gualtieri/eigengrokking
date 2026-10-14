# eigengrok

Grokking is the delayed jump from memorization to generalization: a small
network fits its training set quickly, then sits at chance on held-out data
for many times longer before test accuracy suddenly climbs. This
repo asks whether the curvature of the loss (the Hessian's eigenvalue spectrum,
estimated by stochastic Lanczos quadrature and validated against exact
Hessians) forecasts *when* that jump happens, beyond what the hyperparameters
and cheap training statistics already predict. The testbed is a 1-layer
transformer (~220k parameters) trained full-batch with AdamW on modular
addition (p = 91, 30% of pairs for training), swept over 12 weight decays x 2
initialization scales x 10 seeds.

Status: infrastructure rebuild in progress. Findings: TODO.

## Results

TODO. Until then, `reports/sweep_summary.md` holds the grokking milestones of
all 240 sweep runs (memorization, onset, grokking, post-grokking collapses)
with bootstrap intervals, and `paper/grokking.tex` is the working draft.

## Setup

Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # exact versions pinned in requirements.txt
```

## Check that it works

```bash
make test     # unit + end-to-end tests, about a minute on one CPU core
make smoke    # 300-step run with live Hessian snapshots; writes runs/smoke/...
```

## Regenerate the reports and figures

From the committed data alone (`reports/wd_sweep_accuracy_curves.csv`, one row
per run and logged step), on a CPU in seconds:

```bash
python scripts/sweep_summary.py          # reports/sweep_summary.{md,csv}, reports/sweep_milestones.csv
python -m eigengrok.analysis.onset_vs_wd # reports/figures/onset_vs_wd.{png,pdf}
```

The spectral figures also read each run's training-curve npz
(`<run_dir>/figures/training_data_*.npz`, with the weight norms), which lives
in the run directories under `$EIGENGROK_OUTPUT_DIR` (default `results/`), not
in the repo. With those present, on a CPU:

```bash
RUN=results/MODULAR/no_spectrum/modulus=91/depth=1/width=128/init=1
python -m eigengrok.analysis.make_figures reports/spectra/MODULAR_p91_wd=0.6_seed=0.parquet $RUN/wd=0.6/seed=0 --pdf
python -m eigengrok.analysis.make_figures reports/spectra/MODULAR_p91_wd=1_seed=0.parquet $RUN/wd=1/seed=0
    # reports/figures/gate_p91_wd{0.6,1}_seed0{.png,_robustness.png,_summary.md} (+ .pdf with --pdf)
python -m eigengrok.analysis.effective_rank_compare    # reports/figures/effective_rank_compare.{png,pdf}
python -m eigengrok.analysis.aggregate_effective_rank  # reports/figures/effective_rank_multiseed{.png,.pdf,_summary.md}
```

These need a GPU and the run's checkpoints:

```bash
python scripts/spectral_from_checkpoints.py <run_dir> --steps 100:15000:100 --out reports/spectra/<name>.parquet
python scripts/validate_hessian_real.py --out-dir reports   # reports/hessian_validation.{md,csv}
python scripts/bench_hessian.py --out reports/hessian_benchmark.md
```

`reports/wd_sweep_accuracy_curves.csv` (all 240 runs) and
`reports/wd_sweep_norms.csv` (weight, last-layer and gradient norms per logged
step) are extracted from the run directories with
`python scripts/build_wd_sweep_accuracy_csv.py results`. The norms file covers
158 runs: the other 82 run directories were deleted from scratch storage before
their norms were extracted. The script never overwrites a file with fewer runs
than it already holds.
The paper builds with `cd paper && pdflatex grokking.tex` (twice, for references).

## Train

Every run is a YAML config plus optional overrides:

```bash
python -m eigengrok.train --config configs/mod91_wd_sweep.yaml --set weight_decay=0.6 seed=3
```

The resolved config is saved as `config_full.json` in the run directory
(layout in `eigengrok/outputs.py`). `EIGENGROK_OUTPUT_DIR` and
`EIGENGROK_DATA_DIR` set the default output and dataset roots.

On SLURM, activate the environment, then submit from the repo root (jobs inherit
it; alternatively set `EIGENGROK_VENV=/path/to/venv`):

```bash
mkdir -p logs
sbatch --export="ALL,CONFIG=configs/smoke.yaml" scripts/run_gpu.sh
```

## Layout

```
eigengrok/
  config.py      ExperimentConfig (YAML in, config_full.json out)
  data/          dataset builders (modular arithmetic, torchvision)
  models/        MLP, transformer
  tasks/         task = data + model + loss
  train/         training loop, CLI (python -m eigengrok.train)
  hessian/       HVP / Gauss-Newton operators, Lanczos, SLQ spectral statistics
  analysis/      grokking milestones (milestones.py), bootstrap intervals, report figures
  plotting/      per-run training-curve plots
  outputs.py     run-directory layout, checkpoints, npz logs
configs/         YAML experiment configs
scripts/         SLURM launchers and post-hoc analysis scripts
tests/           pytest suite
reports/         sweep summary, validation notes, spectral parquets, committed figures
paper/           LaTeX write-up
```
