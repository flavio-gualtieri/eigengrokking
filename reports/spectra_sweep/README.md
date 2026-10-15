# reports/spectra_sweep/

Hessian (H) and Gauss-Newton (G) spectral statistics for every run of the
p=91 sweep rerun (`configs/mod91_wd_sweep_beta2_0.98.yaml`), one parquet per
run, written by `scripts/run_spectral_sweep_array.sh`
(`scripts/spectral_from_checkpoints.py --steps all --operators hessian,gauss_newton`).

**Naming:** `MODULAR_p91_init=<init>_wd=<wd>_seed=<seed>.parquet`, with init and
wd formatted as in the run directories (`eigengrok/outputs.py`): `init=1`,
`wd=0.45`, `wd=0.5`, `wd=1`. `eigengrok/analysis/forecast_features.py::load_spectra`
parses these names.

**Rows:** every checkpoint the run kept (`scripts/prune_checkpoints.py`): a
log-spaced grid identical in every run, plus every 100 steps within 0.5-1.5 x
the run's own onset. Phase 3 features use only the log-spaced grid; the onset
window is for descriptive figures.

**Columns:** `step`; the settings `m`, `k`, `probe_seed` (the run's seed, the
same probes at every checkpoint), `n_points` (the Hessian is of the loss on the
whole training set, 2484 examples); `seconds` (wall time for the checkpoint);
every statistic of `eigengrok/hessian/spectrum.py` for H (unprefixed, e.g.
`top_eig`) and G (prefixed `gn_`, e.g. `gn_top_eig`), each with a `_stderr`
column over probes.

**Settings:** m=20, k=20, chosen in `reports/spectral_validation.md`
("Measuring changes between checkpoints").
