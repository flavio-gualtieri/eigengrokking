# reports/spectra/

Post-hoc spectral parquets (`scripts/spectral_from_checkpoints.py`), one
per run, checked in so `analysis/make_figures.py` regenerates the gate
figure from a committed artifact rather than a notebook.

Naming convention: `<task>_p<modulus>_wd=<weight_decay>_seed=<seed>.parquet`,
e.g. `MODULAR_p91_wd=1_seed=0.parquet`. Not enforced by code -- pick
whatever's unambiguous for the run/config the parquet came from, `analysis/make_figures.py` only cares about the path you pass it.

Generate one (production defaults, `configs/base.py`'s `spectral_m=100,
spectral_k=100`; expect several minutes per checkpoint on a full sweep --
see `scripts/run_spectral_from_checkpoints.sh` for the SLURM wrapper):

    python scripts/spectral_from_checkpoints.py \
        <run_dir> --steps 100:15000:100 \
        --out reports/spectra/MODULAR_p91_wd=1_seed=0.parquet

Then the gate figure:

    python -m analysis.make_figures \
        reports/spectra/MODULAR_p91_wd=1_seed=0.parquet <run_dir>

`<run_dir>` is the same run's base directory (contains `checkpoints/`,
`config_full.json`, `figures/training_data_*.npz`) -- see
`project_io/dir_making.py` for the layout.
