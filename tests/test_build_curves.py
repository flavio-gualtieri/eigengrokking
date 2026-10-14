# tests/test_build_curves.py

"""scripts/build_wd_sweep_accuracy_csv.py: curves out, and no run lost when run directories disappear."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_wd_sweep_accuracy_csv.py"
spec = importlib.util.spec_from_file_location("build_wd_sweep_accuracy_csv", SCRIPT)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def _fake_run(results: Path, seed: int, *, grad_norms: bool) -> Path:
    run = results / f"MODULAR/no_spectrum/modulus=91/depth=1/width=128/init=1/wd=0.5/seed={seed}"
    (run / "figures").mkdir(parents=True)
    steps = np.array([1, 100, 200])
    data = dict(train_steps=steps, test_steps=steps, train_accuracies=[0.1, 0.9, 1.0],
                test_accuracies=[0.0, 0.2, 0.6], train_losses=[4.0, 1.0, 0.1], test_losses=[4.5, 3.0, 1.0],
                norms=np.float32([60.5, 58.25, 57.125]).astype(float), last_layer_norms=[5.0, 6.0, 7.0])
    if grad_norms:
        data["grad_norms"] = [1.5, 0.5, 0.25]
    np.savez(run / "figures" / "training_data_x.npz", **data)
    return run


def _read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_curves_and_norms_are_written(tmp_path):
    results = tmp_path / "results"
    _fake_run(results, 0, grad_norms=False)
    _fake_run(results, 1, grad_norms=True)
    acc_rows, norm_rows = build.collect(results)
    assert build.write_guarded(tmp_path / "acc.csv", build.ACCURACY_COLUMNS, acc_rows, allow_fewer_runs=False)
    assert build.write_guarded(tmp_path / "norms.csv", build.NORM_COLUMNS, norm_rows, allow_fewer_runs=False)

    acc, norms = _read(tmp_path / "acc.csv"), _read(tmp_path / "norms.csv")
    assert len(acc) == len(norms) == 6
    assert [r["test_accuracy"] for r in acc[:3]] == ["0.0", "0.2", "0.6"]
    assert [r["norm"] for r in norms[:3]] == ["60.5", "58.25", "57.125"]
    assert [r["grad_norm"] for r in norms] == ["", "", "", "1.5", "0.5", "0.25"]


def test_existing_output_never_loses_runs(tmp_path):
    results = tmp_path / "results"
    _fake_run(results, 0, grad_norms=False)
    gone = _fake_run(results, 1, grad_norms=False)
    out = tmp_path / "acc.csv"
    acc_rows, _ = build.collect(results)
    build.write_guarded(out, build.ACCURACY_COLUMNS, acc_rows, allow_fewer_runs=False)
    before = out.read_bytes()

    (gone / "figures" / "training_data_x.npz").unlink()  # a purged run directory
    acc_rows, _ = build.collect(results)
    assert not build.write_guarded(out, build.ACCURACY_COLUMNS, acc_rows, allow_fewer_runs=False)
    assert out.read_bytes() == before

    assert build.write_guarded(out, build.ACCURACY_COLUMNS, acc_rows, allow_fewer_runs=True)
    assert len(_read(out)) == 3
