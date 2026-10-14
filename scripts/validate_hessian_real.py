#!/usr/bin/env python3
# scripts/validate_hessian_real.py

"""
SLQ accuracy on a real trained network (p=91 modular addition, wd=1, seed 0
checkpoints), complementing the exact-Hessian tests in tests/test_hessian.py,
which can only use tiny models.

1. Continuity. reports/spectra/MODULAR_p91_wd=1_seed=0.parquet was produced
   by the previous implementation (m=100, k=100, Hessian on the first 512
   training points). Rerunning the current code with those settings and the
   same probes must reproduce its trace, top_eig, effective_rank and
   negative_mass.

2. Settings vs. reference. On checkpoints spread over the run (Hessian on the
   full training set, as now), each (m, k) setting is compared against a
   high-accuracy reference (more probes, deeper Lanczos, independent probe
   seed): median relative error per statistic, and the Spearman correlation
   between the setting's trajectory over training and the reference's. The
   trajectory is what downstream forecasting uses; with the probes fixed for
   a whole run, per-snapshot error is strongly correlated across steps.

    python scripts/validate_hessian_real.py --out-dir reports
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from eigengrok.config import ExperimentConfig
from eigengrok.hessian.operators import CurvatureOperators
from eigengrok.hessian.snapshot import fixed_eval_batch
from eigengrok.hessian.spectrum import rademacher_probes, slq, spectral_stats
from eigengrok.tasks.registry import get_task

RUN_DIR = Path("results/MODULAR/no_spectrum/modulus=91/depth=1/width=128/init=1/wd=1/seed=0")
OLD_PARQUET = Path("reports/spectra/MODULAR_p91_wd=1_seed=0.parquet")
CONTINUITY_STEPS = (100, 2400, 6000, 15000)
CONTINUITY_STATS = ("trace", "top_eig", "effective_rank", "negative_mass")
CONTINUITY_MK = (100, 100)  # the previous implementation's production (m, k)

SETTINGS = [(20, 8), (50, 4), (50, 8), (50, 16), (50, 32), (100, 8)]  # (m, k)
REFERENCE = (150, 128)
REFERENCE_SEED = 1000
PROBE_BATCH = 32
COMPARED_STATS = ("top_eig", "trace", "effective_rank", "spectral_entropy", "negative_mass", "n_outliers",
                  "participation_ratio", "top_over_trace", "stable_rank", "moment_2")


def _load(device):
    cfg = ExperimentConfig.from_dict(json.loads((RUN_DIR / "config_full.json").read_text()), strict=False)
    task = get_task(cfg)
    model = task.build_model(device).eval()
    train_ds = task.build_data()["train_ds"]
    loss = lambda z, y: task.compute_training_loss(z, y, device)  # noqa: E731
    return cfg, model, train_ds, loss


def _load_step(model, step, device):
    ckpt = torch.load(RUN_DIR / "checkpoints" / f"checkpoint_step{step}.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])


def _stats(model, x, y, loss, *, m, k, seed, device):
    ops = CurvatureOperators(model, x, y, loss)
    q = slq(ops.hvp, ops.n_params, k=k, m=m, seed=seed, device=device, dtype=ops.theta.dtype, probe_batch=PROBE_BATCH)
    # Krylov dimension actually reached: < m would mean a probe stopped early (breakdown).
    return {**spectral_stats(q), "min_lanczos_steps": min(len(n) for n in q.nodes)}


def continuity(device) -> list[str]:
    cfg, model, train_ds, loss = _load(device)
    first = torch.utils.data.Subset(train_ds, range(min(512, len(train_ds))))
    x, y = next(iter(torch.utils.data.DataLoader(first, batch_size=len(first), shuffle=False)))
    x, y = x.to(device).float(), y.to(device)
    old = pd.read_parquet(OLD_PARQUET).set_index("step")

    # The previous code drew its k probes one torch.randint((P,)) call at a time from one generator.
    P = sum(p.numel() for p in model.parameters())
    gen = torch.Generator().manual_seed(cfg.seed)
    k_old = CONTINUITY_MK[1]
    one_at_a_time = torch.stack([torch.randint(0, 2, (P,), generator=gen) * 2 - 1 for _ in range(k_old)]).float()
    same_probes = torch.equal(one_at_a_time, rademacher_probes(k_old, P, seed=cfg.seed, device=torch.device("cpu"),
                                                                dtype=torch.float32))

    lines = [
        "## 1. Continuity with the previous implementation",
        "",
        f"`{OLD_PARQUET}` (previous code) vs. the current code with the same settings "
        f"(m={CONTINUITY_MK[0]}, k={CONTINUITY_MK[1]}, first 512 training points, probe seed {cfg.seed}). "
        f"Probe vectors identical to the previous code's: **{same_probes}**.",
        "",
        "| step | " + " | ".join(f"{s} (old / new)" for s in CONTINUITY_STATS) + " | max rel. diff |",
        "|---|" + "---|" * (len(CONTINUITY_STATS) + 1),
    ]
    for step in CONTINUITY_STEPS:
        _load_step(model, step, device)
        new = _stats(model, x, y, loss, m=CONTINUITY_MK[0], k=CONTINUITY_MK[1], seed=cfg.seed, device=device)
        rel = max(abs(new[s] - old.loc[step, s]) / max(abs(old.loc[step, s]), 1e-12) for s in CONTINUITY_STATS)
        cells = " | ".join(f"{old.loc[step, s]:.4g} / {new[s]:.4g}" for s in CONTINUITY_STATS)
        lines.append(f"| {step} | {cells} | {rel:.1e} |")
    return lines + [""]


def settings_vs_reference(device, steps: list[int], out_dir: Path) -> list[str]:
    cfg, model, train_ds, loss = _load(device)
    x, y = fixed_eval_batch(train_ds, cfg.spectral_max_points, seed=cfg.seed, device=device)
    rows, times = [], {}
    for step in steps:
        _load_step(model, step, device)
        for label, (m, k), seed in [("reference", REFERENCE, REFERENCE_SEED)] + [(f"m={m},k={k}", (m, k), cfg.seed) for m, k in SETTINGS]:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            st = _stats(model, x, y, loss, m=m, k=k, seed=seed, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.setdefault(label, []).append(time.perf_counter() - t0)
            rows.append({"step": step, "setting": label, **st})
        print(f"step {step} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "hessian_validation.csv", index=False)

    ref = df[df.setting == "reference"].set_index("step")
    full_depth = all(
        (df[df.setting == label]["min_lanczos_steps"] == m).all()
        for label, (m, _) in [("reference", REFERENCE)] + [(f"m={m},k={k}", (m, k)) for m, k in SETTINGS]
    )
    lines = [
        "## 2. SLQ settings vs. a high-accuracy reference",
        "",
        f"{len(steps)} checkpoints from step {steps[0]} to {steps[-1]}; Hessian of the training loss on the full "
        f"training set ({len(x)} examples). Reference: m={REFERENCE[0]}, k={REFERENCE[1]}, probe seed "
        f"{REFERENCE_SEED} (independent of the settings' seed {cfg.seed}). Raw values: `hessian_validation.csv`. "
        f"Every probe ran the full m Lanczos steps (no early breakdown): **{full_depth}**.",
        "",
        "Median relative error vs. the reference over checkpoints (lower is better):",
        "",
        "| setting | time / snapshot | " + " | ".join(COMPARED_STATS) + " |",
        "|---|---|" + "---|" * len(COMPARED_STATS),
    ]
    corr_lines = [
        "",
        "Spearman correlation between the setting's trajectory over training and the reference's "
        "(1 = same ordering of checkpoints):",
        "",
        "| setting | " + " | ".join(COMPARED_STATS) + " |",
        "|---|" + "---|" * len(COMPARED_STATS),
    ]
    for label in ["reference"] + [f"m={m},k={k}" for m, k in SETTINGS]:
        sub = df[df.setting == label].set_index("step")
        t = f"{np.median(times[label]):.2f} s"
        if label == "reference":
            lines.append(f"| {label} | {t} | " + " | ".join("--" for _ in COMPARED_STATS) + " |")
            continue
        errs, cors = [], []
        for s in COMPARED_STATS:
            rel = (sub[s] - ref[s]).abs() / ref[s].abs().clip(lower=1e-12)
            errs.append(f"{np.median(rel):.1%}")
            cors.append(f"{spearmanr(sub[s], ref[s]).statistic:.3f}")
        lines.append(f"| {label} | {t} | " + " | ".join(errs) + " |")
        corr_lines.append(f"| {label} | " + " | ".join(cors) + " |")
    ratio_lines = [
        "",
        "Per-checkpoint ratio setting / reference for effective_rank (bias vs. noise: median near 1 means "
        "unbiased; the quartiles and extremes are the snapshot-to-snapshot noise):",
        "",
        "| setting | median | 25% | 75% | min | max |",
        "|---|---|---|---|---|---|",
    ]
    for label in [f"m={m},k={k}" for m, k in SETTINGS]:
        r = df[df.setting == label].set_index("step")["effective_rank"] / ref["effective_rank"]
        ratio_lines.append(f"| {label} | {r.median():.2f} | {r.quantile(0.25):.2f} | {r.quantile(0.75):.2f} | "
                           f"{r.min():.2f} | {r.max():.2f} |")
    ratio_lines += [
        "",
        f"Reference effective_rank spans {ref['effective_rank'].min():.0f} to {ref['effective_rank'].max():.0f} over "
        f"these checkpoints. Reference negative_mass is {ref['negative_mass'].max():g} at every checkpoint: "
        f"min_eig / top_eig never goes below {(ref['min_eig'] / ref['top_eig']).min():.3f}, so the "
        f"0.05 x |top_eig| margin is never reached on this network.",
    ]
    return lines + corr_lines + ratio_lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--n-steps", type=int, default=25, help="Checkpoints for part 2, evenly spread over the run")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    available = sorted(int(p.stem.removeprefix("checkpoint_step")) for p in (RUN_DIR / "checkpoints").glob("*.pt"))
    idx = np.unique(np.linspace(0, len(available) - 1, args.n_steps).round().astype(int))
    steps = [available[i] for i in idx]

    hw = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    lines = [
        "# SLQ accuracy on a real network",
        "",
        f"Generated by `python scripts/validate_hessian_real.py`. Run: `{RUN_DIR}`. Device: {hw}; "
        f"torch {torch.__version__}.",
        "",
    ]
    lines += continuity(device)
    lines += settings_vs_reference(device, steps, args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "hessian_validation.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
