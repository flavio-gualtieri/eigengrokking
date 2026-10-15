# eigengrok/analysis/forecast_features.py

"""
Phase 3 feature table: one row per run, every indicator trajectory
summarised only up to a cutoff step, and the target y = ln(onset). The onset
is t_onset (primary) or, for sensitivity analysis, one of the secondary
definitions onset_2x_chance / onset_logistic (eigengrok/analysis/milestones.py).

Cutoffs (`Cutoff`): t_mem + offset steps ("after_mem"), or a fixed step
("absolute"). Nothing after the cutoff is read. A run whose onset is already
at or before its cutoff has nothing left to forecast and is dropped from that
cutoff (counted in the table's attrs). At an absolute cutoff a run may not
have memorized yet: its t_mem feature is then censored at the cutoff and
flagged, never read from the future.

Curvature indicators are read only at the log-spaced checkpoint grid every
run of the sweep keeps at identical steps (eigengrok/analysis/checkpoint_grid.py).
The extra checkpoints kept around each run's own onset are placed using the
label, so they are never used as features.

Families (column prefixes):
    B0  hyperparameters: wd, wd^2, log2(init), wd x log2(init)
    B1  cheap training statistics: train loss, weight norm, last-layer norm,
        gradient norm, and t_mem
    C   curvature: H and Gauss-Newton G (prefix gn_) top_eig, trace,
        effective_rank and the scale-invariant top_over_trace,
        participation_ratio, n_outliers

Per indicator (transformed: log for positive quantities, log1p for counts,
asinh where the sign can change): last value, slope against log10(step)
over the second half of the window up to the cutoff (in log step), min and
max over [t_mem, cutoff], and the change since t_mem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eigengrok.analysis.checkpoint_grid import log_grid
from eigengrok.analysis.milestones import onset_2x_chance, onset_logistic
from eigengrok.analysis.milestones import t_mem as milestone_t_mem
from eigengrok.analysis.milestones import t_onset as milestone_t_onset

RUN = ["init", "wd", "seed"]
TARGETS = ("t_onset", "onset_2x_chance", "onset_logistic")
DEFAULT_MODULUS = 91
SWEEP_CHECKPOINT_STEPS = [1] + list(range(100, 150_001, 100))  # every run of the sweep saves these
SWEEP_LOG_GRID = log_grid(SWEEP_CHECKPOINT_STEPS)              # ...and keeps these, at identical steps

B1_INDICATORS = ("train_loss", "norm", "last_layer_norm", "grad_norm")
C_STATS = ("top_eig", "trace", "effective_rank", "top_over_trace", "participation_ratio", "n_outliers")
C_INDICATORS = tuple(f"{prefix}{s}" for prefix in ("", "gn_") for s in C_STATS)
SUMMARIES = ("last", "slope", "min", "max", "change_since_mem")

# Transform per indicator: log for strictly positive quantities, log1p for counts,
# asinh where the sign can change (the full Hessian's trace, and ratios built on it).
TRANSFORM = {name: "log" for name in (*B1_INDICATORS, "top_eig", "effective_rank", "gn_top_eig", "gn_trace",
                                      "gn_effective_rank", "gn_top_over_trace", "gn_participation_ratio")}
TRANSFORM.update({"n_outliers": "log1p", "gn_n_outliers": "log1p", "trace": "asinh", "top_over_trace": "asinh",
                  "participation_ratio": "asinh"})


@dataclass(frozen=True)
class Cutoff:
    kind: str    # "after_mem" or "absolute"
    value: int   # steps after t_mem, or the step itself

    @property
    def label(self) -> str:
        return f"t_mem+{self.value}" if self.kind == "after_mem" else f"step {self.value}"

    def step(self, t_mem: float) -> float:
        if self.kind == "after_mem":
            return t_mem + self.value
        if self.kind == "absolute":
            return float(self.value)
        raise ValueError(f"Unknown cutoff kind {self.kind!r}")


def _transform(name: str, v: np.ndarray) -> np.ndarray:
    kind = TRANSFORM.get(name, "asinh")
    if kind == "log":
        return np.log(np.maximum(v, 1e-300))
    if kind == "log1p":
        return np.log1p(np.maximum(v, 0))
    return np.arcsinh(v)


def summarise(steps: np.ndarray, values: np.ndarray, *, name: str, cutoff: float, ref_step: float) -> dict[str, float]:
    """The SUMMARIES of one indicator's trajectory, from points with step <= cutoff only."""
    keep = (steps <= cutoff) & np.isfinite(values)
    s, v = steps[keep].astype(float), _transform(name, values[keep].astype(float))
    if s.size == 0:
        return {k: np.nan for k in SUMMARIES}
    order = np.argsort(s)
    s, v = s[order], v[order]
    logs = np.log10(np.maximum(s, 1.0))

    tail = logs >= logs[-1] - 0.5 * (logs[-1] - logs[0])  # second half of the window, in log step
    if tail.sum() < 2:
        tail[-min(2, s.size):] = True
    slope = np.polyfit(logs[tail], v[tail], 1)[0] if tail.sum() >= 2 and np.ptp(logs[tail]) > 0 else np.nan

    window = s >= min(ref_step, s[-1])
    ref = v[np.searchsorted(s, min(ref_step, s[-1]), side="right") - 1] if s[0] <= ref_step else v[0]
    return {"last": v[-1], "slope": slope, "min": v[window].min(), "max": v[window].max(),
            "change_since_mem": v[-1] - ref}


def run_targets(curves: pd.DataFrame, target: str = "t_onset") -> pd.DataFrame:
    """t_mem, every onset definition in TARGETS, and y = ln(`target`), one row per run."""
    if target not in TARGETS:
        raise ValueError(f"Unknown target {target!r}; options: {TARGETS}")
    rows = []
    for key, g in curves.groupby(RUN, sort=True):
        g = g.sort_values("step")
        modulus = int(g["modulus"].iloc[0]) if "modulus" in g else DEFAULT_MODULUS
        steps, test = g["step"].to_numpy(), g["test_accuracy"].to_numpy()
        cross = onset_2x_chance(steps, test, modulus)
        rows.append({**dict(zip(RUN, key)), "t_mem": milestone_t_mem(steps, g["train_accuracy"]),
                     "t_onset": milestone_t_onset(steps, test), "onset_2x_chance": cross,
                     "onset_logistic": onset_logistic(steps, test, init_step=cross).fit_step if target == "onset_logistic"
                     else np.nan})
    out = pd.DataFrame(rows)
    out["y"] = np.log(out[target])
    return out


def build_features(curves: pd.DataFrame, spectra: pd.DataFrame | None, cutoff: Cutoff,
                   target: str = "t_onset") -> pd.DataFrame:
    """
    One row per run with RUN keys, t_mem, t_onset, y, cutoff_step and the
    feature columns "<family>:<indicator>:<summary>" (B0 columns are
    "B0:<name>"). Runs already past onset at the cutoff are dropped; their
    count is in attrs["n_dropped"].
    """
    targets = run_targets(curves, target)
    curves_by_run = dict(tuple(curves.groupby(RUN, sort=False)))
    spectra_by_run = dict(tuple(spectra.groupby(RUN, sort=False))) if spectra is not None and len(spectra) else {}
    grid = np.array(SWEEP_LOG_GRID)

    rows, dropped = [], 0
    for _, t in targets.iterrows():
        key = tuple(t[k] for k in RUN)
        mem = t["t_mem"] if np.isfinite(t["t_mem"]) else np.inf
        cut = cutoff.step(mem) if cutoff.kind == "absolute" or np.isfinite(mem) else np.nan
        if not np.isfinite(cut) or not np.isfinite(t[target]) or t[target] <= cut:
            dropped += 1
            continue
        mem_seen = min(mem, cut)
        log2_init = np.log2(t["init"])
        row = {**dict(zip(RUN, key)), "t_mem": t["t_mem"], "t_onset": t["t_onset"], "target": target,
               "target_step": t[target], "y": t["y"], "cutoff_step": cut,
               "B0:wd": t["wd"], "B0:wd2": t["wd"] ** 2, "B0:log2_init": log2_init,
               "B0:wd_x_log2_init": t["wd"] * log2_init,
               "B1:t_mem": np.log(mem_seen), "B1:memorized_by_cutoff": float(mem <= cut)}

        g = curves_by_run[key].sort_values("step")
        for ind in B1_INDICATORS:
            if ind in g:
                for name, val in summarise(g["step"].to_numpy(), g[ind].to_numpy(), name=ind, cutoff=cut,
                                           ref_step=mem_seen).items():
                    row[f"B1:{ind}:{name}"] = val

        sp = spectra_by_run.get(key)
        if sp is not None:
            sp = sp[sp["step"].isin(grid)]
            for ind in C_INDICATORS:
                if ind in sp:
                    for name, val in summarise(sp["step"].to_numpy(), sp[ind].to_numpy(), name=ind, cutoff=cut,
                                               ref_step=mem_seen).items():
                        row[f"C:{ind}:{name}"] = val
        rows.append(row)

    table = pd.DataFrame(rows)
    table.attrs.update(cutoff=cutoff.label, target=target, n_dropped=dropped, n_runs=len(targets))
    return table


def family_columns(table: pd.DataFrame, families: str) -> list[str]:
    """Feature columns of a '+'-joined family list, e.g. 'B0+B1+C'."""
    wanted = families.split("+")
    return [c for c in table.columns if c.split(":")[0] in wanted]


# --- loading the repo's data ---

_SPECTRA_NAME = re.compile(r"init=(?P<init>[\d.]+)_wd=(?P<wd>[\d.]+)_seed=(?P<seed>\d+)\.parquet$")


def _normalise_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["init"] = df["init"].astype(float)
    df["wd"] = df["wd"].astype(float).round(4)
    df["seed"] = df["seed"].astype(int)
    return df


def load_curves(accuracy_csv: Path, stats_csv: Path | None = None) -> pd.DataFrame:
    """Accuracy/loss curves joined with the training statistics (norms etc.), long format."""
    curves = _normalise_keys(pd.read_csv(accuracy_csv))
    if stats_csv is not None and Path(stats_csv).exists():
        stats = _normalise_keys(pd.read_csv(stats_csv)).drop(columns=["modulus", "depth", "width"], errors="ignore")
        curves = curves.merge(stats, on=RUN + ["step"], how="left", validate="one_to_one")
    return curves


def load_spectra(spectra_dir: Path) -> pd.DataFrame:
    """Every per-run spectral parquet in `spectra_dir`, long format with RUN keys from the file names."""
    frames = []
    for path in sorted(Path(spectra_dir).glob("*.parquet")):
        m = _SPECTRA_NAME.search(path.name)
        if m:
            frames.append(pd.read_parquet(path).assign(init=float(m["init"]), wd=float(m["wd"]), seed=int(m["seed"])))
    return _normalise_keys(pd.concat(frames)) if frames else pd.DataFrame(columns=RUN + ["step"])
