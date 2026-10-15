#!/usr/bin/env python3
# scripts/validate_spectral_changes.py

"""
How precisely does SLQ measure the CHANGE of a spectral statistic between two
checkpoints of one run? Every checkpoint of a run is analysed with the same
Rademacher probes (probe seed = the run's seed), so between nearby
checkpoints most of the probe noise is shared and should cancel in the
difference (common random numbers). This measures that against exact
Hessians, to choose the production (m, k) for scripts/spectral_from_checkpoints.py.

Fixture: a 1-layer transformer of the production architecture family, shrunk
so its Hessian can be materialised (p=11, d_model=16, ~2.6k parameters),
trained on modular addition with the production optimizer (AdamW, betas
(0.9, 0.98), wd=1, full batch). States: the trained parameters theta_0;
theta_0 after 100 more training steps (one checkpoint gap of the sweep); and
theta_0 plus random perturbations of known relative size. For each state,
the exact Hessian H and Gauss-Newton matrix G are materialised in float64.
SLQ runs in float32, as in production, over a grid of (m, k) and N_SEEDS
probe seeds.

Per (operator, statistic, perturbation, m, k), over probe seeds, all relative to |s(theta_0)|:
    level error         |s_hat(theta_0) - s(theta_0)|
    change error, same  |(s_hat_1 - s_hat_0) - (s_1 - s_0)|, both estimates with the same probes
    change error, indep the same with independent probes at the two states (the control)
    sign agreement      fraction of seeds where s_hat_1 - s_hat_0 has the sign of s_1 - s_0

    python scripts/validate_spectral_changes.py                # CPU, about an hour
    python scripts/validate_spectral_changes.py --from-csv     # report and choice from the saved CSV

Writes reports/spectral_change_validation.csv and the generated section of
reports/spectral_validation.md between its BEGIN/END markers.
"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from eigengrok.data.modular import N_CTX, get_modular_addition_data
from eigengrok.hessian.operators import CurvatureOperators
from eigengrok.hessian.spectrum import exact_quadrature, slq, spectral_stats
from eigengrok.models.transformer import Transformer

CPU = torch.device("cpu")
STATS = ("top_eig", "trace", "effective_rank", "top_over_trace", "n_outliers", "participation_ratio")
KEY_STATS = ("top_eig", "trace", "effective_rank", "top_over_trace")
OPERATORS = {"H": "hvp", "G": "gauss_newton"}
M_GRID = (10, 20, 30, 50)
K_GRID = (5, 10, 20)
N_SEEDS = 20
CANDIDATES = [(m, k) for m in (20, 30) for k in (10, 20)]  # the production range Phase 2 allows
CHECKPOINT_GAP = "train +100 steps"
CHUNK = 256
BEGIN, END = "<!-- BEGIN generated: scripts/validate_spectral_changes.py -->", "<!-- END generated -->"


def loss_fn(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(z, y)


def build_fixture(*, p: int = 11, train_steps: int = 1500, gap: int = 100, seed: int = 0):
    torch.manual_seed(seed)
    train, test, _ = get_modular_addition_data(modulus=p, train_frac=0.5, seed=seed)
    x, y = train.tensors
    model = Transformer(vocab_size=p + 1, output_dim=p, n_ctx=N_CTX, d_model=16, num_layers=1, num_heads=2,
                        device=CPU, d_mlp=32, activation_fn="GELU").to(torch.float64)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))

    def run(n: int) -> torch.Tensor:
        for _ in range(n):
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        return parameters_to_vector(model.parameters()).detach().clone()

    theta0 = run(train_steps)
    with torch.no_grad():
        acc = {name: float((model(ds.tensors[0]).argmax(1) == ds.tensors[1]).double().mean())
               for name, ds in (("train", train), ("test", test))}
    states = {"base": theta0, f"train +{gap} steps": run(gap)}
    gen = torch.Generator().manual_seed(seed + 1)
    for eps in (1e-3, 1e-2):
        d = torch.randn(theta0.shape, generator=gen, dtype=torch.float64)
        states[f"random {eps:g} x |theta|"] = theta0 + d * (eps * theta0.norm() / d.norm())
    return model, x, y, states, acc


def operators_at(model, theta: torch.Tensor, x, y, dtype: torch.dtype) -> CurvatureOperators:
    m = copy.deepcopy(model).to(dtype)
    vector_to_parameters(theta.to(dtype), m.parameters())
    return CurvatureOperators(m, x, y, loss_fn)


def exact_stats(ops: CurvatureOperators, op_name: str) -> dict[str, float]:
    op, n = getattr(ops, op_name), ops.n_params
    eye = torch.eye(n, dtype=torch.float64)
    A = torch.cat([op(eye[i:i + CHUNK]) for i in range(0, n, CHUNK)])
    A = 0.5 * (A + A.T)
    return spectral_stats(exact_quadrature(torch.linalg.eigvalsh(A).numpy()))


def collect(model, x, y, states) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact_rows, est_rows = [], []
    for state, theta in states.items():
        ops64 = operators_at(model, theta, x, y, torch.float64)
        ops32 = operators_at(model, theta, x, y, torch.float32)
        for op_label, op_name in OPERATORS.items():
            exact_rows.append({"state": state, "operator": op_label, **exact_stats(ops64, op_name)})
            for m in M_GRID:
                for k in K_GRID:
                    for seed in range(N_SEEDS):
                        q = slq(getattr(ops32, op_name), ops32.n_params, k=k, m=m, seed=seed,
                                device=CPU, dtype=torch.float32)
                        est_rows.append({"state": state, "operator": op_label, "m": m, "k": k, "seed": seed,
                                         **spectral_stats(q)})
        print(f"  {state}: done")
    return pd.DataFrame(exact_rows), pd.DataFrame(est_rows)


def metrics(exact: pd.DataFrame, est: pd.DataFrame) -> pd.DataFrame:
    rows = []
    perturbed = [s for s in exact["state"].unique() if s != "base"]
    for op in OPERATORS:
        ex = exact[exact["operator"] == op].set_index("state")
        for state in perturbed:
            for (m, k), g in est[est["operator"] == op].groupby(["m", "k"]):
                e0 = g[g["state"] == "base"].sort_values("seed")
                e1 = g[g["state"] == state].sort_values("seed")
                for stat in STATS:
                    s0, s1 = ex.loc["base", stat], ex.loc[state, stat]
                    a0, a1 = e0[stat].to_numpy(), e1[stat].to_numpy()
                    scale = abs(s0)
                    true = s1 - s0
                    same = np.abs((a1 - a0) - true) / scale
                    indep = np.abs((a1 - np.roll(a0, 1)) - true) / scale  # seed s at theta_1 vs seed s-1 at theta_0
                    level = np.abs(a0 - s0) / scale
                    rows.append({
                        "operator": op, "perturbation": state, "m": m, "k": k, "stat": stat,
                        "exact_value": s0, "exact_change_rel": true / scale,
                        "level_err_median": np.median(level), "level_err_p90": np.quantile(level, 0.9),
                        "change_err_same_median": np.median(same), "change_err_same_p90": np.quantile(same, 0.9),
                        "change_err_indep_median": np.median(indep), "change_err_indep_p90": np.quantile(indep, 0.9),
                        "sign_agreement_same": float(np.mean(np.sign(a1 - a0) == np.sign(true))),
                        "sign_agreement_indep": float(np.mean(np.sign(a1 - np.roll(a0, 1)) == np.sign(true))),
                    })
    return pd.DataFrame(rows)


TOLERANCE = 1.10  # a candidate is as precise as the best if its worst change error is within 10% of it


def choose(table: pd.DataFrame) -> tuple[tuple[int, int], pd.DataFrame]:
    """
    At the one-checkpoint gap, each candidate's worst 90th-percentile same-probe
    change error over the key statistics of H and G; the cheapest candidate (cost
    ~ m * k operator products) within TOLERANCE of the best of them.
    """
    gap = table[(table["perturbation"] == CHECKPOINT_GAP) & table["stat"].isin(KEY_STATS)]
    scores = []
    for m, k in CANDIDATES:
        g = gap[(gap["m"] == m) & (gap["k"] == k)]
        scores.append({"m": m, "k": k, "cost": m * k, "worst_change_err_p90": g["change_err_same_p90"].max(),
                       "worst_sign_agreement": g["sign_agreement_same"].min()})
    scores = pd.DataFrame(scores)
    eligible = scores[scores["worst_change_err_p90"] <= TOLERANCE * scores["worst_change_err_p90"].min()]
    best = eligible.sort_values(["cost", "worst_change_err_p90"]).iloc[0]
    return (int(best["m"]), int(best["k"])), scores


def _pct(x: float) -> str:
    return f"{100 * x:.2g}%" if x < 0.1 else f"{100 * x:.0f}%"


def markdown(table: pd.DataFrame, acc: dict, n_params: int, choice, scores) -> str:
    m0, k0 = choice
    L = [
        BEGIN,
        "",
        "## Measuring changes between checkpoints (common random numbers)",
        "",
        "Generated by `python scripts/validate_spectral_changes.py` (about an hour on CPU; `--from-csv` "
        "regenerates this section from the saved results); every number: `reports/spectral_change_validation.csv`.",
        "",
        "Every checkpoint of a run is analysed with the same probes, so the trajectory of a statistic is what "
        "matters, not its absolute value. Fixture: a 1-layer transformer of the production family shrunk to "
        f"{n_params:,} parameters (p=11, d_model=16), trained with the production optimizer "
        f"(train accuracy {acc['train']:.2f}, test accuracy {acc['test']:.2f} at theta_0). Exact H and G in float64; "
        f"SLQ in float32 as in production, {N_SEEDS} probe seeds per setting. Errors are relative to |statistic| "
        "at theta_0. *Same* = both states with the same probes (production); *indep.* = independent probes.",
        "",
        f"### One checkpoint gap (theta_0 to 100 more training steps), m={m0}, k={k0}",
        "",
        "| operator | statistic | exact change | level error (median) | change error, same probes (median / 90%) "
        "| change error, indep. (median) | sign right, same | sign right, indep. |",
        "|---|---|---|---|---|---|---|---|",
    ]
    sel = table[(table["perturbation"] == CHECKPOINT_GAP) & (table["m"] == m0) & (table["k"] == k0)]
    for _, r in sel.iterrows():
        L.append(f"| {r['operator']} | {r['stat']} | {r['exact_change_rel']:+.2%} | {_pct(r['level_err_median'])} | "
                 f"{_pct(r['change_err_same_median'])} / {_pct(r['change_err_same_p90'])} | "
                 f"{_pct(r['change_err_indep_median'])} | {r['sign_agreement_same']:.0%} | {r['sign_agreement_indep']:.0%} |")
    L += [
        "",
        "### Gain from shared probes, all perturbations",
        "",
        "Median over the grid's (m, k) of (level error) / (same-probe change error), per perturbation: how many "
        "times more precisely a change is measured than the value itself.",
        "",
        "| operator | statistic | " + " | ".join(p for p in table["perturbation"].unique()) + " |",
        "|---|---|" + "---|" * table["perturbation"].nunique(),
    ]
    for (op, stat), g in table.groupby(["operator", "stat"], sort=False):
        cells = []
        for p in table["perturbation"].unique():
            h = g[g["perturbation"] == p]
            cells.append(f"{np.median(h['level_err_median'] / h['change_err_same_median'].clip(lower=1e-300)):.3g}x")
        L.append(f"| {op} | {stat} | " + " | ".join(cells) + " |")
    L += [
        "",
        "### Choice of production settings",
        "",
        "Among the allowed range (m in {20, 30}, k in {10, 20}), at the one-checkpoint gap, over "
        f"{', '.join(KEY_STATS)} for H and G: the worst 90th-percentile same-probe change error, and (for "
        "reference) the worst sign agreement. Cost is m x k operator products per checkpoint and operator.",
        "",
        "| m | k | cost | worst change error (90%) | worst sign agreement |",
        "|---|---|---|---|---|",
    ]
    for _, r in scores.iterrows():
        L.append(f"| {r['m']:.0f} | {r['k']:.0f} | {r['cost']:.0f} | {_pct(r['worst_change_err_p90'])} | "
                 f"{r['worst_sign_agreement']:.0%} |")
    L += [
        "",
        f"**Production setting: m={m0}, k={k0}**: the cheapest candidate whose worst change error is within "
        f"{TOLERANCE - 1:.0%} of the best. The change error is set by k: trace, the worst case, is a pure Monte "
        "Carlo quantity whose error does not depend on m, and m beyond 20 does not measurably improve the "
        "effective rank on this fixture.",
        "",
        "Sign agreement is reported but not used to choose. A first version of this rule ranked by it first and "
        "was degenerate: every candidate scored the same 55%, set by effective rank, whose exact change over "
        "this gap is under 1% (below the estimator's noise), so the sign is unresolvable at any setting.",
        "",
        "Caveat: one ~2.6k-parameter fixture at one training state. The production network has ~220k parameters "
        "and a different spectrum; `reports/hessian_validation.md` covers absolute accuracy on the real network, "
        "and the sweep's own trajectories (stderr over probes) show how noisy the production changes are.",
        "",
        END,
    ]
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-csv", type=Path, default=Path("reports/spectral_change_validation.csv"))
    parser.add_argument("--report", type=Path, default=Path("reports/spectral_validation.md"))
    parser.add_argument("--from-csv", action="store_true", help="Skip the sweep; report from --out-csv")
    args = parser.parse_args()
    torch.set_num_threads(4)

    model, x, y, states, acc = build_fixture()  # deterministic, ~30 s
    n_params = sum(p.numel() for p in model.parameters())
    print(f"fixture: {n_params:,} parameters, accuracy {acc}")
    if args.from_csv:
        table = pd.read_csv(args.out_csv)
    else:
        exact, est = collect(model, x, y, states)
        table = metrics(exact, est)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out_csv, index=False, float_format="%.6g")
    choice, scores = choose(table)
    section = markdown(table, acc, n_params, choice, scores)
    text = args.report.read_text() if args.report.exists() else ""
    pattern = re.compile(re.escape(BEGIN) + ".*?" + re.escape(END), re.S)
    text = pattern.sub(lambda _: section, text) if pattern.search(text) else text.rstrip("\n") + "\n\n" + section + "\n"
    args.report.write_text(text)
    print(f"Wrote {args.out_csv} and the generated section of {args.report}; production choice m={choice[0]}, k={choice[1]}")


if __name__ == "__main__":
    main()
