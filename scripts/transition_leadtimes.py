#!/usr/bin/env python3
# scripts/transition_leadtimes.py

"""
Does any indicator change before the grokking transition starts? For every
run of the sweep rerun and every indicator (curvature of H and G, cheap
training statistics), the changepoint of eigengrok/analysis/changepoints.py,
compared with the run's transition start (test loss leaving its plateau, the
changepoint of test loss found the same way):

    python scripts/transition_leadtimes.py

Writes (see --out-dir, default reports/):
    transition_changepoints.csv          one row per (run, series): changepoint, lead time, fit details
    transition_leadtimes_summary.csv     lead-time distribution per indicator, primary analysis and every
                                         sensitivity variant (column `variant`), with 95% CIs
    transition_leadtimes.md              the write-up: method, headline table, sensitivity, caveats
    figures/transition_leadtimes.{png,pdf}

Intervals: 95% percentile bootstrap over runs, resampled within (init, wd) settings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eigengrok.analysis import changepoints as cp
from eigengrok.analysis.bootstrap import N_BOOT, bootstrap_ci
from eigengrok.analysis.forecast_features import B1_INDICATORS, C_STATS, RUN, load_curves, load_spectra

FAMILIES = (  # (label, indicators), in display order; Okabe-Ito colour and a marker per family
    ("Hessian H", tuple(C_STATS)),
    ("Gauss-Newton G", tuple(f"gn_{s}" for s in C_STATS)),
    ("cheap statistics", tuple(B1_INDICATORS)),
)
FAMILY_STYLE = {"Hessian H": ("#0072B2", "o"), "Gauss-Newton G": ("#D55E00", "s"), "cheap statistics": ("#009E73", "D")}
LABELS = {"top_eig": "top eigenvalue", "trace": "trace", "effective_rank": "effective rank",
          "top_over_trace": "top / trace", "participation_ratio": "participation ratio", "n_outliers": "outlier count",
          "train_loss": "train loss", "norm": "weight norm", "last_layer_norm": "last-layer norm",
          "grad_norm": "gradient norm"}
HEADLINE = ("effective_rank", "gn_effective_rank")
INK, MUTED = "#333333", "#888888"

PRIMARY = "primary"
VARIANTS = {  # name -> keyword arguments of changepoint_table; one knob changed at a time
    PRIMARY: {},
    "spike filter 3 wide": {"despike_width": 3},
    "no spike filter": {"despike_width": 1},
    "episode fraction 1/2": {"fraction": 1 / 2},
    "BIC margin 40": {"min_delta_bic": 40},
    "up to 7 pieces": {"max_segments": 7},
    "steepest fall only": {"direction": -1},
}


def family_of(series: str) -> str:
    return next(label for label, members in FAMILIES if series in members)


def label_of(series: str) -> str:
    return LABELS[series.removeprefix("gn_")]


def null_median_abs_lead(table: pd.DataFrame, n_draws: int = 1000, seed: int = 0) -> float:
    """Median |lead| / t_onset if every changepoint fell uniformly at random in its run's window."""
    ref = table[table["series"] == cp.REFERENCE]
    pos = (ref["ref_step"] / ref["t_onset"]).to_numpy()
    u = np.random.default_rng(seed).uniform(cp.WINDOW_LO, cp.WINDOW_HI, size=(n_draws, pos.size))
    return float(np.median(np.abs(pos[None, :] - u)))


def summarise(table: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """One row per indicator: the lead-time distribution, in steps and in units of the run's t_onset."""
    table = table.assign(setting=table["init"].astype(str) + "/" + table["wd"].astype(str))
    rows = []
    for _, members in FAMILIES:
        for series in members:
            d = table[table["series"] == series]
            if d.empty:
                continue
            lead, frac, strata = d["lead"].to_numpy(float), d["lead_frac"].to_numpy(float), d["setting"].to_numpy()
            ok = np.isfinite(lead)
            before = np.where(ok, (lead > 0).astype(float), np.nan)
            med_ci = bootstrap_ci(frac, np.median, strata=strata, n_boot=n_boot)
            before_ci = bootstrap_ci(before, np.mean, strata=strata, n_boot=n_boot)
            onset_lead = (d["t_onset"] - d["changepoint"]) / d["t_onset"]
            rows.append({
                "series": series, "family": family_of(series), "n_runs": len(d), "n_detected": int(ok.sum()),
                "n_at_window_start": int(d.loc[ok, "at_window_start"].sum()),
                "frac_falling": float((d.loc[ok, "change"] < 0).mean()),
                "median_lead_steps": float(np.nanmedian(lead)),
                "median_lead_frac": float(np.nanmedian(frac)), "median_lead_frac_ci_lo": med_ci[0],
                "median_lead_frac_ci_hi": med_ci[1],
                "q25_lead_frac": float(np.nanquantile(frac, 0.25)), "q75_lead_frac": float(np.nanquantile(frac, 0.75)),
                "frac_before": float(np.nanmean(before)), "frac_before_ci_lo": before_ci[0],
                "frac_before_ci_hi": before_ci[1],
                "frac_same_checkpoint": float((lead[ok] == 0).mean()), "frac_after": float((lead[ok] < 0).mean()),
                "median_abs_lead_frac": float(np.nanmedian(np.abs(frac))),
                "median_lead_frac_in_window": float(np.nanmedian(frac[ok & ~d["at_window_start"].to_numpy(bool)])),
                "median_lead_vs_onset_frac": float(np.nanmedian(onset_lead)),
            })
    return pd.DataFrame(rows)


def reference_summary(table: pd.DataFrame) -> dict:
    ref = table[table["series"] == cp.REFERENCE]
    before_onset = (ref["t_onset"] - ref["ref_step"]) / ref["t_onset"]
    hg = table[table["series"].isin(HEADLINE)].pivot_table(index=RUN, columns="series", values="changepoint")
    hg = hg.join(ref.set_index(RUN)["t_onset"])
    diff = ((hg[HEADLINE[0]] - hg[HEADLINE[1]]) / hg["t_onset"]).dropna()
    return {
        "n_runs": len(ref), "n_detected": int(ref["changepoint"].notna().sum()),
        "n_at_window_start": int(ref["at_window_start"].sum()),
        "before_onset_median": float(before_onset.median()), "before_onset_q25": float(before_onset.quantile(0.25)),
        "before_onset_q75": float(before_onset.quantile(0.75)),
        "before_onset_steps_median": float((ref["t_onset"] - ref["ref_step"]).median()),
        "n_after_onset": int((ref["ref_step"] > ref["t_onset"]).sum()),
        "null_median_abs_lead": null_median_abs_lead(table),
        "hg_median_abs_diff": float(diff.abs().median()), "hg_q25_diff": float(diff.quantile(0.25)),
        "hg_q75_diff": float(diff.quantile(0.75)), "hg_same_within_200": float(
            ((hg[HEADLINE[0]] - hg[HEADLINE[1]]).abs() <= 200).mean()),
    }


# --- figure ---

def figure(table: pd.DataFrame, summary: pd.DataFrame, ref: dict, stem: Path) -> None:
    rows = [s for _, members in FAMILIES for s in members if s in set(summary["series"])]
    gaps = sum(1 for i in range(1, len(rows)) if family_of(rows[i]) != family_of(rows[i - 1]))
    ypos, y = {}, 0.0
    for i, s in enumerate(rows):
        if i and family_of(s) != family_of(rows[i - 1]):
            y += 0.8
        ypos[s] = y
        y += 1
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * (len(rows) + gaps) + 1.9))
    rng = np.random.default_rng(0)
    summ = summary.set_index("series")
    for s in rows:
        color, marker = FAMILY_STYLE[family_of(s)]
        d = table[(table["series"] == s) & table["lead_frac"].notna()]
        jit = ypos[s] + rng.uniform(-0.22, 0.22, len(d))
        cens = d["at_window_start"].to_numpy(bool)
        ax.scatter(d["lead_frac"][~cens], jit[~cens], s=7, color=color, alpha=0.28, linewidths=0, zorder=2)
        ax.scatter(d["lead_frac"][cens], jit[cens], s=9, facecolors="none", edgecolors=color, alpha=0.45,
                   linewidths=0.6, zorder=2)
        r = summ.loc[s]
        ax.errorbar(r["median_lead_frac"], ypos[s], xerr=[[r["median_lead_frac"] - r["median_lead_frac_ci_lo"]],
                                                           [r["median_lead_frac_ci_hi"] - r["median_lead_frac"]]],
                    fmt=marker, color=color, markersize=7, markeredgecolor="white", markeredgewidth=1.2,
                    elinewidth=2, capsize=0, zorder=4)
        ax.text(1.02, ypos[s], f"{r['frac_before']:.0%}", transform=ax.get_yaxis_transform(), va="center",
                fontsize=8, color=INK)
    ax.text(1.02, -1.7, "runs\nfirst", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=MUTED)
    ax.axvline(0, color=INK, linewidth=1)
    onset = -ref["before_onset_median"]
    ax.axvline(onset, color=INK, linewidth=0.9, linestyle="--")
    ax.text(0.01, -1.7, "test loss leaves\nits plateau", fontsize=8, color=INK, va="center", ha="left")
    ax.text(onset - 0.01, -1.7, "onset (test acc. 50%),\nmedian run", fontsize=8, color=INK, va="center", ha="right")
    ax.set_yticks([ypos[s] for s in rows], [label_of(s) for s in rows], fontsize=8.5)
    for label, members in FAMILIES:
        present = [ypos[s] for s in members if s in ypos]
        if present:
            color, _ = FAMILY_STYLE[label]
            ax.text(-0.31, min(present) - 0.55, label, transform=ax.get_yaxis_transform(), fontsize=9,
                    fontweight="bold", color=INK, va="center")
    ax.set_ylim(max(ypos.values()) + 0.6, -2.3)
    ax.set_xlim(-0.66, 0.62)
    ax.set_xlabel("lead time / t_onset  (> 0: the indicator changes before test loss leaves its plateau)")
    ax.grid(True, axis="x", alpha=0.2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.suptitle("When each indicator starts its steepest change, relative to the transition start",
                 fontsize=11, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Dots: one run each (hollow: change already under way at 0.5 x t_onset, a lower bound). "
             "Marker: median with 95% bootstrap CI. 240 runs.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {stem}.png/.pdf")


# --- write-up ---

def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _signed(x: float) -> str:
    return f"{x:+.2f}"


def markdown(summary: pd.DataFrame, ref: dict, n_runs: int) -> str:
    prim = summary[summary["variant"] == PRIMARY].set_index("series")
    h, g = prim.loc["effective_rank"], prim.loc["gn_effective_rank"]
    L = ["# When does each indicator change, relative to the grokking transition?", "",
         "Generated by `python scripts/transition_leadtimes.py`. Method: `eigengrok/analysis/changepoints.py`. "
         "Intervals: 95% bootstrap over runs, resampled within (init, wd) settings.", "",
         "## Method", "",
         f"For each of the {n_runs} runs, each indicator is read at the run's checkpoints in the window "
         f"[{cp.WINDOW_LO:g}, {cp.WINDOW_HI:g}] x t_onset (every 100 steps; every series of a run at the same "
         "steps, so the cheap statistics get no finer time resolution than the curvature). After the forecast "
         f"features' transform (log, log1p or asinh) and removing spikes (a Hampel filter: points more than "
         f"{cp.HAMPEL_SIGMAS:g} robust SDs from their {cp.DESPIKE}-point running median), the series is split "
         f"into at most {cp.MAX_SEGMENTS} linear pieces by exact least squares, with the number of pieces chosen by "
         f"BIC; a change is called if that beats a single line by >= {cp.MIN_DELTA_BIC} in BIC (calibrated by "
         "simulation: a straight line plus white or heavy-tailed noise clears it in about 5% of draws). The "
         "**changepoint** "
         "is where the series' steepest episode starts: the piece with the steepest net slope, extended backwards "
         f"over adjacent pieces moving the same way at >= {cp.STEEP_FRACTION:.2g} of that slope.", "",
         "The **transition start** is the changepoint of test loss, i.e. where it leaves its post-memorization "
         "plateau. **Lead time** = transition start - indicator changepoint, in units of the run's t_onset; "
         "positive means the indicator changes first.", "",
         "## The transition start", "",
         f"Test loss leaves its plateau a median {_pct(ref['before_onset_median'])} of t_onset "
         f"({ref['before_onset_steps_median']:.0f} steps) before onset (interquartile "
         f"{_pct(ref['before_onset_q25'])} to {_pct(ref['before_onset_q75'])}); "
         f"{ref['n_detected']}/{ref['n_runs']} runs have a detected change, {ref['n_at_window_start']} of them "
         f"already falling at 0.5 x t_onset, and in {ref['n_after_onset']} the steepest drop starts after onset "
         "(test loss comes down a staircase there, mostly at init 1 and low weight decay, and a later step is the "
         "steepest).", "",
         "## Headline: the effective-rank collapse", "",
         f"- H effective rank: median lead {_signed(h['median_lead_frac'])} x t_onset "
         f"[{_signed(h['median_lead_frac_ci_lo'])}, {_signed(h['median_lead_frac_ci_hi'])}] "
         f"({h['median_lead_steps']:+.0f} steps), interquartile {_signed(h['q25_lead_frac'])} to "
         f"{_signed(h['q75_lead_frac'])}; it changes first in {_pct(h['frac_before'])} "
         f"[{_pct(h['frac_before_ci_lo'])}, {_pct(h['frac_before_ci_hi'])}] of runs, and is already falling at "
         f"0.5 x t_onset in {h['n_at_window_start']}.",
         f"- G effective rank: median lead {_signed(g['median_lead_frac'])} x t_onset "
         f"[{_signed(g['median_lead_frac_ci_lo'])}, {_signed(g['median_lead_frac_ci_hi'])}] "
         f"({g['median_lead_steps']:+.0f} steps), interquartile {_signed(g['q25_lead_frac'])} to "
         f"{_signed(g['q75_lead_frac'])}; first in {_pct(g['frac_before'])} "
         f"[{_pct(g['frac_before_ci_lo'])}, {_pct(g['frac_before_ci_hi'])}] of runs, already falling in "
         f"{g['n_at_window_start']}.",
         f"- How precise is one changepoint? H and G effective rank move together, yet their changepoints in the "
         f"same run differ by a median {_pct(ref['hg_median_abs_diff'])} of t_onset (interquartile "
         f"{_signed(ref['hg_q25_diff'])} to {_signed(ref['hg_q75_diff'])}); they fall within 200 steps of each other "
         f"in {_pct(ref['hg_same_within_200'])} of runs.",
         f"- For scale: a changepoint placed uniformly at random in the window would sit a median "
         f"{ref['null_median_abs_lead']:.2f} x t_onset from the transition start; H and G effective rank sit "
         f"{h['median_abs_lead_frac']:.2f} and {g['median_abs_lead_frac']:.2f}.", "",
         "## Every indicator, side by side", "",
         "Lead in units of t_onset. *Already falling/rising* counts runs whose steepest episode is under way when "
         "the window opens (0.5 x t_onset): their lead is a lower bound, and a median marked >= is one where those "
         "runs are half or more; *in window* is the median over the other runs. *Falls* is the share of detected "
         "changes that are decreases. *vs onset* measures the lead from t_onset (test accuracy 50%) instead.", "",
         "| family | indicator | changes detected | already under way | falls | median lead [CI] | interquartile | "
         "in window | runs first [CI] | same checkpoint | runs after | vs onset |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, r in prim.iterrows():
        censored = r["n_at_window_start"] >= 0.5 * r["n_detected"]
        L.append(
            f"| {r['family']} | {label_of(s)} | {r['n_detected']}/{r['n_runs']} | {r['n_at_window_start']} | "
            f"{_pct(r['frac_falling'])} | {'>= ' if censored else ''}{_signed(r['median_lead_frac'])} "
            f"[{_signed(r['median_lead_frac_ci_lo'])}, {_signed(r['median_lead_frac_ci_hi'])}] | "
            f"{_signed(r['q25_lead_frac'])} to {_signed(r['q75_lead_frac'])} | "
            f"{_signed(r['median_lead_frac_in_window'])} | {_pct(r['frac_before'])} "
            f"[{_pct(r['frac_before_ci_lo'])}, {_pct(r['frac_before_ci_hi'])}] | {_pct(r['frac_same_checkpoint'])} | "
            f"{_pct(r['frac_after'])} | {_signed(r['median_lead_vs_onset_frac'])} |")

    L += ["", "![Lead times](figures/transition_leadtimes.png)", "",
          "## Sensitivity", "",
          "Median lead (units of t_onset) under each variant of the detector, one setting changed at a time; the "
          "transition start is re-detected under the same variant.", ""]
    piv = summary.pivot(index="series", columns="variant", values="median_lead_frac")
    order = [s for _, members in FAMILIES for s in members if s in piv.index]
    cols = list(VARIANTS)
    L += ["| indicator | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for s in order:
        L.append(f"| {family_of(s)}: {label_of(s)} | " + " | ".join(_signed(piv.loc[s, c]) for c in cols) + " |")

    L += ["", "## Caveats", "",
          "- The window is placed using each run's own onset, so this describes the transition after the fact; it "
          "is not a forecast. Any change more than about 0.45 x t_onset before the transition start is outside the "
          "window and shows up only as *already under way*.",
          "- One changepoint per series. A series with several separate changes in the window is dated by the "
          "steepest. Where test loss comes down a staircase, the transition start is its steepest step, and an "
          "indicator that steps down with an earlier step of the staircase (effective rank often does) shows a "
          "long lead.",
          "- Ties: series are read every 100 steps, so a lead of 0 means the same checkpoint.", ""]
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accuracy-csv", type=Path, default=Path("reports/sweep_beta2_0.98_accuracy_curves.csv.gz"))
    parser.add_argument("--stats-csv", type=Path, default=Path("reports/sweep_beta2_0.98_training_stats.csv.gz"))
    parser.add_argument("--milestones", type=Path, default=Path("reports/sweep_beta2_0.98/sweep_milestones.csv"))
    parser.add_argument("--spectra-dir", type=Path, default=Path("reports/spectra_sweep"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()

    curves = load_curves(args.accuracy_csv, args.stats_csv)
    spectra = load_spectra(args.spectra_dir)
    milestones = pd.read_csv(args.milestones)
    milestones["init"] = milestones["init"].astype(float)
    milestones["wd"] = milestones["wd"].astype(float).round(4)

    summaries, primary = [], None
    for name, kwargs in VARIANTS.items():
        table = cp.changepoint_table(curves, spectra, milestones, **kwargs)
        if name == PRIMARY:
            primary = table
        # the sensitivity table shows the variants' medians only, so their intervals can be rougher
        summaries.append(summarise(table, args.n_boot if name == PRIMARY else 1000).assign(variant=name))
        print(f"{name}: done")
    summary = pd.concat(summaries, ignore_index=True)
    ref = reference_summary(primary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    primary.to_csv(args.out_dir / "transition_changepoints.csv", index=False, float_format="%.6g")
    summary.to_csv(args.out_dir / "transition_leadtimes_summary.csv", index=False, float_format="%.6g")
    n_runs = primary[RUN].drop_duplicates().shape[0]
    (args.out_dir / "transition_leadtimes.md").write_text(markdown(summary, ref, n_runs))
    (args.out_dir / "figures").mkdir(exist_ok=True)
    figure(primary, summary[summary["variant"] == PRIMARY], ref, args.out_dir / "figures" / "transition_leadtimes")
    print(f"Wrote {args.out_dir / 'transition_leadtimes.md'}")


if __name__ == "__main__":
    main()
