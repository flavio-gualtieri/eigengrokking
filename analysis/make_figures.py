#!/usr/bin/env python3
# analysis/make_figures.py

"""
The "gate" figure: does Hessian trace/effective_rank move independently of
weight-norm shrinkage around the grokking transition, or is the pre-onset
rise just ||theta||^-2?

Regenerates entirely from a committed parquet (scripts/spectral_from_checkpoints.py)
plus the run's training-curve npz -- no notebook, one command:

    python -m analysis.make_figures \\
        reports/spectra/modulus=91_wd=1_seed=0.parquet \\
        runs_arch_sweep/MODULAR/no_spectrum/modulus=91/depth=1/width=128/init=1/wd=1/seed=0

Nothing here hardcodes p=91 or a step range: `modulus` is read off the
run's config_full.json (override with --modulus for a non-MODULAR task),
and the x-axis defaults to the parquet's own step range. Re-pointing this
at a different p's parquet/run_dir needs no code change -- see
analysis/transitions.py's module docstring.

Outputs (see --out-dir, default reports/figures/):
    gate_<tag>.png              the two-panel gate figure (step 5)
    gate_<tag>_robustness.png   trace_corr_rel under two alternate normalizers (step 6)
    gate_<tag>_summary.md       onset numbers, sanity checks, the headline
                                 sentence, and a limitations note -- all
                                 computed, not hand-maintained (step 7)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.transitions import (
    DEFAULT_ACCURACY_CSV,
    TRAIN_ACC_LOW_THRESHOLD,
    Onset,
    add_corrected_quantities,
    add_robustness_variants,
    find_onset,
    load_and_join,
    load_modulus,
    load_train_accuracy,
    sanity_checks,
)

# Okabe-Ito: colorblind-safe, print-safe, the standard choice for scientific
# figures with a handful of fixed-identity series (never reassigned by
# filtering/order -- see the dataviz skill's categorical-color rule).
COLOR_TEST_ACC = "#000000"
COLOR_WNORM = "#0072B2"
COLOR_LL_NORM = "#56B4E9"
COLOR_TRACE = "#0072B2"
COLOR_TRACE_CORR = "#D55E00"
COLOR_TOP_CORR = "#009E73"
COLOR_EFF_RANK = "#CC79A7"
COLOR_ONSET = "#666666"


def _errorevery(n: int, target: int = 20) -> int:
    return max(1, n // target)


def _mark_onset(ax, onset: Onset, *, label: bool = True) -> None:
    ax.axvline(onset.cross_step, color=COLOR_ONSET, linestyle="-", linewidth=1, alpha=0.8,
               label=f"onset (2x chance) @ step {onset.cross_step}" if label else None)
    if not onset.agree:
        ax.axvline(onset.fit_step, color=COLOR_ONSET, linestyle=":", linewidth=1, alpha=0.8,
                    label=f"onset (logistic fit) @ step {onset.fit_step:.0f}" if label else None)


def _mark_low_train_acc(ax, df: pd.DataFrame, col: str, color: str, *, label: str = None) -> None:
    """
    Overlays hollow (open) markers on the points flagged `train_acc_low` by
    load_train_accuracy() -- the line itself already carries every point;
    this only adds a fill-state cue, so identity stays with color/line as
    everywhere else in this figure and reliability rides on top of it,
    never replacing it. A no-op if the join wasn't done or nothing's flagged.
    """
    mask = df.get("train_acc_low")
    if mask is None or not mask.any():
        return
    ax.scatter(df.loc[mask, "step"], df.loc[mask, col], facecolors="none", edgecolors=color,
               marker="o", s=40, linewidths=1.3, zorder=5, label=label)


def plot_gate_figure(df: pd.DataFrame, onset: Onset, *, title: str,
                      x_min: float = None, x_max: float = None) -> plt.Figure:
    """Step 5: the two-panel gate figure -- context on top, the corrected result below."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]},
    )

    # --- top: context ---
    ax_top.plot(df["step"], df["test_acc"], color=COLOR_TEST_ACC, linewidth=2, label="test accuracy")
    ax_top.axhline(onset.chance, color=COLOR_TEST_ACC, linestyle="--", linewidth=1, alpha=0.5,
                    label=f"chance (1/p = {onset.chance:.3f})")
    ax_top.set_ylabel("Test accuracy")
    ax_top.set_ylim(-0.02, 1.02)

    ax_norm = ax_top.twinx()
    ax_norm.plot(df["step"], df["wnorm"], color=COLOR_WNORM, linewidth=1.5, label="||theta|| (total)")
    ax_norm.plot(df["step"], df["ll_norm"], color=COLOR_LL_NORM, linewidth=1.5, linestyle="--",
                 label="||theta|| (last layer)")
    ax_norm.set_ylabel("Weight norm")

    _mark_onset(ax_top, onset)
    lines_top = ax_top.get_lines() + ax_norm.get_lines()
    ax_top.legend(lines_top, [l.get_label() for l in lines_top], loc="center right", fontsize=8)
    ax_top.set_title(title)

    # --- bottom: the result, all relative to the reference step, log-log ---
    n = len(df)
    ee = _errorevery(n)

    ax_bot.errorbar(df["step"], df["trace_rel"], yerr=df.get("trace_rel_stderr"),
                     color=COLOR_TRACE, linewidth=1.5, errorevery=ee, capsize=2,
                     label="trace_rel (raw)")
    ax_bot.plot(df["step"], df["wnorm_pred"], color=COLOR_TRACE, linewidth=1.5, linestyle="--",
                label="||theta||^-2 prediction")
    ax_bot.errorbar(df["step"], df["trace_corr_rel"], yerr=df.get("trace_corr_rel_stderr"),
                     color=COLOR_TRACE_CORR, linewidth=2.8, errorevery=ee, capsize=2,
                     label="trace_corr_rel (norm-corrected)")
    ax_bot.errorbar(df["step"], df["top_corr_rel"], yerr=df.get("top_corr_rel_stderr"),
                     color=COLOR_TOP_CORR, linewidth=1.5, errorevery=ee, capsize=2,
                     label="top_eig_corr_rel")
    ax_bot.errorbar(df["step"], df["effective_rank_rel"], yerr=df.get("effective_rank_rel_stderr"),
                     color=COLOR_EFF_RANK, linewidth=1.5, errorevery=ee, capsize=2,
                     label="effective_rank_rel")

    thresh = df.attrs.get("train_acc_low_thresh")
    low_label = f"train_acc < {thresh:g} (unreliable checkpoint)" if thresh is not None else None
    _mark_low_train_acc(ax_bot, df, "trace_rel", COLOR_TRACE)
    _mark_low_train_acc(ax_bot, df, "trace_corr_rel", COLOR_TRACE_CORR)
    _mark_low_train_acc(ax_bot, df, "top_corr_rel", COLOR_TOP_CORR)
    _mark_low_train_acc(ax_bot, df, "effective_rank_rel", COLOR_EFF_RANK, label=low_label)

    ax_bot.axhline(1.0, color="black", linewidth=0.75, alpha=0.4)
    _mark_onset(ax_bot, onset, label=False)
    ax_bot.set_yscale("log")
    ax_bot.set_xlabel("Optimization step")
    ax_bot.set_ylabel(f"Relative to step {df.attrs.get('ref_step', df['step'].iloc[0])} (log)")
    ax_bot.legend(loc="best", fontsize=8)

    ax_bot.set_xscale("log")
    lo = x_min if x_min is not None else df["step"].min()
    hi = x_max if x_max is not None else df["step"].max()
    ax_bot.set_xlim(lo, hi)

    fig.tight_layout()
    return fig


def plot_robustness_figure(df: pd.DataFrame, onset: Onset, *, title: str,
                            x_min: float = None, x_max: float = None) -> plt.Figure:
    """
    Step 6: the same bottom-panel story, redrawn under two alternative
    normalizers, stacked for direct visual comparison against the headline
    ||theta||^-2 correction.
    """
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    n = len(df)
    ee = _errorevery(n)

    specs = [
        ("trace_corr_rel", COLOR_TRACE_CORR, "||theta||^-2 corrected (headline)"),
        ("trace_over_top_rel", COLOR_TOP_CORR, "trace_rel / top_eig_rel (norm-free)"),
        ("trace_llcorr_rel", COLOR_LL_NORM, "last-layer-norm corrected"),
    ]
    thresh = df.attrs.get("train_acc_low_thresh")
    low_label = f"train_acc < {thresh:g} (unreliable checkpoint)" if thresh is not None else None
    for ax, (col, color, label) in zip(axes, specs):
        stderr_col = f"{col}_stderr"
        ax.errorbar(df["step"], df[col], yerr=df.get(stderr_col), color=color, linewidth=2,
                     errorevery=ee, capsize=2, label=label)
        _mark_low_train_acc(ax, df, col, color, label=(low_label if ax is axes[0] else None))
        ax.axhline(1.0, color="black", linewidth=0.75, alpha=0.4)
        _mark_onset(ax, onset, label=(ax is axes[0]))
        ax.set_yscale("log")
        ax.set_ylabel(col)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Optimization step")
    axes[0].set_title(f"Robustness: {title}")
    axes[-1].set_xscale("log")
    lo = x_min if x_min is not None else df["step"].min()
    hi = x_max if x_max is not None else df["step"].max()
    axes[-1].set_xlim(lo, hi)

    fig.tight_layout()
    return fig


def _classify_pre_onset_movement(df: pd.DataFrame, onset: Onset, *, flat_tol: float = 0.10,
                                  lead_frac: float = 0.30) -> str:
    """
    A rough, numeric classification of which of the three reportable
    outcomes (step 7 of the spec) this run landed in -- drives the
    auto-generated headline sentence. Always inspect the figure; this is a
    starting point for the wording, not a substitute for reading the plot.
    """
    ref = df.attrs["ref_index"]
    log_rel = np.log(df["trace_corr_rel"])
    total_change = float(log_rel.iloc[-1] - log_rel.iloc[ref])

    pre = df[df["step"] <= onset.cross_step]
    if len(pre) < 2:
        return "insufficient_pre_onset_data"
    pre_change = float(np.log(pre["trace_corr_rel"]).iloc[-1] - np.log(pre["trace_corr_rel"]).iloc[0])

    if abs(total_change) < np.log(1 + flat_tol):
        return "flat"
    if abs(total_change) > 0 and abs(pre_change / total_change) > lead_frac:
        return "leads_onset"
    return "tracks_then_departs"


def _pct_change(series: pd.Series, i0: int, i1: int) -> float:
    return 100.0 * (series.iloc[i1] / series.iloc[i0] - 1.0)


def write_summary(
    df: pd.DataFrame, onset: Onset, sanity: dict, *, tag: str, run_dir: Path, parquet_path: Path,
    out_path: Path, robustness_consistent: bool,
) -> None:
    """Step 7: the README sentence, computed off real numbers, plus a limitations note (non-optional)."""
    ref = df.attrs["ref_index"]
    ref_step = df.attrs["ref_step"]
    classification = _classify_pre_onset_movement(df, onset)

    onset_report_step = onset.cross_step
    idx_onset = int((df["step"] - onset_report_step).abs().idxmin())
    idx_last = len(df) - 1

    pre_pct = _pct_change(df["trace_corr_rel"], ref, idx_onset)
    post_trace_pct = _pct_change(df["trace_corr_rel"], idx_onset, idx_last)
    top_mult = df["top_corr_rel"].iloc[idx_last] / df["top_corr_rel"].iloc[idx_onset]
    rank_div = df["effective_rank_rel"].iloc[idx_onset] / df["effective_rank_rel"].iloc[idx_last]

    if classification == "tracks_then_departs":
        headline = (
            f"Across steps {ref_step}-{onset_report_step} the corrected trace changes by "
            f"{pre_pct:+.1f}% (norm-corrected, i.e. within noise of the ||theta||^-2 prediction); "
            f"from step {onset_report_step} the corrected trace and the raw prediction diverge, "
            f"with corrected trace moving {post_trace_pct:+.1f}% while top_eig_corr rises "
            f"{top_mult:.1f}x and effective_rank falls {rank_div:.1f}x."
        )
    elif classification == "leads_onset":
        headline = (
            f"corrected trace moves {pre_pct:+.1f}% before the onset step ({onset_report_step}), "
            f"a lead-time candidate -- ahead of the accuracy transition rather than concurrent "
            f"with it. From onset to the end of this window it moves a further {post_trace_pct:+.1f}%, "
            f"alongside top_eig_corr rising {top_mult:.1f}x and effective_rank falling {rank_div:.1f}x."
        )
    elif classification == "flat":
        headline = (
            f"corrected trace stays flat throughout this window ({pre_pct:+.1f}% pre-onset, "
            f"{post_trace_pct:+.1f}% post-onset) -- no independent curvature signal in trace at this "
            f"normalization. top_eig_corr (x{top_mult:.1f}) and effective_rank (/{rank_div:.1f}) still "
            f"show concurrent sharpening."
        )
    else:
        headline = "Not enough pre-onset data in this parquet to classify the trace's pre-onset behavior."

    lines = [
        f"# Gate figure summary: {tag}",
        "",
        f"Generated by `python -m analysis.make_figures {parquet_path} {run_dir}`.",
        "",
        "## Onset",
        "",
        f"- 2x-chance crossing: step {onset.cross_step}",
        f"- Logistic-fit landmark (exp(mu - 2 sigma)): step {onset.fit_step:.1f}",
        f"- Agree within 1.2x: {onset.agree}"
        + ("" if onset.agree else " -- **inspect the accuracy curve before trusting the fit value.**"),
        "",
        "## Sanity checks",
        "",
        f"- Pre-onset log-residual (trace_rel vs. ||theta||^-2 prediction), all {sanity['n_total']} "
        f"checkpoints: mean {sanity['all']['pre_onset_log_residual_mean']:.3f}, "
        f"sd {sanity['all']['pre_onset_log_residual_std']:.3f}, "
        f"range {sanity['all']['pre_onset_log_residual_range']:.3f}",
        f"- Smoothness (mean |d(log)| per step), all checkpoints: trace_rel="
        f"{sanity['all']['mean_abs_dlog_trace_rel']:.4f}, trace_corr_rel="
        f"{sanity['all']['mean_abs_dlog_trace_corr_rel']:.4f}",
    ]
    thresh = df.attrs.get("train_acc_low_thresh", TRAIN_ACC_LOW_THRESHOLD)
    if sanity["n_excluded"]:
        n_remain = sanity["n_total"] - sanity["n_excluded"]
        excl = sanity["excl_train_acc_low"]
        lines += [
            f"- Same two checks with the {sanity['n_excluded']} checkpoint(s) below "
            f"train_acc={thresh:g} dropped ({n_remain} remain): pre-onset log-residual "
            f"mean {excl['pre_onset_log_residual_mean']:.3f}, sd {excl['pre_onset_log_residual_std']:.3f}, "
            f"range {excl['pre_onset_log_residual_range']:.3f}; smoothness trace_rel="
            f"{excl['mean_abs_dlog_trace_rel']:.4f}, trace_corr_rel={excl['mean_abs_dlog_trace_corr_rel']:.4f}.",
        ]
    else:
        lines.append(
            f"- No checkpoints fell below train_acc={thresh:g} in this window -- masked/unmasked "
            "stats above coincide."
        )
    lines += [
        "",
        "## Headline",
        "",
        headline,
        "",
        "## Limitations",
        "",
    ]
    if robustness_consistent:
        lines.append(
            "The qualitative conclusion above (see `*_robustness.png`) survives both alternative "
            "normalizers (trace/top_eig and last-layer-norm-corrected trace) -- not a normalization "
            "artifact of the ||theta||^-2 choice specifically."
        )
    else:
        lines.append(
            "**The qualitative conclusion above did not survive both alternative normalizers** "
            "(see `*_robustness.png`) -- treat the pre-onset window as inconclusive rather than as "
            "evidence either way. The ||theta||^-2 correction is a heuristic, not validated against an "
            "independent normalizer for this run."
        )
    lines += [
        "",
        "`bulk_edge`/`conditioning` are excluded from this figure -- the weighted median sits inside "
        "the near-zero degenerate spike and the sign is noise-determined; see "
        "\"Demoted from headline figures/claims\" in `reports/spectral_validation.md`.",
        "",
    ]

    out_path.write_text("\n".join(lines))


def _robustness_consistent(df: pd.DataFrame, onset: Onset, *, flat_tol: float = 0.10) -> bool:
    """
    A rough automated check for step 6: do all three normalizers agree on
    "materially flat pre-onset" vs. "materially moving pre-onset"? Anything
    subtler than that binary should be read off the robustness figure by
    eye, not trusted to this heuristic.
    """
    ref = df.attrs["ref_index"]
    cross_idx = int((df["step"] - onset.cross_step).abs().idxmin())

    def moved(col: str) -> bool:
        pre_change = np.log(df[col].iloc[cross_idx]) - np.log(df[col].iloc[ref])
        return abs(pre_change) > np.log(1 + flat_tol)

    verdicts = {moved(c) for c in ("trace_corr_rel", "trace_over_top_rel", "trace_llcorr_rel")}
    return len(verdicts) == 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet", type=Path, help="Spectral parquet from scripts/spectral_from_checkpoints.py")
    parser.add_argument("run_dir", type=Path, help="Run base dir (contains figures/training_data_*.npz, config_full.json)")
    parser.add_argument("--modulus", type=int, default=None, help="Override modulus (default: read from run_dir/config_full.json)")
    parser.add_argument("--ref-step", type=int, default=None, help="Reference step for all *_rel columns (default: earliest step in the parquet)")
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--tag", default=None, help="Output filename stem (default: derived from run_dir)")
    parser.add_argument("--pdf", action="store_true",
                         help="Also save the two-panel gate figure as gate_<tag>.pdf (for LaTeX inclusion)")
    parser.add_argument("--accuracy-csv", type=Path, default=DEFAULT_ACCURACY_CSV,
                         help="wd-sweep accuracy CSV to join train_accuracy from (scripts/build_wd_sweep_accuracy_csv.py)")
    parser.add_argument("--train-acc-thresh", type=float, default=TRAIN_ACC_LOW_THRESHOLD,
                         help="Flag checkpoints with train_accuracy below this as unreliable")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    modulus = args.modulus if args.modulus is not None else load_modulus(run_dir)
    tag = args.tag or f"p{modulus}_{run_dir.parent.name.replace('=', '')}_{run_dir.name.replace('=', '')}"

    df = load_and_join(args.parquet, run_dir)
    df = load_train_accuracy(df, run_dir, csv_path=args.accuracy_csv, low_thresh=args.train_acc_thresh)
    onset = find_onset(df, modulus)
    print(f"onset: cross_step={onset.cross_step}, fit_step={onset.fit_step:.1f}, agree={onset.agree}")

    df = add_corrected_quantities(df, ref_step=args.ref_step)
    sanity = sanity_checks(df, onset)
    df = add_robustness_variants(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    title = f"Gate figure: {run_dir.parent.name}/{run_dir.name} (p={modulus})"

    fig = plot_gate_figure(df, onset, title=title, x_min=args.x_min, x_max=args.x_max)
    fig_path = args.out_dir / f"gate_{tag}.png"
    fig.savefig(fig_path, dpi=200)
    if args.pdf:
        pdf_path = args.out_dir / f"gate_{tag}.pdf"
        fig.savefig(pdf_path)
        print(f"Wrote {pdf_path}")
    plt.close(fig)
    print(f"Wrote {fig_path}")

    fig_r = plot_robustness_figure(df, onset, title=title, x_min=args.x_min, x_max=args.x_max)
    fig_r_path = args.out_dir / f"gate_{tag}_robustness.png"
    fig_r.savefig(fig_r_path, dpi=200)
    plt.close(fig_r)
    print(f"Wrote {fig_r_path}")

    consistent = _robustness_consistent(df, onset)
    summary_path = args.out_dir / f"gate_{tag}_summary.md"
    write_summary(
        df, onset, sanity, tag=tag, run_dir=run_dir, parquet_path=args.parquet,
        out_path=summary_path, robustness_consistent=consistent,
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
