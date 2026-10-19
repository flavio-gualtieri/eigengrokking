#!/usr/bin/env python3
# scripts/make_numbers.py

"""
Every number the project reports, computed from the repo's generated data by
the same functions (and seeded bootstraps) as the report that shows it:

    RESULTS.md     every claim -> the script that computes it, the data file, and the exact numbers
    --tex FILE     optionally, the same numbers as one LaTeX \\newcommand each

    python scripts/make_numbers.py

A claim whose data doesn't exist yet (e.g. the curvature forecast before the
spectra are in) is written as TODO, never estimated.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from eigengrok.analysis.bootstrap import bootstrap_ci, bootstrap_ratio_ci

SCRIPTS = Path(__file__).resolve().parent
# The repo whose data is read (and where RESULTS.md goes); EIGENGROK_REPO points a
# frozen copy of the code (e.g. a SLURM job's snapshot) at the live repo.
ROOT = Path(os.environ.get("EIGENGROK_REPO", SCRIPTS.parent))


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses in the script look their module up there
    spec.loader.exec_module(module)
    return module


@dataclass
class Claim:
    section: str
    text: str            # may contain {macro} placeholders
    script: str
    data: list[str]
    numbers: dict[str, str] = field(default_factory=dict)  # macro name -> formatted value

    def rendered(self) -> str:
        return self.text.format(**self.numbers)


def _f(x: float, fmt: str) -> str:
    return "TODO" if x is None or not np.isfinite(x) else fmt.format(x)


def _num(prefix: str, value, ci, fmt: str) -> dict[str, str]:
    return {prefix: _f(value, fmt), prefix + "Lo": _f(ci[0], fmt), prefix + "Hi": _f(ci[1], fmt)}


def original_sweep() -> list[Claim]:
    ss = _script("sweep_summary")
    runs = pd.read_csv(ROOT / "reports/sweep_milestones.csv")
    _, ip = ss.init_effect(runs)
    wd = ss.wd_effect(runs).set_index("init")
    c = runs["n_collapses"]
    data = ["reports/wd_sweep_accuracy_curves.csv", "reports/sweep_milestones.csv", "reports/sweep_summary.md"]
    return [
        Claim("Original sweep (AdamW beta2 = 0.999)",
              "Init 8 delays onset only {OrigInitOnsetRatio}x [{OrigInitOnsetRatioLo}, {OrigInitOnsetRatioHi}] "
              "relative to init 1 (geometric mean over all weight decays).",
              "scripts/sweep_summary.py", data,
              _num("OrigInitOnsetRatio", ip["onset_gmean_ratio"], ip["onset_gmean_ratio_ci"], "{:.2f}")),
        Claim("Original sweep (AdamW beta2 = 0.999)",
              "Init 8 mainly delays memorization: median t_mem {OrigMemInitOne} (init 1) vs {OrigMemInitEight} "
              "[{OrigMemInitEightLo}, {OrigMemInitEightHi}] (init 8); the memorization-to-onset gap is shorter at init 8, "
              "ratio {OrigGapRatio} [{OrigGapRatioLo}, {OrigGapRatioHi}].",
              "scripts/sweep_summary.py", data,
              {"OrigMemInitOne": _f(ip["t_mem_median_init1"], "{:,.0f}"),
               **_num("OrigMemInitEight", ip["t_mem_median_init8"], ip["t_mem_median_init8_ci"], "{:,.0f}"),
               **_num("OrigGapRatio", ip["gap_median_ratio"], ip["gap_median_ratio_ci"], "{:.2f}")}),
        Claim("Original sweep (AdamW beta2 = 0.999)",
              "Weight decay 0.45 vs 1.0 delays the median onset {OrigWdRatioInitOne}x [{OrigWdRatioInitOneLo}, "
              "{OrigWdRatioInitOneHi}] at init 1 and {OrigWdRatioInitEight}x [{OrigWdRatioInitEightLo}, "
              "{OrigWdRatioInitEightHi}] at init 8.",
              "scripts/sweep_summary.py", data,
              {**_num("OrigWdRatioInitOne", wd.loc[1, "ratio_median"], wd.loc[1, "ratio_median_ci"], "{:.2f}"),
               **_num("OrigWdRatioInitEight", wd.loc[8, "ratio_median"], wd.loc[8, "ratio_median_ci"], "{:.2f}")}),
        Claim("Original sweep (AdamW beta2 = 0.999)",
              "Every one of the {OrigRuns} runs collapses after grokking (test accuracy from >= 0.95 to < 0.5): "
              "median {OrigCollapseMedian} times per run, range {OrigCollapseMin}-{OrigCollapseMax}.",
              "scripts/sweep_summary.py", data,
              {"OrigRuns": str(len(runs)), "OrigCollapseMedian": f"{c.median():.0f}",
               "OrigCollapseMin": f"{c.min():.0f}", "OrigCollapseMax": f"{c.max():.0f}"}),
    ]


def collapse_pilot() -> list[Claim]:
    runs = pd.read_csv(ROOT / "reports/collapse_pilot_runs.csv")
    a = runs[runs["arm"] == "A_default"]
    out = []
    names = {"B_beta2_0.98": "BetaTwo", "C_beta2_0.98_clip1": "Clip", "D_float64": "Float"}
    nums = {**_num("PilotCollapsesDefault", a["n_collapses"].mean(), bootstrap_ci(a["n_collapses"], np.mean), "{:.1f}")}
    for arm, tag in names.items():
        r = runs[runs["arm"] == arm]
        nums.update(_num(f"Pilot{tag}CollapseRatio", r["n_collapses"].mean() / a["n_collapses"].mean(),
                         bootstrap_ratio_ci(r["n_collapses"], a["n_collapses"], np.mean), "{:.2f}"))
        nums.update(_num(f"Pilot{tag}OnsetRatio", r["t_onset"].median() / a["t_onset"].median(),
                         bootstrap_ratio_ci(r["t_onset"], a["t_onset"], np.median), "{:.2f}"))
    out.append(Claim(
        "Collapse pilot (p=91, wd=1, init=1, 30k steps, 10 seeds per arm)",
        "Default AdamW collapses {PilotCollapsesDefault} [{PilotCollapsesDefaultLo}, {PilotCollapsesDefaultHi}] times "
        "per run. beta2 = 0.98 cuts that to {PilotBetaTwoCollapseRatio}x [{PilotBetaTwoCollapseRatioLo}, "
        "{PilotBetaTwoCollapseRatioHi}] with median onset {PilotBetaTwoOnsetRatio}x [{PilotBetaTwoOnsetRatioLo}, "
        "{PilotBetaTwoOnsetRatioHi}]; float64 leaves it at {PilotFloatCollapseRatio}x [{PilotFloatCollapseRatioLo}, "
        "{PilotFloatCollapseRatioHi}]; clipping removes collapses ({PilotClipCollapseRatio}x) but delays onset "
        "{PilotClipOnsetRatio}x [{PilotClipOnsetRatioLo}, {PilotClipOnsetRatioHi}].",
        "scripts/collapse_report.py", ["reports/collapse_pilot_curves.csv", "reports/collapse_pilot_runs.csv",
                                       "reports/collapse_pilot_summary.md"], nums))
    return out


def rerun_comparison() -> list[Claim]:
    path = ROOT / "reports/sweep_beta2_0.98/sweep_milestones.csv"
    if not path.exists():
        return [Claim("Sweep rerun (beta2 = 0.98) vs original", "TODO: rerun not summarised yet.",
                      "scripts/compare_sweeps.py", [str(path)])]
    cs = _script("compare_sweeps")
    new = pd.read_csv(path)
    _, pooled = cs.compare(pd.read_csv(ROOT / "reports/sweep_milestones.csv"), new)
    pooled = pooled.set_index("init")
    nums = {"RerunRuns": str(len(new))}
    for init, tag in ((1.0, "InitOne"), (8.0, "InitEight")):
        if init in pooled.index:
            p = pooled.loc[init]
            nums.update(_num(f"Rerun{tag}OnsetRatio", p["onset_ratio"], p["onset_ratio_ci"], "{:.2f}"))
            nums.update(_num(f"Rerun{tag}CollapseRatio", p["collapse_ratio"], p["collapse_ratio_ci"], "{:.3f}"))
            nums[f"Rerun{tag}Wds"] = str(int(p["n_wd"]))
        else:
            nums.update({f"Rerun{tag}{k}": "TODO" for k in ("OnsetRatio", "OnsetRatioLo", "OnsetRatioHi", "CollapseRatio",
                                                          "CollapseRatioLo", "CollapseRatioHi", "Wds")})
    return [Claim(
        "Sweep rerun (beta2 = 0.98) vs original",
        "Over the {RerunRuns} rerun runs: at init 1 ({RerunInitOneWds} weight decays) onset is "
        "{RerunInitOneOnsetRatio}x [{RerunInitOneOnsetRatioLo}, {RerunInitOneOnsetRatioHi}] the original and collapses "
        "{RerunInitOneCollapseRatio}x [{RerunInitOneCollapseRatioLo}, {RerunInitOneCollapseRatioHi}]; at init 8 "
        "({RerunInitEightWds} weight decays) onset {RerunInitEightOnsetRatio}x [{RerunInitEightOnsetRatioLo}, "
        "{RerunInitEightOnsetRatioHi}], collapses {RerunInitEightCollapseRatio}x [{RerunInitEightCollapseRatioLo}, "
        "{RerunInitEightCollapseRatioHi}].",
        "scripts/compare_sweeps.py", ["reports/sweep_beta2_0.98_accuracy_curves.csv.gz", str(path.relative_to(ROOT)),
                                      "reports/sweep_comparison.md"], nums)]


def spectral_validation() -> list[Claim]:
    vs = _script("validate_spectral_changes")
    table = pd.read_csv(ROOT / "reports/spectral_change_validation.csv")
    (m, k), scores = vs.choose(table)
    gap = table[(table["perturbation"] == vs.CHECKPOINT_GAP) & (table["m"] == m) & (table["k"] == k)]
    tr = gap[(gap["operator"] == "H") & (gap["stat"] == "trace")].iloc[0]
    # How many times more precisely a one-checkpoint change is measured than the value, per statistic (median over
    # the (m, k) grid, as in the report's "Gain from shared probes" table); top_eig is exact either way, so excluded.
    at_gap = table[(table["perturbation"] == vs.CHECKPOINT_GAP) & (table["stat"] != "top_eig")]
    gains = (at_gap.assign(g=at_gap["level_err_median"] / at_gap["change_err_same_median"])
             .groupby(["operator", "stat"])["g"].median())
    best = scores.set_index(["m", "k"]).loc[(m, k)]
    return [Claim(
        "Spectral estimator",
        "Production SLQ settings m={SlqM}, k={SlqK}: over one checkpoint gap (100 training steps), the worst "
        "90th-percentile change error over the key statistics is {SlqWorstChangeErr}; for the Hessian trace the "
        "median change error is {SlqTraceChangeSame} with the run's fixed probes vs {SlqTraceChangeIndep} with "
        "independent probes (value error {SlqTraceLevel}). Across the other statistics (trace, effective rank, "
        "top/trace, outlier count, participation ratio; H and G), changes are measured {SlqGainMin}-{SlqGainMax}x "
        "more precisely than values.",
        "scripts/validate_spectral_changes.py", ["reports/spectral_change_validation.csv",
                                                 "reports/spectral_validation.md"],
        {"SlqM": str(m), "SlqK": str(k), "SlqWorstChangeErr": f"{100 * best['worst_change_err_p90']:.1f}\\%",
         "SlqTraceChangeSame": f"{100 * tr['change_err_same_median']:.1f}\\%",
         "SlqTraceChangeIndep": f"{100 * tr['change_err_indep_median']:.0f}\\%",
         "SlqTraceLevel": f"{100 * tr['level_err_median']:.1f}\\%",
         "SlqGainMin": f"{gains.min():.1f}", "SlqGainMax": f"{gains.max():.1f}"})]


def spectral_cost() -> list[Claim]:
    path = ROOT / "reports/spectral_benchmark.md"
    nums = {"CostHG": "TODO", "CostOld": "TODO", "CostSpeedup": "TODO"}
    if path.exists():
        rows = [line.strip("|").split("|") for line in path.read_text().splitlines()
                if line.startswith("| ") and line[2].isdigit()]
        cost = {(int(r[0]), int(r[1])): float(r[4]) for r in rows}  # (m, k) -> H + G seconds
        vs = _script("validate_spectral_changes")
        (m, k), _ = vs.choose(pd.read_csv(ROOT / "reports/spectral_change_validation.csv"))
        if (m, k) in cost and (100, 100) in cost:
            nums = {"CostHG": f"{cost[(m, k)]:.1f}", "CostOld": f"{cost[(100, 100)]:.0f}",
                    "CostSpeedup": f"{cost[(100, 100)] / cost[(m, k)]:.0f}"}
    return [Claim(
        "Spectral estimator",
        "At those settings, H and G together cost {CostHG} s per checkpoint on one A100 (Hessian of the loss on the "
        "full training set), {CostSpeedup}x less than the previous m=100, k=100 setting ({CostOld} s).",
        "scripts/bench_spectral_settings.py", [str(path.relative_to(ROOT))], nums)]


def forecast() -> list[Claim]:
    path = ROOT / "reports/forecast_results.csv"
    fb = _script("forecast_benchmark")
    data = ["reports/sweep_beta2_0.98_accuracy_curves.csv.gz", "reports/sweep_beta2_0.98_training_stats.csv.gz",
            "reports/spectra_sweep/", "reports/forecast_results.csv", "reports/forecast_summary.md"]
    keys = ("ForecastWithinBase", "ForecastGain", "ForecastWithinBaseGbt", "ForecastGainGbt")
    nums = {k + suffix: "TODO" for k in keys for suffix in ("", "Lo", "Hi")}
    nums.update({k: "TODO" for k in ("RobustN", "RobustPos", "RobustNeg")})
    nums.update({f"ForecastGainNoClip{x}": "TODO" for x in ("", "Lo", "Hi")})
    if path.exists():
        r = pd.read_csv(path)
        r["target"] = r["target"].fillna("t_onset") if "target" in r else "t_onset"
        r["clipping"] = r["clipping"].fillna("on") if "clipping" in r else "on"
        nc = r[(r["clipping"] == "off") & (r["metric"] == "r2_within") & (r["model"] == "ridge")
               & (r["split"] == "settings") & (r["features"] == "B0+B1+C")]
        if len(nc):
            g = nc.iloc[0]
            nums.update(_num("ForecastGainNoClip", g["delta"], (g["delta_ci_lo"], g["delta_ci_hi"]), "{:+.3f}"))
        r = r[r["clipping"] == "on"]
        within = r[r["metric"] == "r2_within"]
        for model, tag in (("ridge", ""), ("gbt", "Gbt")):
            head = within[(within["cutoff"] == fb.PRIMARY_CUTOFF.label) & (within["model"] == model)
                          & (within["split"] == "settings") & (within["target"] == "t_onset")].set_index("features")
            if fb.BASELINE in head.index:
                b = head.loc[fb.BASELINE]
                nums.update(_num(f"ForecastWithinBase{tag}", b["value"], (b["ci_lo"], b["ci_hi"]), "{:.2f}"))
            if "B0+B1+C" in head.index:
                g = head.loc["B0+B1+C"]
                nums.update(_num(f"ForecastGain{tag}", g["delta"], (g["delta_ci_lo"], g["delta_ci_hi"]), "{:+.3f}"))
        # Every curvature-vs-baseline comparison other than the headline: does its interval exclude zero?
        curv = within[within["features"].isin(["B0+B1+C", "B0+B1+C(H only)", "B0+B1+C(G only)"])
                      & within["delta"].notna()]
        headline = ((curv["cutoff"] == fb.PRIMARY_CUTOFF.label) & (curv["model"] == "ridge") & (curv["split"] == "settings")
                    & (curv["target"] == "t_onset") & (curv["features"] == "B0+B1+C"))
        rest = curv[~headline & (curv["split"] != "init")]
        nums.update(RobustN=str(len(rest)), RobustPos=str(int((rest["delta_ci_lo"] > 0).sum())),
                    RobustNeg=str(int((rest["delta_ci_hi"] < 0).sum())))
    return [
        Claim("Forecasting ln(t_onset)",
              "Headline (fixed in advance: ridge, cutoff t_mem+500, leave one setting out): hyperparameters plus cheap "
              "training statistics explain R^2 within settings = {ForecastWithinBase} [{ForecastWithinBaseLo}, "
              "{ForecastWithinBaseHi}] of the seed-to-seed variation in onset; adding curvature (H and G) changes it "
              "by {ForecastGain} [{ForecastGainLo}, {ForecastGainHi}].",
              "scripts/forecast_benchmark.py", data,
              {k: nums[k] for k in nums if k.startswith(("ForecastWithinBase", "ForecastGain"))
               and "Gbt" not in k and "NoClip" not in k}),
        Claim("Forecasting ln(t_onset)",
              "Every model sees held-out features clipped to the range of its training fold, a step added after the "
              "first full run to stop the init-scale split extrapolating. Without it, the headline gain from curvature "
              "is {ForecastGainNoClip} [{ForecastGainNoClipLo}, {ForecastGainNoClipHi}].",
              "scripts/forecast_benchmark.py", data, {k: nums[k] for k in nums if "NoClip" in k}),
        Claim("Forecasting ln(t_onset)",
              "With gradient-boosted trees instead of ridge, the cheap statistics explain R^2 within settings = "
              "{ForecastWithinBaseGbt} [{ForecastWithinBaseGbtLo}, {ForecastWithinBaseGbtHi}], and curvature adds "
              "{ForecastGainGbt} [{ForecastGainGbtLo}, {ForecastGainGbtHi}].",
              "scripts/forecast_benchmark.py", data, {k: nums[k] for k in nums if "Gbt" in k}),
        Claim("Forecasting ln(t_onset)",
              "Across the other {RobustN} curvature-vs-baseline comparisons (cutoffs, models, H/G subsets, the "
              "weight-decay-range split and the secondary onset definitions), {RobustPos} intervals lie above zero and "
              "{RobustNeg} below; the rest include zero.",
              "scripts/forecast_benchmark.py", data, {k: nums[k] for k in ("RobustN", "RobustPos", "RobustNeg")}),
    ]


def transition_timing() -> list[Claim]:
    path = ROOT / "reports/transition_leadtimes_summary.csv"
    changes = ROOT / "reports/transition_changepoints.csv"
    data = ["reports/sweep_beta2_0.98_accuracy_curves.csv.gz", "reports/sweep_beta2_0.98_training_stats.csv.gz",
            "reports/spectra_sweep/", str(path.relative_to(ROOT)), str(changes.relative_to(ROOT)),
            "reports/transition_leadtimes.md"]
    tags = {"effective_rank": "EffRankH", "gn_effective_rank": "EffRankG", "n_outliers": "OutliersH",
            "gn_n_outliers": "OutliersG", "top_eig": "TopEigH", "trace": "TraceH", "train_loss": "TrainLoss",
            "grad_norm": "GradNorm"}
    nums = {f"Lead{t}{x}": "TODO" for t in tags.values() for x in ("", "Lo", "Hi", "First")}
    nums.update({f"LeadEffRank{h}{q}": "TODO" for h in "HG" for q in ("QOne", "QThree")})
    nums.update({k: "TODO" for k in ("TransRefBeforeOnset", "TransRefBeforeOnsetSteps", "TransHGDiff",
                                     "TransNullAbsLead")})
    if path.exists() and changes.exists():
        tl = _script("transition_leadtimes")
        s = pd.read_csv(path)
        s = s[s["variant"] == tl.PRIMARY].set_index("series")
        for series, t in tags.items():
            r = s.loc[series]
            nums.update(_num(f"Lead{t}", r["median_lead_frac"], (r["median_lead_frac_ci_lo"], r["median_lead_frac_ci_hi"]),
                             "{:+.2f}"))
            nums[f"Lead{t}First"] = f"{100 * r['frac_before']:.0f}\\%"
        for series, h in (("effective_rank", "H"), ("gn_effective_rank", "G")):
            nums[f"LeadEffRank{h}QOne"] = f"{s.loc[series, 'q25_lead_frac']:+.2f}"
            nums[f"LeadEffRank{h}QThree"] = f"{s.loc[series, 'q75_lead_frac']:+.2f}"
        ref = tl.reference_summary(pd.read_csv(changes))
        nums.update(TransRefBeforeOnset=f"{100 * ref['before_onset_median']:.0f}\\%",
                    TransRefBeforeOnsetSteps=f"{ref['before_onset_steps_median']:,.0f}",
                    TransHGDiff=f"{ref['hg_median_abs_diff']:.2f}", TransNullAbsLead=f"{ref['null_median_abs_lead']:.2f}")

    def pick(*prefixes, extra=()):
        return {k: v for k, v in nums.items() if k.startswith(prefixes) or k in extra}

    script = "scripts/transition_leadtimes.py"
    section = "Timing of the transition"
    return [
        Claim(section,
              "Test loss leaves its post-memorization plateau (the transition start) a median {TransRefBeforeOnset} of "
              "t_onset ({TransRefBeforeOnsetSteps} steps) before onset.",
              script, data, pick("TransRefBeforeOnset")),
        Claim(section,
              "The effective-rank collapse starts at the transition start: median lead {LeadEffRankH} x t_onset "
              "[{LeadEffRankHLo}, {LeadEffRankHHi}] for H (interquartile {LeadEffRankHQOne} to {LeadEffRankHQThree}) and "
              "{LeadEffRankG} [{LeadEffRankGLo}, {LeadEffRankGHi}] for G ({LeadEffRankGQOne} to {LeadEffRankGQThree}); "
              "it changes first in {LeadEffRankHFirst} (H) and {LeadEffRankGFirst} (G) of runs. H and G changepoints in "
              "the same run differ by a median {TransHGDiff} x t_onset; a random changepoint would sit a median "
              "{TransNullAbsLead} from the transition start.",
              script, data, pick("LeadEffRank", extra=("TransHGDiff", "TransNullAbsLead"))),
        Claim(section,
              "The outlier count changes earlier: median lead {LeadOutliersH} [{LeadOutliersHLo}, {LeadOutliersHHi}] (H) "
              "and {LeadOutliersG} [{LeadOutliersGLo}, {LeadOutliersGHi}] (G), first in {LeadOutliersHFirst} and "
              "{LeadOutliersGFirst} of runs.",
              script, data, pick("LeadOutliers")),
        Claim(section,
              "Later than the transition start: H top eigenvalue {LeadTopEigH} [{LeadTopEigHLo}, {LeadTopEigHHi}] (first "
              "in {LeadTopEigHFirst} of runs), H trace {LeadTraceH} [{LeadTraceHLo}, {LeadTraceHHi}] ({LeadTraceHFirst}), "
              "training loss {LeadTrainLoss} [{LeadTrainLossLo}, {LeadTrainLossHi}] ({LeadTrainLossFirst}), gradient "
              "norm {LeadGradNorm} [{LeadGradNormLo}, {LeadGradNormHi}] ({LeadGradNormFirst}).",
              script, data, pick("LeadTopEig", "LeadTrace", "LeadTrainLoss", "LeadGradNorm")),
    ]


def write(claims: list[Claim], tex: Path | None, md: Path) -> None:
    if tex is not None:
        lines = [f"% {tex.name} -- generated by `python scripts/make_numbers.py --tex {tex.name}`; do not edit by hand.", ""]
        for c in claims:
            lines.append(f"% {c.section}: {c.script}")
            lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in c.numbers.items()]
            lines.append("")
        tex.write_text("\n".join(lines))

    out = ["# Results", "",
           "Generated by `python scripts/make_numbers.py`: every claim the write-up makes, the script that computes "
           "it, the data it reads, and the exact numbers (`--tex FILE` also writes them as LaTeX macros). "
           "TODO marks a claim whose data does not exist yet.", ""]
    section = None
    for c in claims:
        if c.section != section:
            section = c.section
            out += [f"## {section}", ""]
        out += [f"- {c.rendered().replace(chr(92) + '%', '%')}",
                f"  - script: `{c.script}`; data: " + ", ".join(f"`{d}`" for d in c.data),
                "  - macros: " + (", ".join(f"`\\{n}`" for n in c.numbers) if c.numbers else "none"), ""]
    md.write_text("\n".join(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tex", type=Path, default=None, help="Also write every number as a LaTeX macro here")
    parser.add_argument("--md", type=Path, default=ROOT / "RESULTS.md")
    args = parser.parse_args()
    claims = (original_sweep() + collapse_pilot() + rerun_comparison() + spectral_validation() + spectral_cost()
              + forecast() + transition_timing())
    write(claims, args.tex, args.md)
    n = sum(len(c.numbers) for c in claims)
    todo = sum(v == "TODO" for c in claims for v in c.numbers.values())
    print(f"Wrote {args.md}{' and ' + str(args.tex) if args.tex else ''}: {len(claims)} claims, {n} numbers, {todo} TODO")


if __name__ == "__main__":
    main()
