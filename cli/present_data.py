#!/usr/bin/env python3
import re
from pathlib import Path
from collections import defaultdict

OUT_TEX = Path("figures_slides.tex")

# Add roots here (label -> root path)
# Add roots here (label -> root path)
ROOTS = [
    ("FashionMNIST", Path("flavio/gcp_runs/runs/FashionMNIST/init=8")),
    ("MNIST",        Path("flavio/runs/MNIST/init=8")),
]

# Decide the "type" based on filename endings
TYPE_ORDER = ["kappa", "phi_mean", "phi_var", "base"]
TYPE_TITLE = {
    "kappa": "kappa",
    "phi_mean": "phi mean",
    "phi_var": "phi var",
    "base": "overall",
}

wd_re = re.compile(r"wd=([^/]+)")
seed_re = re.compile(r"seed=([^/]+)")

def figure_type_from_name(name: str) -> str:
    stem = Path(name).stem
    if stem.endswith("_kappa"):
        return "kappa"
    if stem.endswith("_phi_mean"):
        return "phi_mean"
    if stem.endswith("_phi_var"):
        return "phi_var"
    # "the last": assume the one with no extra suffix
    return "base"

def tex_escape(s: str) -> str:
    return (s.replace("_", r"\_")
             .replace("%", r"\%")
             .replace("&", r"\&")
             .replace("#", r"\#"))

def as_float(x):
    try:
        return float(x)
    except:
        return x

def frame_for_type(dataset: str, ftype: str, entries: list[tuple[str, str, Path]]) -> str:
    title = f"{dataset} — init=8 — {TYPE_TITLE.get(ftype, ftype)}"
    lines = []
    lines.append(r"\begin{frame}[t]{" + tex_escape(title) + r"}")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.0}")
    lines.append(r"\begin{tabular}{cc}")

    for i, (wd, seed, p) in enumerate(entries):
        label = tex_escape(f"wd={wd}, seed={seed}")
        cell = (
            r"\begin{minipage}[t]{0.48\linewidth}\centering"
            + "\n" + r"{\scriptsize " + label + r"}\\[2pt]"
            + "\n" + r"\includegraphics[width=\linewidth]{" + p.as_posix() + r"}"
            + "\n" + r"\end{minipage}"
        )

        if i % 2 == 0:
            lines.append(cell + " & ")
        else:
            lines[-1] = lines[-1] + cell + r" \\[4pt]"

    if len(entries) % 2 == 1:
        lines[-1] = lines[-1] + r" \\[4pt]"

    lines.append(r"\end{tabular}")
    lines.append(r"\end{frame}")
    return "\n".join(lines)

# Collect: dataset -> type -> list[(wd, seed, path)]
all_data = defaultdict(lambda: defaultdict(list))
total_images = 0

for dataset, root in ROOTS:
    pngs = sorted(root.glob("wd=*/seed=*/figures/*.png"))
    total_images += len(pngs)

    for p in pngs:
        s = p.as_posix()
        mwd = wd_re.search(s)
        mseed = seed_re.search(s)
        if not (mwd and mseed):
            continue
        wd = mwd.group(1)
        seed = mseed.group(1)
        ftype = figure_type_from_name(p.name)
        all_data[dataset][ftype].append((wd, seed, p))

    # sort each type by wd then seed
    for ftype in list(all_data[dataset].keys()):
        all_data[dataset][ftype].sort(key=lambda x: (as_float(x[0]), as_float(x[1])))

preamble = r"""\documentclass[aspectratio=169]{beamer}
\usetheme{Madrid}
\usepackage{graphicx}
\usepackage{booktabs}

\title{Collected Figures}
\subtitle{init=8 sweeps}
\date{\today}

\begin{document}
\begin{frame}
\titlepage
\end{frame}
"""

enddoc = r"\end{document}"

frames = []
# For each dataset, add 4 slides in the same type order
for dataset, _root in ROOTS:
    for ftype in TYPE_ORDER:
        entries = all_data[dataset].get(ftype, [])
        frames.append(frame_for_type(dataset, ftype, entries))

OUT_TEX.write_text(preamble + "\n\n".join(frames) + "\n" + enddoc)
print(f"Wrote {OUT_TEX} with {len(frames)} frames and {total_images} images found across {len(ROOTS)} roots.")