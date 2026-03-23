"""
Generate a Beamer presentation comparing init=1 vs init=8 figures side by side
for FashionMNIST or MNIST experiments.

If some experiments or figures are missing, the corresponding panel will show
"EXPERIMENT PENDING" instead of failing.

Expected directory structure:
<DATASET>/
├── init=1/
│   ├── wd=0/
│   │   └── seed=0/
│   │       └── figures/
│   │           ├── training_curve_dataset=<DATASET>_alpha=1_wd=0.png
│   │           ├── training_curve_dataset=<DATASET>_alpha=1_wd=0_kappa.png
│   │           └── ...
└── init=8/
    ├── wd=0/
    │   └── seed=0/
    │       └── figures/
    │           ├── training_curve_dataset=<DATASET>_alpha=8_wd=0.png
    │           ├── training_curve_dataset=<DATASET>_alpha=8_wd=0_kappa.png
    │           └── ...

Examples:
    python make_dataset_beamer.py \
        --dataset FashionMNIST \
        --root /path/to/runs/FashionMNIST \
        --output fashionmnist_init_comparison.tex

    python make_dataset_beamer.py \
        --dataset MNIST \
        --root /path/to/runs/MNIST \
        --output mnist_init_comparison.tex

Then compile:
    pdflatex <output>.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path


WDS = ["0", "0.001", "0.01", "0.1", "1"]

# Expected figure suffixes for each experiment
# "" corresponds to:
#   training_curve_dataset=<DATASET>_alpha=<init>_wd=<wd>.png
FIGURE_SUFFIXES = [
    "",
    "_kappa",
    "_phi_2_moment",
    "_phi_mean",
    "_phi_var",
]


def latex_escape_text(s: str) -> str:
    """Escape text for normal LaTeX text fields like titles."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def latex_path(path: Path) -> str:
    """
    Safely embed an arbitrary filesystem path in LaTeX using \\detokenize.
    This handles underscores, equals signs, spaces, etc.
    """
    return r"\detokenize{" + str(path) + "}"


def figure_label_from_suffix(suffix: str) -> str:
    """Convert a filename suffix into a readable label."""
    if suffix == "":
        return "training curve"
    return suffix[1:].replace("_", r"\_")


def expected_figure_path(
    root: Path,
    dataset: str,
    init: str,
    wd: str,
    seed: str,
    suffix: str,
) -> Path:
    """Construct the expected path for a given figure."""
    filename = f"training_curve_dataset={dataset}_alpha={init}_wd={wd}{suffix}.png"
    return root / f"init={init}" / f"wd={wd}" / f"seed={seed}" / "figures" / filename


def build_panel(img: Path | None, init_label: str) -> str:
    """Build one side of the slide: either an image or a pending placeholder."""
    header = rf"\textbf{{{latex_escape_text(init_label)}}}\\[0.5em]"

    if img is not None and img.exists():
        return (
            r"\centering" + "\n"
            + header + "\n"
            + rf"\includegraphics[width=\linewidth,height=0.78\textheight,keepaspectratio]{{{latex_path(img)}}}"
        )

    return (
        r"\centering" + "\n"
        + header + "\n"
        + r"\vfill" + "\n"
        + r"{\Large EXPERIMENT PENDING}" + "\n"
        + r"\vfill"
    )


def build_frame(title: str, left_img: Path | None, right_img: Path | None) -> str:
    """Build a two-column frame with placeholders for missing images."""
    title_tex = latex_escape_text(title)
    left_panel = build_panel(left_img, "init = 1")
    right_panel = build_panel(right_img, "init = 8")

    return rf"""
\begin{{frame}}{{{title_tex}}}
\begin{{columns}}[T,onlytextwidth]
    \column{{0.49\textwidth}}
    {left_panel}

    \column{{0.49\textwidth}}
    {right_panel}
\end{{columns}}
\end{{frame}}
""".strip()


def generate_beamer(root: Path, dataset: str, seed: str = "0") -> str:
    """Generate the full Beamer document."""
    init1_root = root / "init=1"
    init8_root = root / "init=8"

    if not init1_root.exists():
        raise FileNotFoundError(f"Missing directory: {init1_root}")
    # Do not fail if init=8 does not exist yet; slides will show placeholders.
    _ = init8_root  # kept for symmetry / readability

    frames: list[str] = []
    notes: list[str] = []

    for wd in WDS:
        frames.append(rf"\section{{wd = {latex_escape_text(wd)}}}")

        for suffix in FIGURE_SUFFIXES:
            img1 = expected_figure_path(root, dataset, "1", wd, seed, suffix)
            img8 = expected_figure_path(root, dataset, "8", wd, seed, suffix)

            if not img1.exists():
                notes.append(f"[pending] missing init=1 figure: {img1}")
                left = None
            else:
                left = img1

            if not img8.exists():
                notes.append(f"[pending] missing init=8 figure: {img8}")
                right = None
            else:
                right = img8

            metric = figure_label_from_suffix(suffix)
            frame_title = f"{dataset} | wd = {wd} : {metric}"
            frames.append(build_frame(frame_title, left, right))

    notes_block = ""
    if notes:
        notes_block = (
            "% Missing files/directories detected during generation:\n"
            + "\n".join(f"% {line}" for line in notes)
            + "\n"
        )

    tex = rf"""
\documentclass[aspectratio=169]{{beamer}}

\usepackage{{graphicx}}
\usepackage{{grffile}}
\usepackage{{bookmark}}

\title{{{latex_escape_text(dataset)}: init=1 vs init=8}}
\author{{Automated comparison}}
\date{{\today}}

\begin{{document}}

\begin{{frame}}
    \titlepage
\end{{frame}}

{notes_block}

{chr(10).join(frames)}

\end{{document}}
""".strip() + "\n"

    return tex


def build_default_output_name(dataset: str) -> Path:
    """Build a dataset-specific default output filename."""
    return Path(f"{dataset.lower()}_init_comparison.tex")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="FashionMNIST",
        choices=["FashionMNIST", "MNIST"],
        help="Dataset name used in figure filenames, slide titles, and path expectations.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Path to the dataset directory containing init=1 and optionally init=8.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .tex file. Defaults to <dataset>_init_comparison.tex",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="0",
        help="Seed directory to use (default: 0)",
    )
    args = parser.parse_args()

    output_path = args.output if args.output is not None else build_default_output_name(args.dataset)

    tex = generate_beamer(args.root, dataset=args.dataset, seed=args.seed)
    output_path.write_text(tex, encoding="utf-8")
    print(f"Wrote LaTeX presentation to: {output_path}")


if __name__ == "__main__":
    main()