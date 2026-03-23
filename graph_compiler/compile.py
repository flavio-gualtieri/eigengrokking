from __future__ import annotations

import argparse
from pathlib import Path


FIXED_WD = "0.01"
INITS = ["1", "2", "4", "6", "8"]

# Expected figure suffixes for each experiment
# "" corresponds to:
#   training_curve_dataset=<DATASET>_alpha=<init>_wd=0.01.png
FIGURE_SUFFIXES = [
    "",
    "_kappa",
    "_phi_1_moment",
    "_phi_2_moment",
    "_phi_mean",
    "_phi_var",
    "_phi_pos_neg_ratio",
    "_mass_gt1",
    "_weight_norms",
    "_last_layer_weight_norms",
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
    """Build one panel: either an image or a pending placeholder."""
    header = rf"\textbf{{{latex_escape_text(init_label)}}}\\[0.5em]"

    if img is not None and img.exists():
        return (
            r"\centering" + "\n"
            + header + "\n"
            + rf"\includegraphics[width=\linewidth,height=0.72\textheight,keepaspectratio]{{{latex_path(img)}}}"
        )

    return (
        r"\centering" + "\n"
        + header + "\n"
        + r"\vspace{0.72\textheight}"  # reserve same vertical space as images
    )


def build_frame(title: str, images: list[Path | None], init_labels: list[str]) -> str:
    """Build a 5-column frame with placeholders for missing images."""
    title_tex = latex_escape_text(title)
    panels = [build_panel(img, label) for img, label in zip(images, init_labels)]

    columns = []
    for panel in panels:
        columns.append(
            rf"""
    \column{{0.19\textwidth}}
    {panel}
""".rstrip()
        )

    return rf"""
\begin{{frame}}{{{title_tex}}}
\begin{{columns}}[T,onlytextwidth]
{chr(10).join(columns)}
\end{{columns}}
\end{{frame}}
""".strip()


def generate_beamer(root: Path, dataset: str, seed: str = "0") -> str:
    """Generate the full Beamer document."""
    init1_root = root / "init=1"
    if not init1_root.exists():
        raise FileNotFoundError(f"Missing directory: {init1_root}")

    notes: list[str] = []
    frames: list[str] = []

    frames.append(rf"\section{{wd = {latex_escape_text(FIXED_WD)}}}")

    for suffix in FIGURE_SUFFIXES:
        images: list[Path | None] = []
        init_labels: list[str] = []

        for init in INITS:
            img = expected_figure_path(root, dataset, init, FIXED_WD, seed, suffix)
            init_labels.append(f"init = {init}")

            if not img.exists():
                notes.append(f"[pending] missing init={init} figure: {img}")
                images.append(None)
            else:
                images.append(img)

        metric = figure_label_from_suffix(suffix)
        frame_title = f"{dataset} | wd = {FIXED_WD} : {metric}"
        frames.append(build_frame(frame_title, images, init_labels))

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

\title{{{latex_escape_text(dataset)}: init sweep at wd={latex_escape_text(FIXED_WD)}}}
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
    DATASETS = ["FashionMNIST", "MNIST"]

    BASE_ROOT = Path("/gpfs/scratch/qp252676/globus/grokking/runs")

    for dataset in DATASETS:
        root = BASE_ROOT / dataset
        output_path = Path(f"init_sweep_{dataset}.tex")

        print(f"Processing dataset: {dataset}")
        print(f"Root: {root}")

        tex = generate_beamer(root, dataset=dataset, seed="0")
        output_path.write_text(tex, encoding="utf-8")

        print(f"Wrote LaTeX presentation to: {output_path}")


if __name__ == "__main__":
    main()