# tests/test_numbers.py

"""scripts/make_numbers.py emits valid, unique LaTeX macros and a matching RESULTS.md."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_numbers", ROOT / "scripts" / "make_numbers.py")
make_numbers = importlib.util.module_from_spec(spec)
sys.modules["make_numbers"] = make_numbers  # dataclasses look their module up there
spec.loader.exec_module(make_numbers)


def test_numbers_are_valid_unique_macros(tmp_path):
    claims = make_numbers.original_sweep() + make_numbers.collapse_pilot() + make_numbers.spectral_validation()
    make_numbers.write(claims, tmp_path / "numbers.tex", tmp_path / "RESULTS.md")
    names = re.findall(r"\\newcommand\{\\([^}]*)\}", (tmp_path / "numbers.tex").read_text())
    assert names and len(names) == len(set(names))
    assert all(re.fullmatch(r"[A-Za-z]+", n) for n in names)  # LaTeX command names: letters only
    md = (tmp_path / "RESULTS.md").read_text()
    assert "{" not in md.split("## ", 1)[1].replace("`\\", "")  # every placeholder was filled
    for c in claims:
        assert c.rendered() and all(v for v in c.numbers.values())
