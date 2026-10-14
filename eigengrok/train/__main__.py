"""
Train one run from a YAML config, with optional key=value overrides:

    python -m eigengrok.train --config configs/smoke.yaml
    python -m eigengrok.train --config configs/mod91_wd_sweep.yaml --set weight_decay=0.6 seed=3

Outputs land under the config's output_dir (see eigengrok/outputs.py for the layout).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eigengrok.config import ExperimentConfig
from eigengrok.tasks.registry import get_task


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="YAML config file (see configs/)")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override config keys, e.g. --set seed=3 weight_decay=0.6")
    args = parser.parse_args(argv)

    cfg = ExperimentConfig.from_yaml(args.config, overrides=args.overrides)
    get_task(cfg).run()


if __name__ == "__main__":
    main()
