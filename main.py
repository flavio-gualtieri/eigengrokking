from __future__ import annotations

import argparse

from configs.experiments import EXPERIMENTS
from tasks.registry import TASKS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_index",
        type=int,
        default=None,
        help="Run only a single experiment index from EXPERIMENTS",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.exp_index is not None:
        experiments = [EXPERIMENTS[args.exp_index]]
    else:
        experiments = EXPERIMENTS

    for cfg in experiments:
        task = TASKS[cfg.task](cfg)
        task.run()


if __name__ == "__main__":
    main()
