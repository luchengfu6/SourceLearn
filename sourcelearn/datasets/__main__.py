"""python -m sourcelearn.datasets <benchmark> --out <dir> [benchmark options]"""
from __future__ import annotations

import argparse
import importlib
import sys

from sourcelearn.datasets import BENCHMARKS


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m sourcelearn.datasets",
                                description="Prepare one benchmark of the paper as SourceLearn inputs (workspace + questions.json + splits).")
    subs = p.add_subparsers(dest="bench", required=True)
    for name, spec in BENCHMARKS.items():
        mod = importlib.import_module(spec["module"])
        sp = subs.add_parser(name, help=spec["paper"], description=mod.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        sp.add_argument("--out", required=True, help="output directory of the prepared benchmark")
        mod.add_arguments(sp)
        sp.set_defaults(run=mod.run)
    args = p.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
