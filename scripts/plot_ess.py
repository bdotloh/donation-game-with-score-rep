#!/usr/bin/env python3
"""Draw the ESS figure (see plots/ess_plot.py) from the results of scripts/ess_sweep.py.

    python scripts/plot_ess.py /nfs/scratch/$USER/ess/binary-cn-noise0.01 --out figures/binary-cn
    python scripts/plot_ess.py data-default/tokens_CmeetsN_Noise0.01_B2 --paper-data --out figures/tokens

Writes <out>.png and <out>.pdf. The input is the directory of a sweep (ess.csv, or the per-task files, and
config.json) or one of its csv files; --paper-data reads a data file of the paper's Julia code instead.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plots import ess_figure, read_paper_data, read_sweep  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="sweep directory or csv (a data file with --paper-data)")
    parser.add_argument("--out", type=Path, help="output path without extension (default: next to the results)")
    parser.add_argument("--paper-data", action="store_true", help="read a data file of the paper's Julia code")
    parser.add_argument("--tau", type=int, help="cutoff of ternary norms (default: from config.json)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="leave out ESS whose cooperation level (fitness / (benefit - cost)) is below this")
    parser.add_argument("--max-rows", type=int, help="draw only this many rows, evenly spread over the sorted rows")
    parser.add_argument("--show-cons", action="store_true", help="mark conservative norms (scores are only moved around)")
    parser.add_argument("--show-strategies", action="store_true", help="write the ESS strategy of every row")
    parser.add_argument("--show-bb-labels", action="store_true", help="label the columns of the norm")
    parser.add_argument("--no-legend", action="store_true")
    parser.add_argument("--title")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rows = read_paper_data(args.results) if args.paper_data else read_sweep(args.results, tau=args.tau)
    total = len(rows)
    rows = [r for r in rows if r.cooperation >= args.threshold]
    if not rows:
        sys.exit(f"None of the {total} ESS rows has a cooperation level of at least {args.threshold}.")
    if args.max_rows and len(rows) > args.max_rows:
        step = len(rows) / args.max_rows
        rows = sorted(rows, key=lambda r: (r.cooperation, r.strategy.name, r.norm.barcode))
        rows = [rows[int(k * step)] for k in range(args.max_rows)]
    print(f"{len(rows)} ESS rows (of {total}).")
    fig = ess_figure(rows, title=args.title, show_cons=args.show_cons, show_strategies=True if args.show_strategies else None,
                     show_bb_labels=args.show_bb_labels, show_legend=not args.no_legend)
    base = args.out or (Path(f"{args.results}_ess") if args.results.is_file() else args.results / "ess_figure")
    if base.suffix in (".png", ".pdf"):
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = Path(f"{base}{suffix}")
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
