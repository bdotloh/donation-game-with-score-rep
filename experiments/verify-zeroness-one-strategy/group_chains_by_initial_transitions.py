#!/usr/bin/env python3
"""Run and group all two-state CmeetsN-only BigB/strategy chains.

The chains are run by engine.StableScoreSolver (no noise), whose score range
grows as needed: there is no maximum score.
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine import BinaryStrategy, Noise, Population, ScoringNorm, StableScoreSolver, all_norms  # noqa: E402


RATE_NAMES = ("alpha", "beta", "gamma")


def strategy_strings():
    return [
        f"{a},{b}|{c},{d}"
        for a, b, c, d in product(("0", "g"), repeat=4)
    ]


def bigb_name(norm):
    """Barcode in the "CmeetsN:... NmeetsC:..." spelling of the earlier results."""
    cn, nc = (part.split(":")[1] for part in norm.barcode.split())
    return f"CmeetsN:{cn} NmeetsC:{nc}"


def class_rates(solver, population, w):
    """alpha, beta, gamma: the up-rate from score 0, and the up- and down-rates at score 1."""
    w = np.pad(w, ((0, 0), (0, max(0, 3 - w.shape[1]))))
    rates = solver.rates(population, w)
    return rates.g[0, 1], rates.g[0, 2], rates.f[0, 1]


def group_name(rates):
    return ",".join(
        f"{name}{'>0' if value > 0 else '=0'}"
        for name, value in zip(RATE_NAMES, rates)
    )


def run_combo(job):
    bb_string, strategy_string, w_0, maxn, tolerance = job
    solver = StableScoreSolver(
        ScoringNorm.from_barcode(bb_string),
        Noise(),
        tol=tolerance,
        max_iter=maxn,
        stop_when_transient=False,
    )
    population = Population([BinaryStrategy.from_name(strategy_string)])

    initial_rates = class_rates(solver, population, np.array([[0.5, 0.5]]))
    result = solver(population, np.asarray(w_0, dtype=float)[None])
    # As before, the final rates are those of the last step, computed from w_prev.
    final_rates = class_rates(solver, population, result.w_prev)

    return {
        "group": group_name(initial_rates),
        "alpha_positive": int(initial_rates[0] > 0),
        "beta_positive": int(initial_rates[1] > 0),
        "gamma_positive": int(initial_rates[2] > 0),
        "bb_string": bb_string,
        "strategy": strategy_string,
        "initial_alpha": initial_rates[0],
        "initial_beta": initial_rates[1],
        "initial_gamma": initial_rates[2],
        "final_alpha": final_rates[0],
        "final_beta": final_rates[1],
        "final_gamma": final_rates[2],
        "final_score_0": result.w[0, 0],
        # Number of scores the chain tracked (it grows only while mass reaches the top score).
        "final_distribution_length": result.w.shape[1],
        "converged": int(result.converged),
        # Up-rate >= down-rate at the top score: no stable distribution, the scores keep rising.
        "drifting": int(result.drifting[0]),
    }


def write_rows(path, rows):
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, rows):
    counts = {}
    for row in rows:
        key = (
            row["alpha_positive"],
            row["beta_positive"],
            row["gamma_positive"],
            row["group"],
        )
        counts[key] = counts.get(key, 0) + 1

    summary_rows = [
        {
            "alpha_positive": alpha,
            "beta_positive": beta,
            "gamma_positive": gamma,
            "group": name,
            "combo_count": counts.get((alpha, beta, gamma, name), 0),
        }
        for alpha, beta, gamma in product((0, 1), repeat=3)
        for name in [group_name((alpha, beta, gamma))]
    ]
    write_rows(path, summary_rows)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxw", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--maxn", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results") / "initial_transition_groups",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.maxw is not None:
        raise ValueError("--maxw is no longer supported: the score range grows as needed, so scores are not capped.")
    if args.maxn < 1:
        raise ValueError("maxn must be positive.")

    w_0 = np.array([0.5, 0.5])
    bigbs = [bigb_name(norm) for norm in all_norms(use_NC=False)]   # 2^4 * 3^4 = 1296
    strategies = strategy_strings()

    jobs = [
        (bb_string, strategy, w_0, args.maxn, args.tolerance)
        for bb_string in bigbs
        for strategy in strategies
    ]
    print(f"Running {len(jobs)} chains with {args.workers} workers.")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(run_combo, jobs, chunksize=32))

    rows.sort(
        key=lambda row: (
            row["alpha_positive"],
            row["beta_positive"],
            row["gamma_positive"],
            row["bb_string"],
            row["strategy"],
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combo_path = args.output_dir / "combinations.csv"
    summary_path = args.output_dir / "group_summary.csv"
    parameters_path = args.output_dir / "parameters.json"
    write_rows(combo_path, rows)
    write_summary(summary_path, rows)
    with parameters_path.open("w") as handle:
        json.dump(
            {
                "n_rep_states": 2,
                "cmeetsn": True,
                "nmeetsc": False,
                "initial_score_distribution": w_0.tolist(),
                "maxn": args.maxn,
                "tolerance": args.tolerance,
                "bigb_count": len(bigbs),
                "strategy_count": len(strategies),
                "combo_count": len(rows),
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print(combo_path.resolve())
    print(summary_path.resolve())
    print(parameters_path.resolve())


if __name__ == "__main__":
    main()
