#!/usr/bin/env python3
"""Run and group all two-state CmeetsN-only BigB/strategy chains."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
import os
from pathlib import Path
import sys

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

import compute_stable
from compute_score_transitions import ComputeTransitionRates
from helpers import count_bb_strings, generate_bb_strings, parse_bb_string, parse_strategy_string


RATE_NAMES = ("alpha", "beta", "gamma")


def strategy_strings():
    return [
        f"{a},{b}|{c},{d}"
        for a, b, c, d in product(("0", "g"), repeat=4)
    ]


def group_name(rates):
    return ",".join(
        f"{name}{'>0' if value > 0 else '=0'}"
        for name, value in zip(RATE_NAMES, rates)
    )


def run_combo(job):
    bb_string, strategy_string, w_0, maxw, maxn, tolerance = job
    bb = parse_bb_string(bb_string, n_rep_states=2)
    strategy = parse_strategy_string(strategy_string, n_rep_states=2)

    initial_rates = ComputeTransitionRates(bb, strategy)(np.array([0.5, 0.5]))
    result = compute_stable.with_1strategy(
        bb,
        strategy,
        np.asarray(w_0, dtype=float),
        maxn=maxn,
        maxw=maxw,
        n_rep_states=2,
        tolerance=tolerance,
        stop_when_transient=False,
    )

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
        "final_alpha": result["trans"][0],
        "final_beta": result["trans"][1],
        "final_gamma": result["trans"][2],
        "final_score_0": result["w"][0],
        "final_distribution_length": len(result["w"]),
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
    parser.add_argument("--maxw", type=int, default=100)
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
    if args.maxw < 1 or args.maxn < 1:
        raise ValueError("maxw and maxn must be positive.")

    w_0 = np.array([0.5, 0.5])
    bigbs = list(generate_bb_strings(n_rep_states=2, nmeetsc=False))
    strategies = strategy_strings()
    expected_bigbs = count_bb_strings(n_rep_states=2, cmeetsn=True, nmeetsc=False)
    if len(bigbs) != expected_bigbs:
        raise RuntimeError(f"Expected {expected_bigbs} BigBs; generated {len(bigbs)}.")

    jobs = [
        (bb_string, strategy, w_0, args.maxw, args.maxn, args.tolerance)
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
                "maxw": args.maxw,
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
