#!/usr/bin/env python3
"""Empirically verify limiting claims for the eight initial-rate groups."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

from compute_score_transitions import ComputeTransitionRates
from helpers import parse_bb_string, parse_strategy_string


GROUP_TO_CONDITION = {
    (0, 0, 0): "a",
    (0, 1, 0): "b",
    (1, 0, 0): "c",
    (1, 1, 0): "d",
    (0, 0, 1): "e",
    (0, 1, 1): "f",
    (1, 0, 1): "g",
    (1, 1, 1): "h",
}

EXPECTED_RESULTS = {
    "a": "Final distribution equals the initial distribution.",
    "b": "P(0)=P_0(0) and P(maxw)=1-P_0(0).",
    "c": "P(1)=P_0(0)+P_0(1); all scores n>=2 retain their initial mass.",
    "d": "P(maxw)=1.",
    "e": "P(0)=1.",
    "f": "P(0)=P_0(0)+y for some y in [0,1-P_0(0)].",
    "g": "No mass moves above the initial support.",
    "h": "Exploratory all-positive case.",
}


def evolve_to_stationarity(rate_computer, w_0, max_steps, tolerance):
    w = np.array(w_0, dtype=float, copy=True)
    for step in range(1, max_steps + 1):
        alpha, beta, gamma = rate_computer(np.array([w[0], 1 - w[0]]))
        new_w = np.empty_like(w)
        new_w[0] = (1 - alpha) * w[0] + gamma * w[1]
        new_w[1] = (
            alpha * w[0]
            + (1 - beta - gamma) * w[1]
            + gamma * w[2]
        )
        new_w[2:-1] = (
            beta * w[1:-2]
            + (1 - beta - gamma) * w[2:-1]
            + gamma * w[3:]
        )
        new_w[-1] = beta * w[-2] + (1 - gamma) * w[-1]
        change = np.max(np.abs(new_w - w))
        w = new_w
        if change <= tolerance:
            return w, step, change, True
    return w, max_steps, change, False


def expected_error(condition, final_w, initial_w):
    if condition == "a":
        return np.max(np.abs(final_w - initial_w))
    if condition == "b":
        expected = np.zeros_like(final_w)
        expected[0] = initial_w[0]
        expected[-1] = 1 - initial_w[0]
        return np.max(np.abs(final_w - expected))
    if condition == "c":
        expected = initial_w.copy()
        expected[1] += expected[0]
        expected[0] = 0
        return np.max(np.abs(final_w - expected))
    if condition == "d":
        return 1 - final_w[-1]
    if condition == "e":
        return 1 - final_w[0]
    if condition == "f":
        return max(0, initial_w[0] - final_w[0], final_w[0] - 1)
    if condition == "g":
        return np.sum(final_w[2:])
    return np.nan


def verify_combo(job):
    row, maxw, max_steps, stationary_tolerance, claim_tolerance = job
    bb = parse_bb_string(row["bb_string"], n_rep_states=2)
    strategy = parse_strategy_string(row["strategy"], n_rep_states=2)
    rate_computer = ComputeTransitionRates(bb, strategy)
    initial_w = np.zeros(maxw + 1)
    initial_w[:2] = (0.5, 0.5)
    final_w, steps, final_change, converged = evolve_to_stationarity(
        rate_computer, initial_w, max_steps, stationary_tolerance
    )

    key = (
        int(row["alpha_positive"]),
        int(row["beta_positive"]),
        int(row["gamma_positive"]),
    )
    condition = GROUP_TO_CONDITION[key]
    error = expected_error(condition, final_w, initial_w)
    support = np.flatnonzero(final_w > claim_tolerance)
    return {
        "condition": condition,
        "group": row["group"],
        "bb_string": row["bb_string"],
        "strategy": row["strategy"],
        "converged": int(converged),
        "steps": steps,
        "final_change": final_change,
        "claim_error": error,
        "claim_passed": int(np.isnan(error) or error <= claim_tolerance),
        "final_p0": final_w[0],
        "final_p1": final_w[1],
        "final_pmax": final_w[-1],
        "final_mean": np.arange(maxw + 1) @ final_w,
        "final_support_min": int(support[0]) if support.size else -1,
        "final_support_max": int(support[-1]) if support.size else -1,
    }


def rate_signature(row):
    bb = parse_bb_string(row["bb_string"], n_rep_states=2)
    strategy = parse_strategy_string(row["strategy"], n_rep_states=2)
    rate_computer = ComputeTransitionRates(bb, strategy)
    rates_at_zero = rate_computer(np.array([0.0, 1.0]))
    rates_at_one = rate_computer(np.array([1.0, 0.0]))
    return tuple(rates_at_zero) + tuple(rates_at_one)


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summaries = []
    for condition in "abcdefgh":
        condition_rows = [row for row in rows if row["condition"] == condition]
        errors = np.array([row["claim_error"] for row in condition_rows], dtype=float)
        converged_failures = sum(
            row["converged"] and not row["claim_passed"] for row in condition_rows
        )
        passed_count = sum(row["claim_passed"] for row in condition_rows)
        converged_count = sum(row["converged"] for row in condition_rows)
        if condition == "h":
            empirical_status = "exploratory"
        elif converged_failures:
            empirical_status = "false"
        elif passed_count == len(condition_rows):
            empirical_status = "true"
        else:
            empirical_status = "consistent but slow convergence"
        summaries.append(
            {
                "condition": condition,
                "group": condition_rows[0]["group"],
                "expected_result": EXPECTED_RESULTS[condition],
                "empirical_status": empirical_status,
                "combo_count": len(condition_rows),
                "converged_count": converged_count,
                "nonconverged_count": len(condition_rows) - converged_count,
                "claim_passed_count": passed_count,
                "converged_failure_count": converged_failures,
                "max_claim_error": (
                    np.nanmax(errors) if not np.all(np.isnan(errors)) else np.nan
                ),
                "min_final_p0": min(row["final_p0"] for row in condition_rows),
                "max_final_p0": max(row["final_p0"] for row in condition_rows),
                "min_final_pmax": min(row["final_pmax"] for row in condition_rows),
                "max_final_pmax": max(row["final_pmax"] for row in condition_rows),
                "min_final_mean": min(row["final_mean"] for row in condition_rows),
                "max_final_mean": max(row["final_mean"] for row in condition_rows),
                "min_support_max": min(row["final_support_max"] for row in condition_rows),
                "max_support_max": max(row["final_support_max"] for row in condition_rows),
            }
        )
    return summaries


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Results") / "initial_transition_groups" / "combinations.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results") / "initial_transition_groups" / "limit_checks",
    )
    parser.add_argument("--maxw", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--stationary-tolerance", type=float, default=1e-12)
    parser.add_argument("--claim-tolerance", type=float, default=1e-6)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    return parser


def main():
    args = build_parser().parse_args()
    with args.input.open() as handle:
        input_rows = list(csv.DictReader(handle))
    signatures = {id(row): rate_signature(row) for row in input_rows}
    representatives = {}
    for row in input_rows:
        representatives.setdefault(signatures[id(row)], row)
    jobs = [
        (
            row,
            args.maxw,
            args.max_steps,
            args.stationary_tolerance,
            args.claim_tolerance,
        )
        for row in representatives.values()
    ]

    print(
        f"Verifying {len(input_rows)} chains through "
        f"{len(jobs)} unique rate functions with {args.workers} workers."
    )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        unique_rows = list(executor.map(verify_combo, jobs, chunksize=8))
    results_by_signature = {
        signature: result
        for signature, result in zip(representatives, unique_rows)
    }
    rows = []
    for input_row in input_rows:
        result = results_by_signature[signatures[id(input_row)]].copy()
        result["bb_string"] = input_row["bb_string"]
        result["strategy"] = input_row["strategy"]
        rows.append(result)
    rows.sort(key=lambda row: (row["condition"], row["bb_string"], row["strategy"]))
    summaries = summarize(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "combination_limit_checks.csv", rows)
    write_rows(args.output_dir / "condition_summary.csv", summaries)
    with (args.output_dir / "parameters.json").open("w") as handle:
        json.dump(vars(args) | {"input": str(args.input), "output_dir": str(args.output_dir)}, handle, indent=2)
        handle.write("\n")

    print((args.output_dir / "condition_summary.csv").resolve())


if __name__ == "__main__":
    main()
