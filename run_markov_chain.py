#!/usr/bin/env python3
"""Run one- or two-strategy score-reputation Markov chains."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

import compute_stable
from helpers import parse_bb_string, parse_strategy_string, validate_update_string


RATE_NAMES = {
    2: ["alpha", "beta", "gamma"],
    3: ["alpha", "beta1", "gamma1", "beta2", "gamma2"],
}
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]
LINE_ALPHAS = [0.65, 0.65, 1.0, 0.65, 1.0]


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run a Markov chain until it reaches the tolerance or maxn, then "
            "save per-step CSV files and plots."
        )
    )
    parser.add_argument("--n-rep-states", type=int, choices=(2, 3), default=2)
    parser.add_argument(
        "--cmeetsn",
        help="CmeetsN update string: 8 characters for 2 states or 18 for 3 states.",
    )
    parser.add_argument(
        "--nmeetsc",
        help="NmeetsC update string: 8 characters for 2 states or 18 for 3 states.",
    )
    parser.add_argument(
        "--strategy-a",
        help='Strategy A, for example "0,0|g,g". Defaults to all zero actions.',
    )
    parser.add_argument(
        "--strategy-b",
        help="Optional strategy B. Omit this flag to run one strategy.",
    )
    parser.add_argument(
        "--rho-a",
        type=float,
        default=0.5,
        help="Population proportion using strategy A in a two-strategy run.",
    )
    parser.add_argument(
        "--initial-score-distribution-a",
        dest="initial_score_distribution_a",
        default="0.5,0.5",
        help="Comma-separated initial score probabilities for strategy A.",
    )
    parser.add_argument(
        "--initial-score-distribution-b",
        dest="initial_score_distribution_b",
        help="Initial score probabilities for B; defaults to A's distribution.",
    )
    parser.add_argument("--rich-threshold", type=int)
    parser.add_argument("--maxw", type=int, default=100)
    parser.add_argument("--maxn", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("Results") / "markov_runs",
        help="Parent directory for unique run folders.",
    )
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument(
        "--save-results-all-iterations",
        dest="save_results_all_iterations",
        action="store_true",
        help="Save every iteration (default).",
    )
    save_group.add_argument(
        "--no-save-results-all-iterations",
        dest="save_results_all_iterations",
        action="store_false",
        help="Run without saving per-iteration CSV files and plots.",
    )
    parser.set_defaults(save_results_all_iterations=True)
    return parser


def default_strategy(n_rep_states):
    row = ",".join(["0"] * n_rep_states)
    return "|".join([row] * n_rep_states)


def parse_distribution(text, name):
    try:
        distribution = np.array([float(value) for value in text.split(",")])
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of numbers.") from exc
    if distribution.size == 0 or np.any(distribution < 0):
        raise ValueError(f"{name} must contain non-negative probabilities.")
    if not np.isclose(distribution.sum(), 1.0):
        raise ValueError(f"{name} must sum to 1; got {distribution.sum():.12g}.")
    return distribution


def validate_strategy_string(value, name, n_rep_states):
    tokens = value.replace("|", ",").split(",")
    expected = n_rep_states**2
    if len(tokens) != expected:
        raise ValueError(f"{name} must contain exactly {expected} actions.")
    invalid = set(tokens) - {"0", "1", "g"}
    if invalid:
        raise ValueError(f"{name} actions must be 0, 1, or g; got {sorted(invalid)}.")


def validate_args(args):
    expected_update_length = 8 if args.n_rep_states == 2 else 18
    default_update = "0" * expected_update_length
    args.cmeetsn = args.cmeetsn or default_update
    args.nmeetsc = args.nmeetsc or default_update
    args.strategy_a = args.strategy_a or default_strategy(args.n_rep_states)

    validate_update_string(args.cmeetsn, "CmeetsN", args.n_rep_states)
    validate_update_string(args.nmeetsc, "NmeetsC", args.n_rep_states)
    validate_strategy_string(args.strategy_a, "strategy-a", args.n_rep_states)
    if args.strategy_b:
        validate_strategy_string(args.strategy_b, "strategy-b", args.n_rep_states)

    if args.n_rep_states == 2 and args.rich_threshold not in (None, 0):
        raise ValueError("rich-threshold must be omitted or 0 for two reputation states.")
    if args.n_rep_states == 3 and (
        args.rich_threshold is None or args.rich_threshold <= 0
    ):
        raise ValueError("rich-threshold must be a positive integer for three states.")
    if args.maxn <= 0 or args.maxw <= 0:
        raise ValueError("maxn and maxw must be positive integers.")
    if args.tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    if not 0 <= args.rho_a <= 1:
        raise ValueError("rho-a must be between 0 and 1.")


def make_run_directory(parent):
    parent.mkdir(parents=True, exist_ok=True)
    while True:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        run_dir = parent / f"run_{stamp}_{uuid4().hex[:8]}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue


def pad_score_distribution(row, maxw):
    values = np.asarray(row, dtype=float)
    width = maxw + 1
    if len(values) > width:
        raise ValueError(
            f"Score distribution has {len(values)} bins, exceeding maxw + 1 ({width})."
        )
    return np.pad(values, (0, width - len(values)), constant_values=0.0)


def write_rows(path, header, rows, score_maxw=None):
    rows = [np.asarray(row, dtype=float).tolist() for row in rows]
    if score_maxw is not None:
        rows = [pad_score_distribution(row, score_maxw).tolist() for row in rows]
        header = [f"score_{index}" for index in range(score_maxw + 1)]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def style_axes(ax):
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_line_plot(path, rows, labels, title, ylabel, selected_indices=None):
    values = np.asarray(rows, dtype=float)
    timesteps = np.arange(1, len(values) + 1)
    selected_indices = selected_indices or range(values.shape[1])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for line_index, value_index in enumerate(selected_indices):
        ax.plot(
            timesteps,
            values[:, value_index],
            color="black",
            linestyle=LINE_STYLES[line_index % len(LINE_STYLES)],
            linewidth=2.6,
            alpha=LINE_ALPHAS[line_index % len(LINE_ALPHAS)],
            label=labels[value_index],
        )
    ax.set(title=title, xlabel="Timestep", ylabel=ylabel)
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_score_snapshots(figures_dir, rows, suffix, maxw):
    snapshot_indices = np.unique(
        np.linspace(0, len(rows) - 1, min(5, len(rows)), dtype=int)
    )
    for index in snapshot_indices:
        values = pad_score_distribution(rows[index], maxw)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.bar(
            np.arange(maxw + 1),
            values,
            color=COLORS[0],
            edgecolor="black",
            linewidth=0.4,
        )
        timestep = index + 1
        ax.set(
            title=f"Score distribution at timestep {timestep}",
            xlabel="Score",
            ylabel="Probability",
            xlim=(-0.5, maxw + 0.5),
        )
        style_axes(ax)
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"score_trans_{timestep}{suffix}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def save_strategy_results(run_dir, result, n_rep_states, suffix, maxw):
    write_rows(
        run_dir / f"evo_trans{suffix}.csv",
        RATE_NAMES[n_rep_states],
        result["trans_history"],
    )
    write_rows(
        run_dir / f"score_trans{suffix}.csv",
        [],
        result["score_history"],
        score_maxw=maxw,
    )
    rep_names = ["bad", "good"] if n_rep_states == 2 else ["bad", "middle", "rich"]
    write_rows(
        run_dir / f"rep_trans{suffix}.csv",
        rep_names,
        result["rep_history"],
    )

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    save_line_plot(
        figures_dir / f"evo_trans{suffix}.png",
        result["trans_history"],
        RATE_NAMES[n_rep_states],
        "Transition rates over time",
        "Probability",
    )
    rep_indices = [0] if n_rep_states == 2 else [0, 2]
    save_line_plot(
        figures_dir / f"rep_trans{suffix}.png",
        result["rep_history"],
        rep_names,
        "Reputation probabilities over time",
        "Probability",
        rep_indices,
    )
    save_score_snapshots(figures_dir, result["score_history"], suffix, maxw)


def named_values(names, values):
    return {name: float(value) for name, value in zip(names, values)}


def result_snapshot(result, index, n_rep_states, maxw):
    return {
        "timestep": index + 1,
        "transition_probabilities": named_values(
            RATE_NAMES[n_rep_states], result["trans_history"][index]
        ),
        "rep_distribution": np.asarray(result["rep_history"][index]).tolist(),
        "score_distribution": pad_score_distribution(
            result["score_history"][index], maxw
        ).tolist(),
    }


def transition_stationary_snapshot(result, n_rep_states, maxw, tolerance):
    rates = result["trans_history"]
    for index in range(1, len(rates)):
        change = np.max(
            np.abs(np.asarray(rates[index]) - np.asarray(rates[index - 1]))
        )
        if change <= tolerance:
            return result_snapshot(result, index, n_rep_states, maxw)
    return None


def strategy_details(result, n_rep_states, maxw, tolerance):
    final_index = len(result["trans_history"]) - 1
    return {
        "transition_probabilities_stopped_changing": transition_stationary_snapshot(
            result, n_rep_states, maxw, tolerance
        ),
        "final": result_snapshot(result, final_index, n_rep_states, maxw),
    }


def run(args):
    validate_args(args)
    w_a = parse_distribution(
        args.initial_score_distribution_a, "initial-score-distribution-a"
    )
    w_b = parse_distribution(
        args.initial_score_distribution_b or args.initial_score_distribution_a,
        "initial-score-distribution-b",
    )
    if args.maxw < max(len(w_a), len(w_b)) - 1:
        raise ValueError("maxw must be at least the largest initial score.")

    bb_string = f"CmeetsN:{args.cmeetsn} NmeetsC:{args.nmeetsc}"
    bb = parse_bb_string(bb_string, n_rep_states=args.n_rep_states)
    strategy_a = parse_strategy_string(args.strategy_a, args.n_rep_states)
    common = {
        "maxn": args.maxn,
        "maxw": args.maxw,
        "n_rep_states": args.n_rep_states,
        "rich_threshold": args.rich_threshold,
        "tolerance": args.tolerance,
        # Histories are retained in memory so details.json is complete even
        # when per-iteration CSV files and plots are disabled.
        "save_results_all_iterations": True,
        "stop_when_transient": False,
    }

    if args.strategy_b:
        strategy_b = parse_strategy_string(args.strategy_b, args.n_rep_states)
        results = compute_stable.with_2strategies(
            bb, strategy_a, strategy_b, w_a, w_b, args.rho_a, **common
        )
    else:
        results = (compute_stable.with_1strategy(bb, strategy_a, w_a, **common),)

    run_dir = make_run_directory(args.results_dir)
    details = {
        "parameters": {
            "big_brother": {"CmeetsN": args.cmeetsn, "NmeetsC": args.nmeetsc},
            "strategy_A": args.strategy_a,
            "strategy_B": args.strategy_b,
            "proportion_strategy_A": args.rho_a if args.strategy_b else 1.0,
            "proportion_strategy_B": 1.0 - args.rho_a if args.strategy_b else 0.0,
            "n_rep_states": args.n_rep_states,
            "rich_threshold": args.rich_threshold,
            "maxw": args.maxw,
            "maxn": args.maxn,
            "tolerance": args.tolerance,
            "save_results_all_iterations": args.save_results_all_iterations,
            "initial_score_distribution_A": pad_score_distribution(
                w_a, args.maxw
            ).tolist(),
            "initial_score_distribution_B": (
                pad_score_distribution(w_b, args.maxw).tolist()
                if args.strategy_b
                else None
            ),
        },
        "results": {
            "iterations_completed": len(results[0]["score_history"]),
            "strategy_A": strategy_details(
                results[0], args.n_rep_states, args.maxw, args.tolerance
            ),
            "strategy_B": (
                strategy_details(
                    results[1],
                    args.n_rep_states,
                    args.maxw,
                    args.tolerance,
                )
                if args.strategy_b
                else None
            ),
        },
    }
    with (run_dir / "details.json").open("w") as handle:
        json.dump(details, handle, indent=2)
        handle.write("\n")

    if args.save_results_all_iterations:
        suffixes = ("_A", "_B") if args.strategy_b else ("",)
        for result, suffix in zip(results, suffixes):
            save_strategy_results(run_dir, result, args.n_rep_states, suffix, args.maxw)
    return run_dir


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_dir = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(run_dir.resolve())


if __name__ == "__main__":
    main()
