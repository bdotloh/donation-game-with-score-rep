#!/usr/bin/env python3
"""Run one- or two-strategy score-reputation Markov chains.

The chains are run by engine.StableScoreSolver; every step's score
distributions, class masses and transition rates are saved. The score range
grows as needed: there is no maximum score.
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import BinaryStrategy, Population, ScoringNorm, StableScoreSolver, TernaryStrategy


RATE_NAMES = {
    2: ["alpha", "beta", "gamma"],
    3: ["alpha", "beta1", "gamma1", "beta2", "gamma2"],
}
REP_NAMES = {2: ["bad", "good"], 3: ["bad", "middle", "rich"]}
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]
LINE_ALPHAS = [0.65, 0.65, 1.0, 0.65, 1.0]


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run a Markov chain until it reaches the tolerance or maxn, then "
            "save per-step CSV files and plots. The score range grows as "
            "needed (there is no maximum score)."
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
        help=(
            'Strategy A, for example "0,0|g,g": rows are the receiver\'s class, columns '
            "the donor's, entries 0, g or a help probability. Defaults to all zero actions."
        ),
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
    parser.add_argument(
        "--rich-threshold",
        type=int,
        metavar="K",
        help=(
            "Three states only (required there): the classes are s = 0, "
            "0 < s < K and s >= K, i.e. the engine's ternary classes with "
            "tau = K - 1, so K must be at least 2. Omit it (or pass 0) for two states."
        ),
    )
    parser.add_argument(
        "--maxn", type=int, default=100, help="Maximum number of steps (default 100)."
    )
    parser.add_argument("--maxw", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Stop once no score probability changes by more than this in a step.",
    )
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


def parse_norm(cmeetsn, nmeetsc, tau):
    length = 8 if tau is None else 18
    for name, code in (("CmeetsN", cmeetsn), ("NmeetsC", nmeetsc)):
        if len(code) != length:
            raise ValueError(f"{name} must contain exactly {length} characters.")
        invalid = set(code) - set("0+-")
        if invalid:
            raise ValueError(f"{name} contains invalid characters: {sorted(invalid)}.")
    return ScoringNorm.from_barcode(f"CN:{cmeetsn} NC:{nmeetsc}", tau=tau)


def parse_strategy(text, name, tau):
    try:
        if tau is None:
            return BinaryStrategy.from_name(text)
        return TernaryStrategy.from_name(text, tau)
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from exc


def validate_args(args):
    expected_update_length = 8 if args.n_rep_states == 2 else 18
    default_update = "0" * expected_update_length
    args.cmeetsn = args.cmeetsn or default_update
    args.nmeetsc = args.nmeetsc or default_update
    args.strategy_a = args.strategy_a or default_strategy(args.n_rep_states)

    if args.n_rep_states == 2 and args.rich_threshold not in (None, 0):
        raise ValueError("rich-threshold must be omitted or 0 for two reputation states.")
    if args.n_rep_states == 3 and (args.rich_threshold is None or args.rich_threshold < 2):
        raise ValueError(
            "rich-threshold must be an integer >= 2 for three states "
            "(K = 1 leaves the middle class empty; use two states instead)."
        )
    if args.maxw is not None:
        raise ValueError(
            "--maxw is no longer supported: the score range grows as needed, so scores are not capped."
        )
    if args.maxn <= 0:
        raise ValueError("maxn must be a positive integer.")
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


def pad(values, width):
    values = np.asarray(values, dtype=float)
    return np.pad(values, (0, width - len(values)))


def class_rates(rates, top):
    """alpha, beta, gamma (binary) or alpha, beta1, gamma1, beta2, gamma2 per strategy.

    The rates of a class are read at its lowest score; top is the lowest score of the top class.
    """
    g, f = rates.g, rates.f
    columns = [g[:, 1], g[:, 2], f[:, 1]]
    if top > 1:
        columns += [g[:, top + 1], f[:, top]]
    return np.stack(columns, axis=1)


def run_chains(norm, population, w_init, args, top):
    """Per-strategy histories: one dict of (steps, ...) arrays per strategy, and the solver result."""
    solver = StableScoreSolver(
        norm,
        tol=args.tolerance,
        max_iter=args.maxn,
        stop_when_transient=False,
        record_history=True,
    )
    result = solver(population, w_init)
    width = result.w.shape[1]
    scores = np.array([[pad(row, width) for row in w] for w, _ in result.history])
    rates = np.array([class_rates(r, top) for _, r in result.history])
    onehot = np.eye(norm.partition.n_classes)[norm.partition.classify(np.arange(width))]
    reps = scores @ onehot
    histories = [
        {"score_history": scores[:, i], "trans_history": rates[:, i], "rep_history": reps[:, i]}
        for i in range(population.n)
    ]
    return histories, result


def write_rows(path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(np.asarray(row, dtype=float).tolist())


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


def save_score_snapshots(figures_dir, rows, suffix):
    width = rows.shape[1]
    snapshot_indices = np.unique(
        np.linspace(0, len(rows) - 1, min(5, len(rows)), dtype=int)
    )
    for index in snapshot_indices:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.bar(
            np.arange(width),
            rows[index],
            color=COLORS[0],
            edgecolor="black",
            linewidth=0.4,
        )
        timestep = index + 1
        ax.set(
            title=f"Score distribution at timestep {timestep}",
            xlabel="Score",
            ylabel="Probability",
            xlim=(-0.5, width - 0.5),
        )
        style_axes(ax)
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"score_trans_{timestep}{suffix}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def save_strategy_results(run_dir, history, n_rep_states, suffix):
    rate_names = RATE_NAMES[n_rep_states]
    rep_names = REP_NAMES[n_rep_states]
    scores = history["score_history"]
    write_rows(run_dir / f"evo_trans{suffix}.csv", rate_names, history["trans_history"])
    write_rows(
        run_dir / f"score_trans{suffix}.csv",
        [f"score_{index}" for index in range(scores.shape[1])],
        scores,
    )
    write_rows(run_dir / f"rep_trans{suffix}.csv", rep_names, history["rep_history"])

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    save_line_plot(
        figures_dir / f"evo_trans{suffix}.png",
        history["trans_history"],
        rate_names,
        "Transition rates over time",
        "Probability",
    )
    rep_indices = [0] if n_rep_states == 2 else [0, 2]
    save_line_plot(
        figures_dir / f"rep_trans{suffix}.png",
        history["rep_history"],
        rep_names,
        "Reputation probabilities over time",
        "Probability",
        rep_indices,
    )
    save_score_snapshots(figures_dir, scores, suffix)


def named_values(names, values):
    return {name: float(value) for name, value in zip(names, values)}


def result_snapshot(history, index, n_rep_states):
    return {
        "timestep": index + 1,
        "transition_probabilities": named_values(
            RATE_NAMES[n_rep_states], history["trans_history"][index]
        ),
        "rep_distribution": history["rep_history"][index].tolist(),
        "score_distribution": history["score_history"][index].tolist(),
    }


def transition_stationary_snapshot(history, n_rep_states, tolerance):
    rates = history["trans_history"]
    for index in range(1, len(rates)):
        if np.max(np.abs(rates[index] - rates[index - 1])) <= tolerance:
            return result_snapshot(history, index, n_rep_states)
    return None


def strategy_details(history, drifting, n_rep_states, tolerance):
    final_index = len(history["trans_history"]) - 1
    return {
        "transition_probabilities_stopped_changing": transition_stationary_snapshot(
            history, n_rep_states, tolerance
        ),
        "final": result_snapshot(history, final_index, n_rep_states),
        # Up-rate >= down-rate at the top score: no stable distribution, the scores keep rising.
        "drifting": bool(drifting),
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

    tau = args.rich_threshold - 1 if args.n_rep_states == 3 else None
    norm = parse_norm(args.cmeetsn, args.nmeetsc, tau)
    strategies = [parse_strategy(args.strategy_a, "strategy-a", tau)]
    inits = [w_a]
    rho = [1.0]
    if args.strategy_b:
        strategies.append(parse_strategy(args.strategy_b, "strategy-b", tau))
        inits.append(w_b)
        rho = [args.rho_a, 1.0 - args.rho_a]
    population = Population(strategies, rho)

    # Pad with zeros up to the lowest top-class score, so rates.g and rates.f reach every class.
    top = 1 if tau is None else tau + 1
    width = max(top + 1, *(len(w) for w in inits))
    histories, result = run_chains(norm, population, [pad(w, width) for w in inits], args, top)
    width = result.w.shape[1]

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
            "tau": tau,
            "maxn": args.maxn,
            "tolerance": args.tolerance,
            "save_results_all_iterations": args.save_results_all_iterations,
            "initial_score_distribution_A": pad(w_a, width).tolist(),
            "initial_score_distribution_B": pad(w_b, width).tolist() if args.strategy_b else None,
        },
        "results": {
            "iterations_completed": result.n_iter,
            "converged": bool(result.converged),
            "strategy_A": strategy_details(
                histories[0], result.drifting[0], args.n_rep_states, args.tolerance
            ),
            "strategy_B": (
                strategy_details(
                    histories[1], result.drifting[1], args.n_rep_states, args.tolerance
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
        for history, suffix in zip(histories, suffixes):
            save_strategy_results(run_dir, history, args.n_rep_states, suffix)
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
