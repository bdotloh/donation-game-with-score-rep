"""Find the ESS among a set of strategies for every scoring norm (Big Brother) of a sweep,
one norm per worker, and save one CSV row per (norm, ESS).

Uses engine.ESSFinder, which follows find_ESS in helper_funcs.jl.
"""

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(results, total, desc, unit):
        for completed, result in enumerate(results, start=1):
            print(f"\r{desc}: {completed}/{total} {unit}s", end="", flush=True)
            yield result
        print()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine import (  # noqa: E402
    BinaryStrategy,
    ESSFinder,
    Noise,
    Payoffs,
    Population,
    ScoringNorm,
    StableScoreSolver,
    TernaryStrategy,
    all_norms,
)


def bigb_name(norm):
    """Barcode in the "CmeetsN:... NmeetsC:..." spelling that plot_results.py reads."""
    cn, nc = (part.split(":")[1] for part in norm.barcode.split())
    return f"CmeetsN:{cn} NmeetsC:{nc}"


def class_tau(n_rep_states, rich_threshold):
    """The engine's tau (None for binary). Classes {0}, {1..k-1}, {>=k} with k = rich_threshold are tau = k - 1."""
    if n_rep_states == 2:
        if rich_threshold not in (None, 0):
            raise ValueError("For n_rep_states=2, rich_threshold must be None or 0.")
        return None
    if n_rep_states == 3:
        if rich_threshold is None or rich_threshold < 2:
            raise ValueError("For n_rep_states=3, rich_threshold must be an integer >= 2.")
        return rich_threshold - 1
    raise NotImplementedError(f"n_rep_states {n_rep_states} not implemented. Only 2 or 3 are supported.")


def make_strategy(name, tau):
    return BinaryStrategy.from_name(name) if tau is None else TernaryStrategy.from_name(name, tau)


def class_rates(solver, population, w, tau):
    """alpha, beta, gamma (binary) or alpha, beta1, gamma1, beta2, gamma2 (ternary) of strategy 0:
    up-rate from score 0, and up/down rates of a score in the middle (and top) class."""
    top = 1 if tau is None else tau + 1         # lowest score of the top class
    pad = max(0, top + 1 - w.shape[1])     # rates are only given at tracked scores
    rates = solver.rates(population, np.pad(w, ((0, 0), (0, pad))))
    g, f = rates.g[0], rates.f[0]
    if tau is None:
        return [g[1], g[2], f[1]]
    return [g[1], g[2], f[1], g[top + 1], f[top]]


def run_one_bigb(args):
    """
    One worker = one BigB.
    This must be top-level so multiprocessing can pickle it.
    """
    (
        bb_name,
        strategy_options,
        invader_options,
        w_init,
        rhoA,
        maxn,
        benefit,
        cost,
        n_rep_states,
        rich_threshold,
        fit_threshold,
        noise,
    ) = args

    tau = class_tau(n_rep_states, rich_threshold)
    solver = StableScoreSolver(ScoringNorm.from_barcode(bb_name, tau), noise, max_iter=maxn)
    finder = ESSFinder(solver, Payoffs(benefit, cost), rho=rhoA, fitness_threshold=fit_threshold)
    invaders = [make_strategy(name, tau) for name in invader_options]
    # fitness is the payoff without the role factor 1/4.
    fitness = Payoffs(benefit, cost, role_factor=1)

    ess = []
    ess_details = []
    for ess_name in strategy_options:
        strategy = make_strategy(ess_name, tau)
        if not finder.is_ess(strategy, invaders, w_init):
            continue
        population = Population([strategy])
        stable = solver(population, w_init)
        ess.append(ess_name)
        ess_details.append({
            "name": ess_name,
            "fitness": float(fitness(population, stable.w)[0]),
            "w": stable.w[0],
            "trans": [float(x) for x in class_rates(solver, population, stable.w, tau)],
        })

    return {
        "bb_name": bb_name,
        "ess": ess,
        "ess_details": ess_details,
        "n_ess": len(ess),
    }


def find_ess_parallel_over_bigb(
    bigb_names,
    strategy_options,
    w_init,
    rhoA=0.99,
    maxn=100,
    benefit=2,
    cost=1,
    n_rep_states=2,
    rich_threshold=None,
    fit_threshold=1e-5,
    noise=None,
    n_workers=None,
):
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)
    print(f'there are {n_workers} workers')

    jobs = [
        (
            bb_name,
            strategy_options,
            strategy_options,
            w_init,
            rhoA,
            maxn,
            benefit,
            cost,
            n_rep_states,
            rich_threshold,
            fit_threshold,
            noise or Noise(),
        )
        for bb_name in bigb_names
    ]

    start = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # executor.map preserves the input order, matching Julia's sorted BigBs.
        results = list(
            tqdm(
                executor.map(run_one_bigb, jobs),
                total=len(jobs),
                desc="Computing ESS",
                unit="BigB",
            )
        )

    print(f"Done in {time.time() - start:.2f} seconds")

    return results


def save_results_csv(results, path="ess_results.csv"):
    """One row per (norm, ESS). Ternary sweeps add beta2, gamma2 (alpha, beta, gamma are alpha, beta1, gamma1)."""
    ternary = any(len(d["trans"]) == 5 for row in results for d in row["ess_details"])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bb_name",
            "ess_name",
            "fitness",
            "w0",
            "w1",
            "alpha",
            "beta",
            "gamma",
        ] + (["beta2", "gamma2"] if ternary else []))

        for row in results:
            bb_name = row["bb_name"]
            for detail in row["ess_details"]:
                w = detail["w"]
                writer.writerow([
                    bb_name,
                    detail["name"],
                    detail["fitness"],
                    float(w[0]),
                    float(w[1]) if len(w) > 1 else 0.0,
                    *detail["trans"],
                ])


if __name__ == "__main__":
    # Match the tokens/CmeetsN-only setup in make_datas.jl: 2^4 * 3^4 = 1296 BigBs.
    bigb_names = sorted(bigb_name(norm) for norm in all_norms(use_NC=False))
    strategy_options = [
        f"{a},{b}|{c},{d}"
        for a in ["0", "g"]
        for b in ["0", "g"]
        for c in ["0", "g"]
        for d in ["0", "g"]
    ]

    w_init = np.array([0.5, 0.5])
    maxn = 100
    benefit = 2
    cost = 1
    fit_threshold = 0.01
    noise = Noise.from_noise_val(0.0)   # NOISE_VAL in make_datas.jl

    print(f"BigBs: {len(bigb_names)}")
    print(f"ESS options and invaders: {len(strategy_options)}")
    print(f"Noise: {noise.eps:g}")
    print(f"maxn: {maxn}")

    start_time = time.time()
    results = find_ess_parallel_over_bigb(
        bigb_names=bigb_names,
        strategy_options=strategy_options,
        w_init=w_init,
        rhoA=0.99,
        maxn=maxn,
        benefit=benefit,
        cost=cost,
        n_rep_states=2,
        rich_threshold=None,
        fit_threshold=fit_threshold,
        noise=noise,
        n_workers=None,
    )
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.3f} seconds")
    save_results_csv(results, f"tokens_CmeetsN_Noise{noise.eps:g}_B{benefit}.csv")
