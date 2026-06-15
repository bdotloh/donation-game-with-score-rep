# run_bigb_parallel.py

from concurrent.futures import ProcessPoolExecutor
import os
import time
import csv
import numpy as np
from pathlib import Path
import sys
import time

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(results, total, desc, unit):
        for completed, result in enumerate(results, start=1):
            print(f"\r{desc}: {completed}/{total} {unit}s", end="", flush=True)
            yield result
        print()

src_path = Path(__file__).resolve().parents[1] / "ir-python" / "src"
sys.path.insert(0, str(src_path))

from find_ESS import Find_ESS
from helpers import generate_bb_strings, parse_bb_string, parse_strategy_string
import compute_payoff
import compute_stable

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
        maxw,
        benefit,
        cost,
        n_rep_states,
        rich_threshold,
        fit_threshold,
    ) = args

    bb = parse_bb_string(bb_name, n_rep_states=n_rep_states)

    finder = Find_ESS(
        bb=bb,
        w_init=w_init,
        rhoA=rhoA,
        maxn=maxn,
        maxw=maxw,
        benefit=benefit,
        cost=cost,
        n_rep_states=n_rep_states,
        rich_threshold=rich_threshold,
        fit_threshold=fit_threshold,
    )

    ess = finder(strategy_options, invader_options)
    ess_details = []

    for ess_name in ess:
        strategy = parse_strategy_string(ess_name, n_rep_states=n_rep_states)
        stable = compute_stable.with_1strategy(
            bb,
            strategy,
            w_init,
            maxn,
            maxw,
            n_rep_states,
            rich_threshold,
        )
        fitness = compute_payoff.one_strategy(
            strategy,
            stable["w_prev_state"],
            b=benefit,
            c=cost,
            role_factor=1
        )
        ess_details.append({
            "name": ess_name,
            "fitness": fitness,
            "w": stable["w"],
            "trans": stable["trans"],
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
    maxw=100,
    benefit=2,
    cost=1,
    n_rep_states=2,
    rich_threshold=None,
    fit_threshold=1e-5,
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
            maxw,
            benefit,
            cost,
            n_rep_states,
            rich_threshold,
            fit_threshold,
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
        ])

        for row in results:
            bb_name = row["bb_name"]
            for detail in row["ess_details"]:
                w = detail["w"]
                alpha, beta, gamma = detail["trans"]

                writer.writerow([
                    bb_name,
                    detail["name"],
                    detail["fitness"],
                    w[0],
                    w[1] if len(w) > 1 else 0.0,
                    alpha,
                    beta,
                    gamma,
                ])


if __name__ == "__main__":
    # Match the tokens/CmeetsN-only setup in make_datas.jl.
    bigb_names = sorted(generate_bb_strings(n_rep_states=2, nmeetsc=False))

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

    print(f"BigBs: {len(bigb_names)}")
    print(f"ESS options and invaders: {len(strategy_options)}")
    print(f"Noise: 0")
    print(f"maxn: {maxn}")

    start_time = time.time()
    results = find_ess_parallel_over_bigb(
        bigb_names=bigb_names,
        strategy_options=strategy_options,
        w_init=w_init,
        rhoA=0.99,
        maxn=maxn,
        maxw=maxn,
        benefit=benefit,
        cost=cost,
        n_rep_states=2,
        rich_threshold=None,
        fit_threshold=fit_threshold,
        n_workers=None,
    )
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.3f} seconds")
    save_results_csv(results, f"tokens_CmeetsN_Noise0_B{benefit}.csv")
