#!/usr/bin/env python3
"""ESS sweep over scoring norms, on a laptop or a Slurm cluster such as Rāpoi.

  python scripts/ess_sweep.py count --norms binary-cn
  python scripts/ess_sweep.py run   --norms binary-cn --strategies binary --out results/binary-cn
  python scripts/ess_sweep.py merge results/binary-cn

For every norm, the ESS among the candidate strategies are found as find_ESS in
helper_funcs.jl does (engine.ESSFinder). Binary and ternary norms and strategies
can be mixed; ternary ones need --tau.

Norms (--norms):
  binary-cn, binary-nc, binary      every binary norm with only CN, only NC, or both
  ternary-cn, ternary-nc, ternary   every ternary norm (needs --tau; there are very many,
                                    so use --sample or a file)
  lifted-cn, lifted                 binary norms lifted to ternary (needs --tau)
  file:PATH                         one barcode per line ("CN:... NC:..."; # starts a comment)
Strategies (--strategies, and --invaders, which default to the same):
  binary (16), ternary (512, needs --tau), or file:PATH with one name per line
  ("0,0|0,g", "0,0,0|0,g,0|0,g,0", "phi:0.17").

On Slurm, each array task takes its share of the norms (SLURM_ARRAY_TASK_ID within the
array's range) and uses SLURM_CPUS_PER_TASK worker processes. Each task writes
ess_task<k>.csv (one row per ESS) and done_task<k>.txt (every norm it finished, so a
restarted task skips them). "merge" combines the tasks' files and reports missing norms.
See scripts/raapoi/ for job scripts.
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from engine import (  # noqa: E402
    ESSFinder,
    Noise,
    Payoffs,
    Population,
    ScoringNorm,
    StableScoreSolver,
    all_binary_strategies,
    all_ternary_strategies,
    strategy_from_name,
)
from engine.partitions import PossessionPartition, TernaryPartition  # noqa: E402
from engine.rates import role_factor  # noqa: E402

COLUMNS = ["norm", "ess", "fitness", "w0", "w1", "mean_score", "alpha", "beta", "gamma", "beta_top", "gamma_top",
           "converged"]


# ---------------------------------------------------------------- norms


class NormSet:
    """An indexable set of norms, generated on demand (there can be millions)."""

    def __init__(self, kind, tau=None):
        self.kind = kind
        self.tau = tau
        self._file = None
        if kind.startswith("file:"):
            lines = Path(kind[5:]).read_text().splitlines()
            self._file = [line.split("#")[0].strip() for line in lines]
            self._file = [line for line in self._file if line]
            return
        base, _, rules = kind.partition("-")
        if base not in ("binary", "ternary", "lifted") or rules not in ("", "cn", "nc"):
            raise ValueError(f"Unknown norm set {kind!r}.")
        if base != "binary" and tau is None:
            raise ValueError(f"The norm set {kind!r} needs --tau.")
        self.lift = base == "lifted"
        self.partition = TernaryPartition(tau) if base == "ternary" else PossessionPartition()
        n = self.partition.n_classes
        shape = (n, 2, n)
        cn_free = np.ones(shape, dtype=bool)
        cn_free[:, :, 0] = False                     # a Can with score 0 can't lose a score
        nc_free = np.ones(shape, dtype=bool)
        nc_free[0, :, :] = False                     # a Need with score 0 can't lose a score
        self.shape = shape
        self.cn_radix = [3 if free else 2 for free in cn_free.flat] if rules != "nc" else []
        self.nc_radix = [3 if free else 2 for free in nc_free.flat] if rules != "cn" else []

    def __len__(self):
        if self._file is not None:
            return len(self._file)
        return int(np.prod(self.cn_radix, dtype=object)) * int(np.prod(self.nc_radix, dtype=object))

    def barcode(self, index):
        if self._file is not None:
            return self._file[index]
        nc_count = int(np.prod(self.nc_radix, dtype=object))
        cn_index, nc_index = divmod(index, nc_count)
        return f"CN:{self._code(cn_index, self.cn_radix)} NC:{self._code(nc_index, self.nc_radix)}"

    def norm(self, index):
        barcode = self.barcode(index)
        tau = self.tau if len(barcode.split()[0]) == 3 + 18 else None
        norm = ScoringNorm.from_barcode(barcode, tau=tau)
        return norm.to_ternary(self.tau) if self._file is None and self.lift else norm

    def _code(self, index, radix):
        if not radix:
            return "0" * int(np.prod(self.shape))
        symbols = {2: "0+", 3: "-0+"}
        digits = []
        for r in reversed(radix):                    # last position varies fastest
            index, d = divmod(index, r)
            digits.append(symbols[r][d])
        return "".join(reversed(digits))


def task_indices(n_norms, task, n_tasks, sample=None, seed=0):
    """The norm indices of one task: a contiguous share of all norms (or of a random sample)."""
    if sample is not None and sample < n_norms:
        indices = sorted(random.Random(seed).sample(range(n_norms), sample))
    else:
        indices = range(n_norms)
    total = len(indices)
    start, stop = task * total // n_tasks, (task + 1) * total // n_tasks
    return list(islice(indices, start, stop))


# ---------------------------------------------------------------- strategies


def strategy_set(kind, tau=None):
    if kind == "binary":
        return all_binary_strategies()
    if kind == "ternary":
        if tau is None:
            raise ValueError("Ternary strategies need --tau.")
        return all_ternary_strategies(tau)
    if kind.startswith("file:"):
        lines = [line.split("#")[0].strip() for line in Path(kind[5:]).read_text().splitlines()]
        return [strategy_from_name(line, tau) for line in lines if line]
    raise ValueError(f"Unknown strategy set {kind!r}.")


# ---------------------------------------------------------------- one norm


class Sweep:
    """Everything a worker needs to find the ESS of one norm (picklable)."""

    def __init__(self, args):
        self.norms = NormSet(args.norms, args.tau)
        self.candidates = strategy_set(args.strategies, args.tau)
        self.invaders = strategy_set(args.invaders, args.tau) if args.invaders else self.candidates
        self.noise = Noise(args.eps, args.eps_up, args.eps_down)
        self.w_init = np.array([float(x) for x in args.w_init.split(",")])
        self.args = args

    def ess_rows(self, index):
        a = self.args
        norm = self.norms.norm(index)
        roles = role_factor(a.q)
        solver = StableScoreSolver(norm, self.noise, tol=a.tol, max_iter=a.maxn, role_factor=roles)
        payoffs = Payoffs(benefit=a.benefit, cost=a.cost, role_factor=roles)
        finder = ESSFinder(solver, payoffs, rho=a.rho, fitness_threshold=a.fitness_threshold,
                           final_max_iter=a.final_maxn)
        rows = []
        for strategy in finder(self.candidates, self.invaders, self.w_init):
            population = Population([strategy])
            result = solver(population, self.w_init)
            w = result.w_prev[0]                     # the distribution the payoff is computed from
            g, f = result.rates.g[0], result.rates.f[0]
            top = (self.args.tau or 0) + 1           # first score of the top class (ternary)
            rows.append({
                "norm": norm.barcode if not norm.is_ternary else f"{norm.barcode} (tau={norm.partition.tau})",
                "ess": strategy.name,
                "fitness": 4 * payoffs(population, result.w_prev)[0],   # scaled to at most b - c, as helper_funcs.jl prints
                "w0": w[0],
                "w1": w[1] if len(w) > 1 else 0.0,
                "mean_score": float(np.arange(len(w)) @ w),
                "alpha": g[1],
                "beta": g[2] if len(g) > 2 else "",
                "gamma": f[1] if len(f) > 1 else "",
                "beta_top": g[top + 1] if a.tau and len(g) > top + 1 else "",
                "gamma_top": f[top] if a.tau and len(f) > top else "",
                "converged": int(result.converged),
            })
        return index, self.norms.barcode(index), rows


_sweep = None


def _init_worker(sweep):
    global _sweep
    _sweep = sweep


def _work(index):
    return _sweep.ess_rows(index)


# ---------------------------------------------------------------- commands


def slurm_task():
    """(task index, number of tasks) from the Slurm array environment, or (0, 1)."""
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        return 0, 1
    first = int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
    last = int(os.environ.get("SLURM_ARRAY_TASK_MAX", task_id))
    step = int(os.environ.get("SLURM_ARRAY_TASK_STEP", 1))
    return (int(task_id) - first) // step, (last - first) // step + 1


def config_of(args):
    keys = ["norms", "tau", "strategies", "invaders", "eps", "eps_up", "eps_down", "rho", "maxn", "final_maxn", "tol",
            "w_init", "benefit", "cost", "q", "fitness_threshold", "sample", "seed"]
    return {k: getattr(args, k) for k in keys}


def run(args):
    task, n_tasks = slurm_task()
    task = args.task if args.task is not None else task
    n_tasks = args.n_tasks if args.n_tasks is not None else n_tasks
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count() or 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = config_of(args)
    config_path = out / "config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved != config:
            raise SystemExit(f"{config_path} holds different settings; use another --out directory.\n"
                             f"saved: {saved}\nnow:   {config}")
    else:
        try:
            with open(config_path, "x") as f:       # several tasks may start at once
                json.dump(config, f, indent=2)
        except FileExistsError:
            pass

    sweep = Sweep(args)
    indices = task_indices(len(sweep.norms), task, n_tasks, args.sample, args.seed)
    done_path = out / f"done_task{task:04d}.txt"
    done = set()
    if done_path.exists():
        done = {int(line.split("\t")[0]) for line in done_path.read_text().splitlines() if line}
    todo = [i for i in indices if i not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"task {task + 1}/{n_tasks}: {len(indices)} norms ({len(done)} done before), {len(todo)} to do, "
          f"{len(sweep.candidates)} candidates, {len(sweep.invaders)} invaders, {workers} workers", flush=True)
    if not todo:
        return

    csv_path = out / f"ess_task{task:04d}.csv"
    new_file = not csv_path.exists()
    start = time.time()
    with open(csv_path, "a", newline="") as csv_file, open(done_path, "a") as done_file:
        writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        if workers == 1:
            results = map(sweep.ess_rows, todo)
            pool = None
        else:
            pool = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(sweep,))
            results = pool.map(_work, todo, chunksize=1)
        try:
            for count, (index, barcode, rows) in enumerate(results, 1):
                writer.writerows(rows)
                csv_file.flush()
                done_file.write(f"{index}\t{barcode}\t{len(rows)}\n")
                done_file.flush()
                if count % max(1, len(todo) // 100) == 0 or count == len(todo):
                    rate = (time.time() - start) / count
                    print(f"  {count}/{len(todo)} norms, {rate:.1f} s/norm, "
                          f"about {rate * (len(todo) - count) / 60:.0f} min left", flush=True)
        finally:
            if pool is not None:
                pool.shutdown(cancel_futures=True)


def merge(args):
    out = Path(args.out)
    config = json.loads((out / "config.json").read_text())
    rows, done = [], {}
    for path in sorted(glob.glob(str(out / "ess_task*.csv"))):
        with open(path, newline="") as f:
            rows += list(csv.DictReader(f))
    for path in sorted(glob.glob(str(out / "done_task*.txt"))):
        for line in Path(path).read_text().splitlines():
            if line:
                index, barcode, n_ess = line.split("\t")
                done[int(index)] = (barcode, int(n_ess))
    n_norms = len(NormSet(config["norms"], config["tau"]))
    expected = config["sample"] if config["sample"] is not None and config["sample"] < n_norms else n_norms
    merged = out / "ess.csv"
    rows.sort(key=lambda r: (r["norm"], r["ess"]))
    with open(merged, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with_ess = sum(1 for _, n in done.values() if n)
    print(f"{len(done)}/{expected} norms done, {with_ess} with at least one ESS, {len(rows)} ESS rows -> {merged}")
    if len(done) < expected:
        print(f"{expected - len(done)} norms still to do: rerun the missing array tasks (they resume).")


def count(args):
    norms = NormSet(args.norms, args.tau)
    n = len(norms)
    print(f"{n} norms in {args.norms}" + (f" (tau={args.tau})" if args.tau else ""))
    for k in (0, n // 2, n - 1):
        print(f"  #{k}: {norms.barcode(k)}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    run_p = commands.add_parser("run", help="find the ESS for this task's share of the norms")
    run_p.add_argument("--norms", required=True)
    run_p.add_argument("--tau", type=int, help="cutoff of the ternary classes 0 / 1..tau / >tau")
    run_p.add_argument("--strategies", default="binary", help="candidate strategies (default: binary)")
    run_p.add_argument("--invaders", help="invading strategies (default: the candidates)")
    run_p.add_argument("--out", required=True, help="output directory (e.g. /nfs/scratch/$USER/ess/<name>)")
    noise = run_p.add_argument_group("model (defaults as find_ESS in helper_funcs.jl)")
    noise.add_argument("--eps", type=float, default=0.0, help="noise eps (helper_funcs.jl's NOISE_VAL)")
    noise.add_argument("--eps-up", type=float, default=0.5)
    noise.add_argument("--eps-down", type=float, default=0.5)
    noise.add_argument("--rho", type=float, default=0.99, help="share of the resident when tested against an invader")
    noise.add_argument("--maxn", type=int, default=100, help="score-chain steps (helper_funcs.jl's maxn)")
    noise.add_argument("--final-maxn", type=int, default=100, help="steps of the final 'not boring' check")
    noise.add_argument("--tol", type=float, default=1e-5)
    noise.add_argument("--w-init", default="0.5,0.5", help="initial score distribution, from score 0")
    noise.add_argument("--benefit", type=float, default=2.0)
    noise.add_argument("--cost", type=float, default=1.0)
    noise.add_argument("--q", type=float, default=0.5, help="probability of being able to help")
    noise.add_argument("--fitness-threshold", type=float, default=1e-5)
    jobs = run_p.add_argument_group("splitting the work")
    jobs.add_argument("--sample", type=int, help="use a random sample of this many norms")
    jobs.add_argument("--seed", type=int, default=0, help="seed of the sample")
    jobs.add_argument("--task", type=int, help="task index (default: from SLURM_ARRAY_TASK_ID, else 0)")
    jobs.add_argument("--n-tasks", type=int, help="number of tasks (default: from the Slurm array, else 1)")
    jobs.add_argument("--workers", type=int, help="worker processes (default: SLURM_CPUS_PER_TASK or all CPUs)")
    jobs.add_argument("--limit", type=int, help="only do this many norms (for timing a test run)")
    run_p.set_defaults(func=run)

    merge_p = commands.add_parser("merge", help="combine the tasks' results into ess.csv")
    merge_p.add_argument("out")
    merge_p.set_defaults(func=merge)

    count_p = commands.add_parser("count", help="number of norms in a set")
    count_p.add_argument("--norms", required=True)
    count_p.add_argument("--tau", type=int)
    count_p.set_defaults(func=count)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
