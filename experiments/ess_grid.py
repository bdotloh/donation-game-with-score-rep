"""Every strategy of a one-parameter family (plus Defectors) against every other, under one scoring norm.

Used by ESS_ternary/ess_tau.py and ESS_phi/ess_phi.py. A family is a function family(args) returning
[(label, strategy), ...]; Defectors ("AllD") is added last.

Norm: --norm, a barcode "CN:xxxxxxxx NC:xxxxxxxx" (required). Results go to --out, by default
results_<CN>_<NC>_up<eps_up>_down<eps_down>_tol<tol>[_settled<settle_tol>]/ next to the script.
With --settle-tol the score chains are solved to a fixed point after stepping (settled.py).

For each (resident A, invader B) the score chains are run with A at share rho (0.99, A common) and at
1 - rho (A rare). The heatmap shows fit(A) - fit(B) with A common. A is an ESS when, as in
engine.ESSFinder, its fitness alone is at least the threshold and against every B either
fit(A) > fit(B) with A common, or the two are equal and fit(A) >= fit(B) with A rare ("equal" and the
inequalities up to --tie-tol).

Fitness is 4 * payoff, as in scripts/ess_sweep.py (at most b - c).
"""

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from engine import BinaryStrategy, Noise, Payoffs, Population, ScoringNorm, StableScoreSolver  # noqa: E402
from engine.rates import role_factor  # noqa: E402
from settled import SettledSolver  # noqa: E402

DEFECTORS = "0,0|0,0"


def all_strategies(family, a):
    return family(a) + [("AllD", BinaryStrategy.from_name(DEFECTORS))]


class Model:
    """Solver and payoffs (picklable, one per worker)."""

    def __init__(self, a, family):
        roles = role_factor(a.q)
        self.solver = StableScoreSolver(ScoringNorm.from_barcode(a.norm), Noise(a.eps, a.eps_up, a.eps_down), tol=a.tol,
                                        max_iter=a.maxn, role_factor=roles, tail_tol=a.tail_tol)
        if a.settle_tol:
            self.solver = SettledSolver(self.solver, tol=a.settle_tol, max_iter=a.settle_iter)
        self.payoffs = Payoffs(benefit=a.benefit, cost=a.cost, role_factor=roles)
        self.w_init = np.array([float(x) for x in a.w_init.split(",")])
        self.strategies = all_strategies(family, a)
        self.rho = a.rho

    def fitness(self, members, rho):
        population = Population(members, rho)
        result = self.solver(population, self.w_init)
        w = result.w
        mean_score = float(np.asarray(rho) @ (w @ np.arange(w.shape[1])))
        return 4 * self.payoffs(population, w), bool(result.converged), mean_score

    def row(self, i):
        """Everything for resident i: fitness alone, then (A, B) at both shares for every invader B."""
        A = self.strategies[i][1]
        fit, converged, mean_score = self.fitness([A], [1.0])
        alone = {"fitness": float(fit[0]), "converged": converged, "mean_score": mean_score}
        pairs = []
        for k, (_, B) in enumerate(self.strategies):
            if k == i:
                continue
            for share in (self.rho, 1 - self.rho):
                fit, converged, mean_score = self.fitness([A, B], [share, 1 - share])
                pairs.append({"invader": k, "rho_resident": share, "fit_resident": float(fit[0]),
                              "fit_invader": float(fit[1]), "converged": converged, "mean_score": mean_score})
        return i, alone, pairs


_model = None


def _init(args, family):
    global _model
    _model = Model(args, family)


def _row(i):
    return _model.row(i)


def is_ess(i, common, rare, fitness_alone, threshold, tie=0.0):
    """Differences within tie of 0 count as equal (round-off)."""
    if fitness_alone < threshold:
        return False
    for k in range(common.shape[0]):
        if k == i or common[i, k] > tie:
            continue
        if common[i, k] < -tie or rare[i, k] < -tie:
            return False
    return True


def results_dir(a):
    """Default output directory: the norm, the noise and the tolerance in its name."""
    cn, nc = a.norm.split()
    name = f"results_{cn.replace(':', '')}_{nc.replace(':', '')}_up{a.eps_up:g}_down{a.eps_down:g}_tol{a.tol:g}"
    if a.settle_tol:
        name += f"_settled{a.settle_tol:g}"
    return Path(a.results_root) / name


def run(a, family, family_config):
    a.out = str(a.out or results_dir(a))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = [label for label, _ in all_strategies(family, a)]
    n = len(labels)
    alone, pairs = [None] * n, []
    start = time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a, family)) as pool:
        for done, (i, row_alone, row_pairs) in enumerate(pool.map(_row, range(n)), 1):
            alone[i] = row_alone
            pairs += [dict(resident=i, **p) for p in row_pairs]
            print(f"\r{done}/{n} residents, {time.time() - start:.0f} s", end="", flush=True)
    print()

    common = np.full((n, n), np.nan)   # fit(A) - fit(B), A at share rho
    rare = np.full((n, n), np.nan)     # the same with A at share 1 - rho
    for p in pairs:
        target = common if p["rho_resident"] == a.rho else rare
        target[p["resident"], p["invader"]] = p["fit_resident"] - p["fit_invader"]
    ess = [is_ess(i, common, rare, alone[i]["fitness"], a.fitness_threshold, a.tie_tol) for i in range(n)]

    with open(out / "pairs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, ["resident", "invader", "rho_resident", "fit_resident", "fit_invader",
                                    "difference", "mean_score", "converged"])
        writer.writeheader()
        for p in pairs:
            writer.writerow({**p, "resident": labels[p["resident"]], "invader": labels[p["invader"]],
                             "difference": p["fit_resident"] - p["fit_invader"], "converged": int(p["converged"])})

    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, ["resident", "ess", "fitness_alone", "mean_score_alone", "worst_invader",
                                    "worst_difference", "n_invaders_beaten", "converged_alone"])
        writer.writeheader()
        for i, label in enumerate(labels):
            d = np.delete(common[i], i)
            others = [labels[k] for k in range(n) if k != i]
            writer.writerow({
                "resident": label,
                "ess": int(ess[i]),
                "fitness_alone": alone[i]["fitness"],
                "mean_score_alone": alone[i]["mean_score"],
                "worst_invader": others[int(np.argmin(d))],
                "worst_difference": float(d.min()),
                "n_invaders_beaten": int((d > 0).sum()),
                "converged_alone": int(alone[i]["converged"]),
            })

    np.savez(out / "matrices.npz", labels=labels, common=common, rare=rare,
             fitness_alone=[x["fitness"] for x in alone])
    config = {k: v for k, v in vars(a).items() if k != "func"}
    config.update(invaders_extra=["AllD " + DEFECTORS], **family_config,
                  n_converged=sum(p["converged"] for p in pairs), n_pairs=len(pairs),
                  seconds=round(time.time() - start))
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"ESS: {', '.join(label for label, e in zip(labels, ess) if e) or 'none'}")
    print(f"converged score chains: {config['n_converged']} of {config['n_pairs']} pairs")


def plot(a, parameter, tick_every):
    """Heatmap of fit(resident) - fit(invader); parameter names the axes ("τ", "φ")."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    out = Path(a.out)
    data = np.load(out / "matrices.npz")
    labels, common = list(data["labels"]), data["common"]
    config = json.loads((out / "config.json").read_text())
    with open(out / "summary.csv") as f:
        ess = [row["resident"] for row in csv.DictReader(f) if row["ess"] == "1"]

    # Invaders on x, residents on y, Defectors last on both.
    finite = common[np.isfinite(common)]
    limit = max(np.abs(finite).max(), 1e-12)
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(common, cmap="RdBu", norm=SymLogNorm(a.linthresh, vmin=-limit, vmax=limit), origin="lower",
                      interpolation="nearest")
    n = len(labels)
    ticks = [i for i in range(0, n - 1, tick_every) if i < n - 4] + [n - 1]
    ax.set_xticks(ticks, [labels[i] for i in ticks])
    ax.set_yticks(ticks, [labels[i] for i in ticks])
    ax.set_xlabel(f"invader {parameter}")
    ax.set_ylabel(f"resident {parameter}")
    for i, label in enumerate(labels):
        if label in ess:
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), n, 1, fill=False, edgecolor="black", lw=1.5))
    title = (f"fit(resident) − fit(invader), resident at ρ = {config['rho']}\n"
             f"{config['norm']}, ε = {config['eps']}, ε↑ = {config['eps_up']}, ε↓ = {config['eps_down']}, "
             f"maxn = {config['maxn']}, tol = {config['tol']:g}"
             f"{', settled to ' + format(config['settle_tol'], 'g') if config.get('settle_tol') else ''}, b = {config['benefit']}, c = {config['cost']}, q = {config['q']}"
             f"\nESS (boxed rows): {', '.join(ess) or 'none'}")
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.8,
                 label=f"fit(resident) − fit(invader)   (blue: resident wins; log beyond ±{a.linthresh:g})")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"heatmap.{ext}", dpi=150)
    print(f"wrote {out / 'heatmap.png'}")


def build_parser(doc, results_root, add_family_args):
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run_p = commands.add_parser("run", help="compute the resident x invader fitness differences")
    run_p.add_argument("--norm", required=True, help='barcode "CN:xxxxxxxx NC:xxxxxxxx"')
    run_p.add_argument("--out", help="output directory (default: results_<CN>_<NC>_up<eps_up>_down<eps_down>_tol<tol>)")
    run_p.set_defaults(results_root=str(results_root))
    add_family_args(run_p)
    run_p.add_argument("--eps", type=float, default=0.1)
    run_p.add_argument("--eps-up", type=float, default=0.5)
    run_p.add_argument("--eps-down", type=float, default=0.5)
    run_p.add_argument("--maxn", type=int, default=2000, help="score-chain steps")
    run_p.add_argument("--tol", type=float, default=1e-5, help="stop a score chain when no entry changes by more")
    run_p.add_argument("--tail-tol", type=float, default=0.0,
                       help="drop top scores whose mass is below this in every chain (0: keep all)")
    run_p.add_argument("--settle-tol", type=float, default=0.0,
                       help="then solve the chains to a fixed point to this tolerance (settled.py; 0: don't)")
    run_p.add_argument("--settle-iter", type=int, default=5000, help="fixed-point iterations at most")
    run_p.add_argument("--rho", type=float, default=0.99, help="share of the resident")
    run_p.add_argument("--w-init", default="0.5,0.5", help="initial score distribution, from score 0")
    run_p.add_argument("--benefit", type=float, default=2.0)
    run_p.add_argument("--cost", type=float, default=1.0)
    run_p.add_argument("--q", type=float, default=0.5, help="probability of being able to help")
    run_p.add_argument("--fitness-threshold", type=float, default=1e-5)
    run_p.add_argument("--tie-tol", type=float, default=0.0, help="fitness differences within this of 0 are ties")
    run_p.add_argument("--workers", type=int)
    plot_p = commands.add_parser("plot", help="draw the heatmap from the results")
    plot_p.add_argument("out", help="results directory of a run")
    plot_p.add_argument("--linthresh", type=float, default=1e-5, help="colour scale is linear within ±this")
    return parser
