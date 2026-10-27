"""
Rate of convergence of the score distributions under the money norm.

Simulation 1: a population of disc only. Start from a geometric score distribution pi_0 and
step the chain until the total variation between consecutive steps is <= tol; that gives pi_1.
Simulation 2: disc and ternary, both starting from pi_1, stepped the same way; that gives
pi_2A (disc) and pi_2B (ternary).

For each simulation we record the steps to convergence, the measured decay rate of the
step-to-step TV, and at the stable scores the helping rates H, the payoff matrix and the fitness.
Figures, arrays (money_results.npz) and numbers (money_results.json) are written to slides/;
slides/make_tex.py turns the numbers into results.tex and compiles the slides.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from engine import BinaryStrategy, TernaryStrategy, Population,\
      ScoringNorm, Noise, StableScoreSolver, Payoffs
from engine.population import score_array
from money_analytical import CoupledChains
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "slides"
(OUT / "figures").mkdir(parents=True, exist_ok=True)

disc = BinaryStrategy.from_name("0,0|g,g")
tau = 20
ternary = TernaryStrategy.from_name("0,0,0|g,g,0|g,g,0",tau=tau)

norm = ScoringNorm.from_barcode("CN:000000++ NC:000000--")
eps = 0.1
eps_up = 0.49
eps_down = 1-eps_up
noise = Noise(eps=eps, eps_up=eps_up, eps_down=eps_down)
benefit, cost = 2.0, 1.0
payoffs = Payoffs(benefit=benefit, cost=cost)
tol = 1e-10
max_iter = 1_000_000
rho = .01

solver = StableScoreSolver(
    norm,
    noise,
    tol=tol,
    max_iter=max_iter,
    criterion="target")     # stop when TV(w_t, closed-form pi*) <= tol for every chain


def decay_rate(tv):
    """lam with TV_t ~ lam^t, from the slope of ln TV over the last half of the run."""
    t = np.arange(1, len(tv) + 1)[len(tv) // 2:]
    tv = tv[len(tv) // 2:]
    t, tv = t[tv > 0], tv[tv > 0]
    if len(tv) < 10:
        return np.nan
    return np.exp(np.polyfit(t, np.log(tv), 1)[0])


def linear_rate(result, i):
    """1 - (sqrt(f) - sqrt(g))^2 from the rates at the top score (a fixed random walk)."""
    g, f = result.rates.g[i, -1], result.rates.f[i, -1]
    return 1 - (np.sqrt(f) - np.sqrt(g)) ** 2


def summarise(name, pop, result):
    w = result.w
    H = payoffs.help_probabilities(pop, w)                            # H[i, k]: Can i helps Need k
    # payoff to i from meeting k, in both roles: b H[k, i] - c H[i, k], times the role factor
    P = payoffs.role_factor * (benefit * H.T - cost * H) 
    fitness = payoffs(pop, w)
    assert np.allclose(fitness, P @ pop.rho)
    out = dict(H=H, P=P, fitness=fitness, help_rate=pop.rho @ H @ pop.rho,
               n_iter=result.n_iter, t_conv=result.t_conv, t_target=result.t_target,
               converged=result.converged,
               lam=[decay_rate(result.tv[:, i]) for i in range(pop.n)],
               lam_target=[decay_rate(result.tv_target[:, i]) for i in range(pop.n)],
               lam_lin=[linear_rate(result, i) for i in range(pop.n)])
    print(f"--- {name}: converged={result.converged} drifting={result.drifting} n_iter={result.n_iter}")
    for key, value in out.items():
        print(f"{key}: {value}")
    return out


# Closed forms (the targets): disc alone is chain A with rho = 0; simulation 2 is the coupled pair.
# Keep enough scores for the geometric tail r^N to fall below machine epsilon (s < r, so B's is shorter).
lam = payoffs.role_factor
exact1 = CoupledChains(lam=lam, eps=eps, eps_up=eps_up, eps_down=eps_down, rho=0, tau=tau).solve(xtol=1e-15)
exact2 = CoupledChains(lam=lam, eps=eps, eps_up=eps_up, eps_down=eps_down, rho=rho, tau=tau).solve(xtol=1e-15)
print("closed form, simulation 1:", exact1)
print("closed form, simulation 2:", exact2)
n_exact = lambda c: int(np.ceil(np.log(np.finfo(float).eps) / np.log(c.r))) + 1
target1 = exact1.wA(n_exact(exact1))[None]
target2 = np.vstack([exact2.wA(n_exact(exact2)), exact2.wB(n_exact(exact2))])

# Simulation 1: disc only, from a geometric pi_0
pop1 = Population([disc])
r = eps_up/eps_down
init_length = 1000
pi_0 = (1-r)*np.logspace(0, init_length-1, num=init_length,base=r,dtype=np.float64)
pi_0 = score_array(pi_0, pop1.n)[0]
result1 = solver(pop1, pi_0, target=target1)
pi_1 = result1.w[0]
sim1 = summarise("simulation 1", pop1, result1)

# Simulation 2: disc and ternary, both from pi_1
pop2 = Population([disc, ternary], rho=[1-rho, rho])
result2 = solver(pop2, pi_1, target=target2)
pi_2A, pi_2B = result2.w
sim2 = summarise("simulation 2", pop2, result2)

np.savez(OUT / "money_results.npz", pi_0=pi_0, pi_1=pi_1, pi_2A=pi_2A, pi_2B=pi_2B,
         tv1=result1.tv, tv2=result2.tv, tv_target1=result1.tv_target, tv_target2=result2.tv_target)

# Figures
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
colors = {"pi0": "#9a9a9a", "pi1": "#1f5fa8", "pi2A": "#d9822b", "pi2B": "#2a9d5c"}

j_max = 1 + max(np.flatnonzero(w > 1e-3)[-1] for w in (pi_0, pi_1, pi_2A, pi_2B))
j = np.arange(j_max)
fig, ax = plt.subplots(figsize=(6.0, 3.4))
for label, w, exact, c in [(r"$\pi_0$ (geometric)", pi_0, (1-r)*r**j, colors["pi0"]),
                           (r"$\pi_1$ (disc, sim. 1)", pi_1, exact1.wA(j_max), colors["pi1"]),
                           (r"$\pi_{2A}$ (disc, sim. 2)", pi_2A, exact2.wA(j_max), colors["pi2A"]),
                           (r"$\pi_{2B}$ (ternary, sim. 2)", pi_2B, exact2.wB(j_max), colors["pi2B"])]:
    ax.plot(j, exact, color=c, lw=1.0, alpha=0.6)
    ax.scatter(j, w[:j_max], s=8, color=c, label=label)
ax.axvline(tau, color="k", lw=0.6, alpha=0.4)
ax.text(tau, ax.get_ylim()[1], rf" $\tau={tau}$", fontsize=9, va="top")
ax.set_xlabel("score $j$")
ax.set_ylabel("$w_j$")
ax.set_xlim(-1, j_max)
ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=9, title="dots: simulation, lines: closed form", title_fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "figures" / "distributions.pdf")

fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.semilogy(np.arange(1, len(result1.tv_target) + 1), np.maximum(result1.tv_target[:, 0], 1e-300), "o-",
            color=colors["pi1"], lw=1.4, ms=4, label="sim. 1: disc")
offset = len(result1.tv_target)
t2 = offset + np.arange(1, len(result2.tv_target) + 1)
for i, (label, c) in enumerate([("sim. 2: disc", colors["pi2A"]), ("sim. 2: ternary", colors["pi2B"])]):
    ax.semilogy(t2, np.maximum(result2.tv_target[:, i], 1e-300), color=c, lw=1.4, label=label)
ax.axhline(tol, color="k", lw=0.6, ls=":", alpha=0.5)
ax.axvline(offset, color="k", lw=0.6, alpha=0.4)
ax.set_xlabel("step $t$")
ax.set_ylabel(r"$\mathrm{TV}(\pi^*, w^{t})$")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "figures" / "tv.pdf")


# Numbers for the slides (slides/make_tex.py turns these into results.tex)
def plain(x):
    """numpy values -> JSON-serialisable Python values."""
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [plain(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    return float(x)


exact = lambda c: dict(r=c.r, s=c.s, gA=c.gA, fA=c.fA, gB_tau=c.gB_tau, w0A=c.w0A, w0B=c.w0B)
numbers = dict(
    params=dict(tau=tau, eps=eps, eps_up=eps_up, eps_down=eps_down, rho=rho,
                benefit=benefit, cost=cost, tol=tol, r0=r),
    sim1=sim1, sim2=sim2,
    w0=dict(pi_1=pi_1[0], pi_2A=pi_2A[0], pi_2B=pi_2B[0]),
    exact1=exact(exact1), exact2=exact(exact2),
)
with open(OUT / "money_results.json", "w") as fh:
    json.dump(plain(numbers), fh, indent=1)
print("numbers:", OUT / "money_results.json", "-- run slides/make_tex.py to build the slides")
