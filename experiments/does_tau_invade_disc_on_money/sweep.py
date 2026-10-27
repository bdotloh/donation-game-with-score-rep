"""
Does a ternary strategy with threshold tau invade disc under the money norm?

For each tau in 2..TAU_MAX (ternary classes 0 / 1..tau-1 / >= tau), solve the coupled chains in closed form (money_analytical.CoupledChains)
at rho = 0.01, keep as many scores N as it takes for both distributions' analytical tails beyond N to
fall below machine epsilon, and compare the fitness of ternary(tau) with disc at those distributions.

Each closed form is checked against the engine: one step of the engine's chain from the closed-form
distributions must leave them where they are (max |dw| at round-off).

The fitness difference has a closed form. Disc helps every Need with a positive score; ternary does too,
once its own score is tau or more. With T = sum_{j >= tau} w_j^B = w_0^B r^tau / (1 - s),
K = 1 - rho T and M = (1 - rho)(1 - w_0^A) + rho (1 - w_0^B),

    delta = fit_tau - fit_disc = lam w_0^B r^tau / (1 - s) * S,    S = c M - b K (r - s).

The prefactor is positive and shrinks like r^tau, so fit_tau - fit_disc from the engine falls below
round-off for large tau; the sign of S decides invasion at every tau. Ternary(tau) invades iff
b / c < M / (K (r - s)).

Writes sweep.csv and delta.pdf next to this file.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "convergence"))
import numpy as np
from engine import BinaryStrategy, TernaryStrategy, Population, ScoringNorm, Noise, Payoffs, TransitionRates
from engine import chain
from money_analytical import CoupledChains
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

disc = BinaryStrategy.from_name("0,0|g,g")
ternary_name = "0,0,0|g,g,0|g,g,0"
norm = ScoringNorm.from_barcode("CN:000000++ NC:000000--")
eps, eps_up = 0.1, 0.49
eps_down = 1 - eps_up
noise = Noise(eps=eps, eps_up=eps_up, eps_down=eps_down)
rho = 0.01
benefit, cost = 2.0, 1.0
TAU_MAX = 1000

payoffs = Payoffs(benefit=benefit, cost=cost)
rates = TransitionRates(norm, noise)
lam = payoffs.role_factor
machine_eps = np.finfo(float).eps


def n_states(c):
    """Smallest N with the mass of both closed-form chains beyond N below machine epsilon."""
    k = c.tau
    tail_A = lambda N: c.r ** N
    tail_B = lambda N: (c.w0B * c.r ** k * c.s ** (N - k) / (1 - c.s) if N >= k
                        else c.w0B * ((c.r ** N - c.r ** k) / (1 - c.r) + c.r ** k / (1 - c.s)))
    N = int(np.ceil(np.log(machine_eps) / np.log(c.r)))
    while tail_A(N) > machine_eps or tail_B(N) > machine_eps:
        N += 1
    while N > 1 and tail_A(N - 1) <= machine_eps and tail_B(N - 1) <= machine_eps:
        N -= 1
    return N


rows = []
for tau in range(2, TAU_MAX + 1):
    c = CoupledChains(lam=lam, eps=eps, eps_up=eps_up, eps_down=eps_down, rho=rho, tau=tau).solve()
    N = n_states(c)
    w = np.vstack([c.wA(N), c.wB(N)])
    pop = Population([disc, TernaryStrategy.from_name(ternary_name, tau=tau)], rho=[1 - rho, rho])
    we = chain.extend(w)
    R = rates(pop, we)
    stationarity = np.abs(chain.change(we, R.g, R.f)).max()
    fit_disc, fit_tau = payoffs(pop, w)
    T = c.w0B * c.r ** tau / (1 - c.s)
    K, M = 1 - rho * T, (1 - rho) * (1 - c.w0A) + rho * (1 - c.w0B)
    S = cost * M - benefit * K * (c.r - c.s)
    delta = lam * T * S
    rows.append((tau, N, *(1 - w.sum(axis=1)), stationarity, fit_disc, fit_tau, fit_tau - fit_disc,
                 delta, S, M / (K * (c.r - c.s))))

data = np.array(rows)
header = "tau,N,1-sum_disc,1-sum_ternary,max_abs_dw,fit_disc,fit_tau,delta_engine,delta,S,bc_threshold"
np.savetxt(HERE / "sweep.csv", data, delimiter=",", header=header, comments="",
           fmt=["%d", "%d", "%.3e", "%.3e", "%.3e", "%.17g", "%.17g", "%.17g", "%.17g", "%.17g", "%.17g"])

tau, N, deficit, stationarity = data[:, 0], data[:, 1], data[:, 2:4], data[:, 4]
delta_engine, delta, S, bc = data[:, 7], data[:, 8], data[:, 9], data[:, 10]
resolved = np.abs(delta) > 1e-10
print(f"states kept: N = {int(N.min())}..{int(N.max())}")
print(f"max |1 - sum w|: {np.abs(deficit).max():.1e}   (machine eps {machine_eps:.1e})")
print(f"max |dw| from one engine step at the closed form: {stationarity.max():.1e}")
print(f"closed-form delta vs engine, where |delta| > 1e-10: max rel. error "
      f"{(np.abs(delta - delta_engine) / np.abs(delta))[resolved].max():.1e}")
invaders = tau[S > 0]
print(f"tau with fit_tau > fit_disc (S > 0): {invaders.astype(int).tolist() if invaders.size else 'none'}")
print(f"S in [{S.min():.6f}, {S.max():.6f}]")
print(f"ternary(tau) would invade iff b/c < threshold, threshold in [{bc.min():.6f}, {bc.max():.6f}] (b/c = {benefit / cost:g})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.2))
ax1.semilogy(tau, -delta, color="#1f5fa8", lw=1.4, label="closed form")
ax1.semilogy(tau[delta_engine < 0], -delta_engine[delta_engine < 0], ".", color="#d9822b", ms=2, label="engine")
ax1.set_xlabel(r"$\tau$")
ax1.set_ylabel(r"$\mathrm{fit}_{\mathrm{disc}} - \mathrm{fit}_\tau$")
ax1.legend(frameon=False, fontsize=9)
ax2.plot(tau, bc, color="#1f5fa8", lw=1.4)
ax2.axhline(benefit / cost, color="k", lw=0.8, ls="--")
ax2.text(TAU_MAX, benefit / cost, f"b/c = {benefit / cost:g}", ha="right", va="bottom", fontsize=9)
ax2.set_ylim(1, benefit / cost * 1.1)
ax2.set_xlabel(r"$\tau$")
ax2.set_ylabel(r"invasion threshold $b/c$")
for ax in (ax1, ax2):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.suptitle(rf"ternary($\tau$) vs disc, money norm, $\rho={rho}$", fontsize=10)
fig.tight_layout()
fig.savefig(HERE / "delta.pdf")
