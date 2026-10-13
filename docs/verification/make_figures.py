"""Figures for the verification report: the engine against the closed forms in the notes.

    PYTHONPATH=. python docs/verification/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln

from engine import (Defectors, Discriminators, Noise, PhiDisc, Population, ScoringNorm,
                    StableScoreSolver)

OUT = Path(__file__).parent / "figures"
BLUE, VERMILION, GREY = "#0072B2", "#D55E00", "#555555"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.dpi": 200, "axes.grid": True, "grid.alpha": 0.25})

EPS, EPS_UP, EPS_DOWN = 0.1, 0.4, 0.6


def phi_disc_gamma_solution(n_scores=400):
    """Notes p. 8: pi_k for phi = 1, with pi_0 from the normalisation equation."""
    eps_up, eps_down = EPS * EPS_UP, EPS * EPS_DOWN
    k = np.arange(1, n_scores)

    def tail(pi0):
        B = (1 - EPS) * (1 - pi0) / 4
        a, b, r = B / eps_up, eps_down / (B + eps_down), eps_up / (B + eps_down)
        log_ratio = (np.log(eps_up / eps_down) + np.log(k) + (k - 1) * np.log(r)
                     + gammaln(k + a) + gammaln(1 + b) - gammaln(1 + a) - gammaln(k + b))
        return np.exp(log_ratio)           # pi_k / pi_0

    pi0 = brentq(lambda p: p * (1 + tail(p).sum()) - 1, 1e-6, 1 - 1e-6, xtol=1e-15)
    return pi0 * np.concatenate([[1.0], tail(pi0)])


def figure_phi_disc():
    analytic = phi_disc_gamma_solution()
    norm = ScoringNorm.from_barcode("CN:00000-0+ NC:00000000")
    solver = StableScoreSolver(norm, Noise(EPS, EPS_UP, EPS_DOWN), max_iter=2000, force_max_iter=True,
                               stop_when_transient=False)
    w = solver(Population([PhiDisc(1.0)]), [1.0]).w[0]
    n = min(len(w), len(analytic))
    scores = np.arange(n)
    error = np.abs(w[:n] - analytic[:n])
    print(f"phi-Disc: pi0 = {analytic[0]:.6f}, mean = {(np.arange(len(analytic)) * analytic).sum():.4f}, "
          f"max error {error.max():.2e} over {n} scores")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    m = 31
    ax1.semilogy(scores[:m], analytic[:m], "-", color=BLUE, lw=1.6, label="gamma-function solution (notes p. 8)")
    ax1.semilogy(scores[:m], w[:m], "o", color=VERMILION, ms=3.5, mfc="none", label="engine, 2000 steps from score 0")
    ax1.set(xlabel="score $j$", ylabel="$\\pi_j$", title="Stationary distribution, $\\phi=1$")
    ax1.legend(frameon=False, fontsize=7.5)
    ax2.semilogy(scores[:m], np.maximum(error[:m], 1e-17), "s", color=GREY, ms=3)
    ax2.set(xlabel="score $j$", ylabel="|engine $-$ analytic|", title="Difference", ylim=(1e-17, 1e-12))
    fig.tight_layout()
    fig.savefig(OUT / "phi_disc.png")
    plt.close(fig)


def figure_defectors_disc():
    eps, eu, ed = EPS, EPS_UP, EPS_DOWN
    y = brentq(lambda y: 0.1125 * y**2 + 0.135 * y - 0.04, 0, 1)
    q = y / 2 + 1 / 3
    r = [0.04 / (0.225 * q + 0.06), 2 / 3]
    j = np.arange(21)
    analytic = [(1 - ri) * ri**j for ri in r]
    norm = ScoringNorm.from_barcode("CN:00000-00 NC:00000000")
    population = Population([Defectors(), Discriminators()], rho=[0.5, 0.5])
    solver = StableScoreSolver(norm, Noise(eps, eu, ed), max_iter=2000, force_max_iter=True,
                               stop_when_transient=False)
    w = solver(population, [0.5, 0.5]).w
    errors = [np.abs(w[i, :21] - analytic[i]) for i in range(2)]
    print("Defectors/Discriminators: max error", [f"{e.max():.3e}" for e in errors])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for i, (name, colour) in enumerate((("Defectors", BLUE), ("Discriminators", VERMILION))):
        ax1.semilogy(j, analytic[i], "-", color=colour, lw=1.6, label=f"{name}, analytic")
        ax1.semilogy(j, w[i, :21], "o", color=colour, ms=3.5, mfc="none", label=f"{name}, engine")
        ax2.semilogy(j, np.maximum(errors[i], 1e-12), "s", color=colour, ms=3, label=name)
    ax1.set(xlabel="score $j$", ylabel="$\\pi_j$", title="Stationary distributions (notes p. 6)")
    ax1.legend(frameon=False, fontsize=6.5)
    ax2.set(xlabel="score $j$", ylabel="|engine $-$ analytic|", title="Difference after 2000 steps")
    ax2.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "defectors_discriminators.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_phi_disc()
    figure_defectors_disc()
