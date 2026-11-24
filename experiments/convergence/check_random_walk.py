"""
Check the solver's convergence times against the closed form for a reflected random walk.

With eps = 1 the interactions drop out and every score has up rate p = eps_up and down
rate q = eps_down (none at 0). For p < q this walk on {0, 1, ...} relaxes like

    TV_t ~ C t^(-3/2) lam^t,    lam = 1 - (sqrt(q) - sqrt(p))^2,

(the discrete-time M/M/1 queue), so the relaxation time is t_rel = -1 / ln(lam) and the
number of steps until the step-to-step total variation falls to tol solves

    ln C - 3/2 ln t + t ln(lam) = ln(tol).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.optimize import brentq
from engine import BinaryStrategy, Population, ScoringNorm, Noise, StableScoreSolver

disc = BinaryStrategy.from_name("0,0|g,g")
pop = Population([disc])
norm = ScoringNorm.from_barcode("CN:000000++ NC:000000--")
tol = 1e-10


def closed_form_lam(p, q):
    return 1 - (np.sqrt(q) - np.sqrt(p)) ** 2


def check(p, q):
    noise = Noise.from_old(1.0, p, q)
    solver = StableScoreSolver(norm, noise, tol=tol, max_iter=200_000, criterion="tv")
    result = solver(pop, [1.0])                 # start with everyone at score 0
    tv = result.tv[:, 0]
    t = np.arange(1, len(tv) + 1)

    lam = closed_form_lam(p, q)
    # measured decay rate: slope of ln(TV_t) + 3/2 ln t over the second half of the run
    half = len(tv) // 2
    slope = np.polyfit(t[half:], np.log(tv[half:]) + 1.5 * np.log(t[half:]), 1)[0]
    # predicted steps to tol, with C read off the second half of the run
    lnC = np.mean(np.log(tv[half:]) + 1.5 * np.log(t[half:]) - t[half:] * np.log(lam))
    t_pred = brentq(lambda s: lnC - 1.5 * np.log(s) + s * np.log(lam) - np.log(tol), 1.0, 1e9)

    # the stationary distribution is geometric with ratio p/q
    r = p / q
    j = np.arange(result.w.shape[1])
    tv_stat = 0.5 * np.abs(result.w[0] - (1 - r) * r ** j).sum()
    print(f"p={p:.3f} q={q:.3f}  lam: closed {lam:.6f}  measured {np.exp(slope):.6f}  "
          f"t_rel {-1 / np.log(lam):8.1f}  t_conv: solver {result.t_conv[0]:6d}  predicted {t_pred:8.1f}  "
          f"TV to geometric {tv_stat:.1e}")
    return lam, np.exp(slope), result.t_conv[0], t_pred


if __name__ == "__main__":
    for p, q in [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.1, 0.3), (0.45, 0.55)]:
        lam, lam_hat, t_conv, t_pred = check(p, q)
        # short runs leave O(1/t) corrections to the t^(-3/2) law in the fit
        assert abs(lam_hat - lam) < 0.02 * (1 - lam), "decay rate does not match the closed form"
        assert abs(t_conv - t_pred) / t_pred < 0.02, "convergence time does not match the closed form"
    print("all checks passed")
