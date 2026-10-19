"""Score distributions solved to a fixed point, rather than stepped until the per-step change is small.

StableScoreSolver stops when no entry of w changes by more than tol in one step. A chain that mixes
slowly changes little per step long before it has settled, so that rule can stop far from the
stationary distribution (seen for phi-Disc at small phi, where the mean score is in the hundreds).

SettledSolver starts from StableScoreSolver's result and then repeats
    rates <- rates of the current w;  w <- stationary distribution of each birth-death chain under them
on scores 0..L-1 (reflecting at L-1), with L doubled until every chain's top score holds less than
tail_mass. It stops when no entry of w changes by more than tol. A fixed point of this map is a
stationary point of the score dynamics. When the chains drift (the up-rate at the top is at least the
down-rate) there is no stationary distribution, and the stepped result is returned unchanged.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from engine import StableScoreSolver  # noqa: E402
from engine.chain import stationary  # noqa: E402


@dataclass
class Settled:
    w: np.ndarray
    converged: bool
    drifting: bool
    n_iter: int


class SettledSolver:
    def __init__(self, stepper: StableScoreSolver, tol=1e-12, max_iter=20_000, tail_mass=1e-16, max_scores=2 ** 20,
                 damping=0.5):
        self.stepper = stepper
        self.rates = stepper.rates
        self.tol, self.max_iter, self.tail_mass, self.max_scores = tol, max_iter, tail_mass, max_scores
        self.damping = damping

    def _map(self, population, w):
        rates = self.rates(population, w)
        L = w.shape[1]
        return np.array([stationary(rates.g[i, :L], rates.f[i, :L]) for i in range(w.shape[0])])

    def __call__(self, population, w_init):
        start = self.stepper(population, w_init)
        if start.drifting.any():
            return Settled(start.w, False, True, start.n_iter)
        w = start.w
        for n_iter in range(1, self.max_iter + 1):
            new = self._map(population, w)
            if new[:, -1].max() > self.tail_mass and w.shape[1] < self.max_scores:
                w = np.pad(w, ((0, 0), (0, w.shape[1])))      # widen and try again
                continue
            change = np.abs(new - w).max()
            w = new if change < 1e-6 else (1 - self.damping) * w + self.damping * new
            if change <= self.tol:
                return Settled(w, True, False, start.n_iter + n_iter)
        return Settled(w, False, False, start.n_iter + self.max_iter)
