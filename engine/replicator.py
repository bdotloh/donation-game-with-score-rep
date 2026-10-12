"""Replicator dynamics of strategy frequencies, with scores evolving underneath.

    d rho_i / dt = rho_i (pi_i - pi_bar),   pi_i = (A rho)_i,   pi_bar = rho . A rho

The payoff matrix A depends on rho through the score distributions w:

    A[i, k] = role_factor (benefit H[k, i] - cost H[i, k]),

H[i, k] being the probability that a Can of strategy i helps a Need of strategy k.

At each step the score chains are restarted from w_init and run until they
converge or reach the solver's max_iter, whichever comes first. When the chains
do not converge (e.g. scores drift upwards for ever, or a conservative norm
such as money spreads scores out without settling), the payoffs come from the
distribution after max_iter steps, so they depend on w_init and max_iter;
Trajectory.scores_converged records when this happens. Restarting keeps the
score range bounded by len(w_init) + max_iter, and makes the payoffs a function
of rho alone.

With warm_start=True the chains instead continue from the previous step's
distributions. Combined with a small max_iter this lets scores and strategy
frequencies evolve on comparable timescales, but the score range can then grow
without bound.
"""

from dataclasses import dataclass, field

import numpy as np

from .population import Population


@dataclass
class Trajectory:
    t: np.ndarray           # (T,) times
    rho: np.ndarray         # (T, n) strategy frequencies
    payoffs: np.ndarray     # (T, n) pi_i at each time
    w0: np.ndarray          # (T, n) fraction of each strategy with score 0
    mean_score: np.ndarray  # (T, n)
    scores_converged: np.ndarray  # (T,) whether the score chains converged at each time
    strategies: list
    at_rest: bool           # stopped because rho stopped changing
    w: list = field(default_factory=list)  # (n, L) score distributions, if recorded

    @property
    def mean_payoff(self):
        return (self.rho * self.payoffs).sum(axis=1)


class ReplicatorDynamics:
    def __init__(self, solver, payoffs, dt=0.1, warm_start=False, rest_tol=1e-9):
        """
        solver      StableScoreSolver for the score chains; its tol and max_iter
                    decide when each step's chains stop.
        payoffs     Payoffs (benefit, cost, role factor).
        dt          step size of rho.
        warm_start  False: restart the chains from w_init at every step;
                    True: continue from the previous step's distributions.
        rest_tol    stop when max |d rho / dt| falls to this (a rest point).
        """
        self.solver = solver
        self.payoffs = payoffs
        self.dt = dt
        self.warm_start = warm_start
        self.rest_tol = rest_tol

    def payoff_matrix(self, population, w):
        H = self.payoffs.help_probabilities(population, w)
        p = self.payoffs
        return p.role_factor * (p.benefit * H.T - p.cost * H)

    def velocity(self, population, w):
        """d rho / dt, and the payoffs pi."""
        rho = population.rho
        pi = self.payoff_matrix(population, w) @ rho
        return rho * (pi - rho @ pi), pi

    def run(self, strategies, rho, w_init, n_steps, record_scores=False):
        rho = np.asarray(rho, dtype=float)
        w = w_init
        rows = []
        scores = []
        at_rest = False
        for step in range(n_steps + 1):
            population = Population(strategies, rho)
            result = self.solver(population, w if self.warm_start else w_init)
            w = result.w
            drho, pi = self.velocity(population, w)
            converged = result.converged or result.transient
            rows.append((step * self.dt, rho, pi, w[:, 0], w @ np.arange(w.shape[1]), converged))
            if record_scores:
                scores.append(w)
            if np.abs(drho).max() <= self.rest_tol:
                at_rest = True
                break
            if step < n_steps:
                rho = self._advance(rho, drho)
        t, rho_t, pi_t, w0_t, mean_t, converged_t = (np.array(x) for x in zip(*rows))
        return Trajectory(t, rho_t, pi_t, w0_t, mean_t, converged_t, list(strategies), at_rest, scores)

    def _advance(self, rho, drho):
        """Euler step. A step that overshoots below 0 is clipped (the strategy goes extinct)."""
        rho = np.clip(rho + self.dt * drho, 0.0, None)
        return rho / rho.sum()
