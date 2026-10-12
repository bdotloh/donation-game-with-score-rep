"""Replicator dynamics of strategy frequencies, with scores evolving underneath.

    d rho_i / dt = rho_i (pi_i - pi_bar),   pi_i = (A rho)_i,   pi_bar = rho . A rho

The payoff matrix A depends on rho through the score distributions w:

    A[i, k] = role_factor (benefit H[k, i] - cost H[i, k]),

H[i, k] being the probability that a Can of strategy i helps a Need of strategy k.

Each step first updates the score distributions for the current rho, starting
from the previous ones. By default the chains run until they converge (scores
settle much faster than strategy frequencies change). With score_steps = k,
only k steps of the score chain are taken per step of rho, so both evolve on
comparable timescales.
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
    strategies: list
    at_rest: bool           # stopped because rho stopped changing
    w: list = field(default_factory=list)  # (n, L) score distributions, if recorded

    @property
    def mean_payoff(self):
        return (self.rho * self.payoffs).sum(axis=1)


class ReplicatorDynamics:
    def __init__(self, solver, payoffs, dt=0.1, score_steps=None, rest_tol=1e-9):
        """
        solver       StableScoreSolver for the score chains.
        payoffs      Payoffs (benefit, cost, role factor).
        dt           step size of rho.
        score_steps  None: run the score chains to convergence at each step;
                     k: take k steps of the score chains per step of rho.
        rest_tol     stop when max |d rho / dt| falls to this (a rest point).
        """
        self.solver = solver if score_steps is None else solver.with_options(max_iter=score_steps)
        self.payoffs = payoffs
        self.dt = dt
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
            w = self.solver(population, w).w
            drho, pi = self.velocity(population, w)
            rows.append((step * self.dt, rho, pi, w[:, 0], w @ np.arange(w.shape[1])))
            if record_scores:
                scores.append(w)
            if np.abs(drho).max() <= self.rest_tol:
                at_rest = True
                break
            if step < n_steps:
                rho = self._advance(rho, drho)
        t, rho_t, pi_t, w0_t, mean_t = (np.array(x) for x in zip(*rows))
        return Trajectory(t, rho_t, pi_t, w0_t, mean_t, list(strategies), at_rest, scores)

    def _advance(self, rho, drho):
        """Euler step. A step that overshoots below 0 is clipped (the strategy goes extinct)."""
        rho = np.clip(rho + self.dt * drho, 0.0, None)
        return rho / rho.sum()
