"""Stable score distributions, found by running the score chains until they converge.

All strategies' chains are stepped together, since each chain's rates depend on
every strategy's score distribution. The score range grows as needed; there is
no fixed maximum score. The loop matches calc_stable_w_1strategies and
calc_stable_w_2strategies in helper_funcs.jl, generalised to n strategies.
"""

import copy
from dataclasses import dataclass, field

import numpy as np

from . import chain
from .noise import Noise
from .population import score_array
from .rates import ROLE_FACTOR, TransitionRates


@dataclass
class StableScores:
    w: np.ndarray           # (n, L) score distributions at the end of the run
    w_prev: np.ndarray      # distributions that the last rates were computed from
    rates: object           # Rates computed from w_prev (None if no step was taken)
    n_iter: int
    converged: bool         # change fell to the tolerance
    transient: bool         # every chain left its initial scores (chain is transient)
    history: list = field(default_factory=list)

    @property
    def w0(self):
        """Fraction of each strategy with score 0."""
        return self.w[:, 0]


class StableScoreSolver:
    def __init__(
        self,
        norm,
        noise=None,
        tol=1e-5,
        max_iter=10_000,
        stop_when_transient=True,
        force_max_iter=False,
        record_history=False,
        role_factor=ROLE_FACTOR,
    ):
        self.norm = norm
        self.noise = noise or Noise()
        self.rates = TransitionRates(norm, self.noise, role_factor)
        self.tol = tol
        self.max_iter = max_iter
        self.stop_when_transient = stop_when_transient
        self.force_max_iter = force_max_iter
        self.record_history = record_history

    def with_options(self, **options):
        """Copy of this solver with some settings (tol, max_iter, ...) changed."""
        solver = copy.copy(self)
        for name, value in options.items():
            if not hasattr(solver, name) or name in ("norm", "noise", "rates", "role_factor"):
                raise ValueError(f"Cannot change {name!r}; make a new solver instead.")
            setattr(solver, name, value)
        return solver

    def __call__(self, population, w_init):
        w = score_array(w_init, population.n)
        n_init = w.shape[1]
        w_prev, rates = w, None
        converged = transient = False
        history = []
        n_iter = 0
        while n_iter < self.max_iter:
            n_iter += 1
            w_prev = chain.extend(w)
            rates = self.rates(population, w_prev)
            dw = chain.change(w_prev, rates.g, rates.f)
            w = w_prev + dw
            w = w / w.sum(axis=1, keepdims=True)
            if self.record_history:
                history.append((w, rates))
            transient = self.stop_when_transient and np.all(w[:, :n_init].sum(axis=1) == 0.0)
            converged = np.abs(dw).max() <= self.tol
            if (transient or converged) and not self.force_max_iter:
                break
        return StableScores(
            w=w,
            w_prev=w_prev,
            rates=rates,
            n_iter=n_iter,
            converged=converged,
            transient=transient,
            history=history,
        )
