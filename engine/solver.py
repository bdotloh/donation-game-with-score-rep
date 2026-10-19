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
    converged: bool         # change fell to the tolerance, and no chain is drifting
    transient: bool         # every chain left its initial scores (chain is transient)
    history: list = field(default_factory=list)
    drifting: np.ndarray = None  # (n,) up-rate >= down-rate at the top score: scores keep spreading upwards

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
        tail_tol=0.0,
        max_score=None,
    ):
        """max_score: if given, scores above it do not exist (a donor at max_score who would gain a score stays).
        tail_tol > 0 drops the top scores whose mass is below tail_tol in every chain after each
        step (and renormalises), so long runs don't carry thousands of near-empty scores."""
        self.norm = norm
        self.noise = noise or Noise()
        self.rates = TransitionRates(norm, self.noise, role_factor)
        self.tol = tol
        self.max_iter = max_iter
        self.stop_when_transient = stop_when_transient
        self.force_max_iter = force_max_iter
        self.record_history = record_history
        self.tail_tol = tail_tol
        self.max_score = max_score

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
            if self.max_score is not None and w.shape[1] > self.max_score + 1:
                w = w[:, :self.max_score + 1]
                w = w / w.sum(axis=1, keepdims=True)
            if self.tail_tol > 0:
                w = _trim(w, self.tail_tol, n_init)
            if self.record_history:
                history.append((w, rates))
            transient = self.stop_when_transient and np.all(w[:, :n_init].sum(axis=1) == 0.0)
            converged = np.abs(dw).max() <= self.tol
            if (transient or converged) and not self.force_max_iter:
                break
        drifting = np.zeros(population.n, dtype=bool)
        if rates is not None:
            # Above the highest class cutoff the rates no longer change with the score, so if the
            # up-rate at the top is at least the down-rate there is no stable distribution: the
            # change per step only shrinks slowly, and a small change does not mean convergence.
            up, down = rates.g[:, -1], rates.f[:, -1]
            drifting = (up > 0) & (up >= down * (1 - 1e-9)) & (self.max_score is None)   # equal up to round-off counts as drifting
            converged = converged and not drifting.any()
        return StableScores(
            drifting=drifting,
            w=w,
            w_prev=w_prev,
            rates=rates,
            n_iter=n_iter,
            converged=converged,
            transient=transient,
            history=history,
        )


def _trim(w, tail_tol, n_keep):
    """Drop the top scores whose mass is below tail_tol in every chain (keeping at least n_keep)."""
    significant = np.flatnonzero((w > tail_tol).any(axis=0))
    keep = max(n_keep, significant[-1] + 1 if significant.size else 1)
    if keep >= w.shape[1]:
        return w
    w = w[:, :keep]
    return w / w.sum(axis=1, keepdims=True)
