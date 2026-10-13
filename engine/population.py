"""A well-mixed population of n strategies with frequencies rho."""

import numpy as np


class Population:
    def __init__(self, strategies, rho=None):
        self.strategies = list(strategies)
        n = len(self.strategies)
        if n == 0:
            raise ValueError("A population needs at least one strategy.")
        rho = np.full(n, 1.0 / n) if rho is None else np.asarray(rho, dtype=float)
        if rho.shape != (n,) or np.any(rho < 0) or not np.isclose(rho.sum(), 1.0):
            raise ValueError(f"rho must be {n} non-negative frequencies summing to 1; got {rho}.")
        self.rho = rho

    @property
    def n(self):
        return len(self.strategies)

    def __repr__(self):
        parts = ", ".join(f"{s.name}: {r:g}" for s, r in zip(self.strategies, self.rho))
        return f"Population({parts})"


def score_array(w_init, n):
    """Score distributions as an (n, L) array.

    w_init is either one distribution, shared by all n strategies, or one per
    strategy (padded with zeros to a common length).
    """
    if np.ndim(w_init[0]) == 0:
        rows = [np.asarray(w_init, dtype=float)] * n
    else:
        rows = [np.asarray(row, dtype=float) for row in w_init]
        if len(rows) != n:
            raise ValueError(f"Expected {n} score distributions; got {len(rows)}.")
    L = max(len(row) for row in rows)
    w = np.zeros((n, L))
    for i, row in enumerate(rows):
        if np.any(row < 0) or not np.isclose(row.sum(), 1.0):
            raise ValueError(f"Score distribution {i} must be non-negative and sum to 1.")
        w[i, : len(row)] = row
    return w
