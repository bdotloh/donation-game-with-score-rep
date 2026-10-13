"""Partitions of the score set S = {0, 1, 2, ...} into classes.

Strategies and scoring norms only react to which class a score falls in.
A partition is given by increasing cutoffs: class c holds the scores j with
cutoffs[c-1] <= j < cutoffs[c] (with cutoffs[-1] = 0 and cutoffs[n] = inf).
"""

import numpy as np


class ScorePartition:
    def __init__(self, cutoffs):
        cutoffs = tuple(int(x) for x in cutoffs)
        if any(x <= 0 for x in cutoffs) or list(cutoffs) != sorted(set(cutoffs)):
            raise ValueError(f"cutoffs must be strictly increasing and positive; got {cutoffs}.")
        self.cutoffs = cutoffs

    @property
    def n_classes(self):
        return len(self.cutoffs) + 1

    def classify(self, j):
        """Class index of score(s) j."""
        return np.searchsorted(self.cutoffs, j, side="right")

    def class_masses(self, w):
        """Probability mass of the score distribution w in each class."""
        w = np.asarray(w, dtype=float)
        return np.bincount(self.classify(np.arange(len(w))), weights=w, minlength=self.n_classes)

    def __eq__(self, other):
        return isinstance(other, ScorePartition) and self.cutoffs == other.cutoffs

    def __hash__(self):
        return hash(self.cutoffs)

    def __repr__(self):
        return f"{type(self).__name__}(cutoffs={self.cutoffs})"


class PossessionPartition(ScorePartition):
    """Binary classes, as in the Frean and Marsland model: s = 0 / s >= 1."""

    def __init__(self):
        super().__init__((1,))

    def __repr__(self):
        return "PossessionPartition()"


class TernaryPartition(ScorePartition):
    """Ternary classes: s = 0 / 0 < s <= tau / s > tau."""

    def __init__(self, tau):
        if int(tau) < 1:
            raise ValueError(f"tau must be a positive integer; got {tau}.")
        self.tau = int(tau)
        super().__init__((1, self.tau + 1))

    def __repr__(self):
        return f"TernaryPartition(tau={self.tau})"
