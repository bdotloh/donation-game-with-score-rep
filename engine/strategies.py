"""Strategies for the donation game with integer scores.

Every strategy is described by its help probabilities. A strategy reacts to
the receiver's score only through the class it falls in (under the strategy's
own partition), but may react to the donor's exact score j. So a strategy
supplies

    donor_help(n_scores)[c, j] = Pr(donor with score j helps | receiver in class c)

for j = 0, ..., n_scores - 1. This covers

    BinaryStrategy   classes s = 0 / s >= 1, a 2x2 table of probabilities
    TernaryStrategy  classes s = 0 / 0 < s <= tau / s > tau, a 3x3 table
    PhiDisc          helps a receiver with positive score with probability
                     j^(-phi) when the donor has score j >= 1, never otherwise

Strategy names follow the barcode convention of helper_funcs.jl: rows are the
receiver's class, columns the donor's class, '0' declines and 'g' gives, e.g.
"0,0|0,g" helps only when both have positive score (Discriminators).
Non-deterministic entries are written as numbers, e.g. "0,0|0.5,1".
"""

from abc import ABC, abstractmethod

import numpy as np

from .partitions import PossessionPartition, TernaryPartition


class Strategy(ABC):
    def __init__(self, partition):
        self.partition = partition
        self._help_cache = np.zeros((partition.n_classes, 0))

    def donor_help(self, n_scores):
        """Array [c, j]: probability of helping a class-c receiver at donor score j."""
        if self._help_cache.shape[1] < n_scores:
            self._help_cache = self._compute_donor_help(max(n_scores, 2 * self._help_cache.shape[1]))
        return self._help_cache[:, :n_scores]

    @abstractmethod
    def _compute_donor_help(self, n_scores):
        ...

    @property
    @abstractmethod
    def name(self):
        ...

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"


class ClassStrategy(Strategy):
    """Help probabilities depend only on the classes of the receiver and donor.

    P[c_receiver, c_donor] is the probability of helping.
    """

    def __init__(self, P, partition):
        P = np.asarray(P, dtype=float)
        n = partition.n_classes
        if P.shape != (n, n):
            raise ValueError(f"P must have shape {(n, n)}; got {P.shape}.")
        if np.any(P < 0) or np.any(P > 1):
            raise ValueError("Help probabilities must lie in [0, 1].")
        super().__init__(partition)
        self.P = P

    def _compute_donor_help(self, n_scores):
        return self.P[:, self.partition.classify(np.arange(n_scores))]

    @property
    def name(self):
        return format_table(self.P)


class BinaryStrategy(ClassStrategy):
    def __init__(self, P):
        super().__init__(P, PossessionPartition())

    @classmethod
    def from_name(cls, name):
        return cls(parse_table(name, 2))


class TernaryStrategy(ClassStrategy):
    def __init__(self, P, tau):
        super().__init__(P, TernaryPartition(tau))
        self.tau = self.partition.tau

    @classmethod
    def from_name(cls, name, tau):
        return cls(parse_table(name, 3), tau)

    @classmethod
    def from_binary(cls, strategy, tau):
        """Ternary strategy that treats both positive classes like s >= 1."""
        lift = [0, 1, 1]
        return cls(strategy.P[np.ix_(lift, lift)], tau)

    @property
    def name(self):
        return f"{format_table(self.P)} (tau={self.tau})"


class PhiDisc(Strategy):
    """phi-Disc: don't help when score-less, else help a receiver with positive
    score with probability j^(-phi), j being the donor's score (phi >= 0)."""

    def __init__(self, phi):
        if phi < 0:
            raise ValueError(f"phi must be >= 0; got {phi}.")
        super().__init__(PossessionPartition())
        self.phi = float(phi)

    def _compute_donor_help(self, n_scores):
        help_ = np.zeros((2, n_scores))
        j = np.arange(1, n_scores)
        help_[1, 1:] = j.astype(float) ** (-self.phi)
        return help_

    @property
    def name(self):
        return f"phi-Disc(phi={self.phi:g})"


def Defectors():
    """Never help."""
    return BinaryStrategy.from_name("0,0|0,0")


def Discriminators():
    """Help only when both donor and receiver have a positive score."""
    return BinaryStrategy.from_name("0,0|0,g")


def parse_table(name, n):
    rows = name.split("|")
    if len(rows) != n or any(len(row.split(",")) != n for row in rows):
        raise ValueError(f"Expected {n} rows of {n} entries in {name!r}.")
    symbols = {"0": 0.0, "g": 1.0}
    return np.array(
        [[symbols[x] if x in symbols else float(x) for x in row.split(",")] for row in rows]
    )


def format_table(P):
    def entry(p):
        if p == 0:
            return "0"
        if p == 1:
            return "g"
        return f"{p:g}"

    return "|".join(",".join(entry(p) for p in row) for row in P)
