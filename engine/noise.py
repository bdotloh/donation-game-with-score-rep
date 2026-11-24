"""Rates not caused by interactions, and errors in actions.

Score transitions: g_j = G_j + eta_up(j) (up from score j), f_j = F_j + eta_down(j) (down from score j >= 1,
f_0 = 0), where G, F come from interactions (rates.py). eta_up and eta_down are rates of gaining and losing
a token by other means: they are rates, not probabilities (they need not lie in [0, 1] or sum to 1), only >= 0.
Each is a number, an array over the scores (the last value holds for higher scores) or a function j -> rate,
for example eta_down = lambda j: 0.01 * j.

One solver step is one unit of time, so keep the total rate out of a score (G + F + eta_up + eta_down) below 1;
above 1 a step can leave a score with negative mass.

epsilon is the probability that a donor's action is the opposite of the intended one: the help probability
becomes (1 - epsilon) help + epsilon (1 - help).
"""

from dataclasses import dataclass, field

import numpy as np


def _as_rate(value, name):
    """A number >= 0, an array of them, or a callable (checked when evaluated)."""
    if callable(value):
        return value
    array = np.asarray(value, dtype=float)
    if array.ndim > 1 or array.size == 0 or not np.all(array >= 0) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a number >= 0, a 1-D array of them or a function of the score; got {value}.")
    return float(array) if array.ndim == 0 else array


def _table(rate, name, n_scores):
    """The rates at scores 0 .. n_scores - 1."""
    j = np.arange(n_scores)
    if callable(rate):
        values = np.array([rate(int(x)) for x in j], dtype=float)
        if not np.all(values >= 0) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name}(j) must be >= 0 for every score j.")
        return values
    if np.ndim(rate) == 0:
        return np.full(n_scores, rate)
    return np.asarray(rate)[np.minimum(j, len(rate) - 1)]


@dataclass(frozen=True)
class Noise:
    eta_up: object = 0.0
    eta_down: object = 0.0
    epsilon: float = 0.0
    _tables: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "eta_up", _as_rate(self.eta_up, "eta_up"))
        object.__setattr__(self, "eta_down", _as_rate(self.eta_down, "eta_down"))
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError(f"epsilon must lie in [0, 1]; got {self.epsilon}.")

    def up(self, n_scores):
        """eta_up(j) for j = 0 .. n_scores - 1 (evaluated once, then reused)."""
        return self._rates("up", self.eta_up, n_scores)

    def down(self, n_scores):
        """eta_down(j) for j = 0 .. n_scores - 1."""
        return self._rates("down", self.eta_down, n_scores)

    def _rates(self, which, rate, n_scores):
        table = self._tables.get(which)
        if table is None or len(table) < n_scores:
            table = self._tables[which] = _table(rate, f"eta_{which}", max(n_scores, 2 * (0 if table is None else len(table))))
        return table[:n_scores]

    def act(self, help_):
        """Help probability after action errors."""
        return (1 - self.epsilon) * help_ + self.epsilon * (1 - help_)

    @classmethod
    def from_old(cls, eps, eps_up=0.5, eps_down=0.5):
        """The old mixture g = (1 - eps) G + eps eps_up, f = (1 - eps) F + eps eps_down as additive rates:
        eta = eps eps_ud / (1 - eps), so that g and f equal the old rates divided by 1 - eps. The ratios
        g_{j-1} / f_j, and so every stationary distribution, are the same; with the same lambda the chain
        runs 1 / (1 - eps) times faster (multiply lambda and both eta by 1 - eps to get the old speed too)."""
        if not 0.0 <= eps < 1.0:
            raise ValueError(f"eps must lie in [0, 1); got {eps}.")
        scale = eps / (1 - eps)
        return cls(eta_up=scale * eps_up, eta_down=scale * eps_down)
