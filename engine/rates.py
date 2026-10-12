"""Transition rates of the score process.

For strategy i, g_j^(i) is the rate from score j-1 to j, and f_j^(i) the rate
from j to j-1. Interactions alone give G_j^(i) and F_j^(i): each individual can
help (is a Can) with probability q and is in need otherwise, and a Can meets a
Need, so a given ordered role assignment has probability q(1 - q) (the role
factor; 1/4 for q = 1/2, as in helper_funcs.jl). Noise then gives
g = (1 - eps) G + eps eps_up and f = (1 - eps) F + eps eps_down.

Arrays use the document's indices: G[i, j] = G_j^(i) for j = 1..L and
F[i, j] = F_j^(i) for j = 1..L-1, where L is the number of scores tracked.
G[i, 0] and F[i, 0] are unused and set to 0.

The computation works on the coarsest partition of scores that refines the
norm's classes and every strategy's receiver classes, so it is a few array
operations per step whatever the strategies are.
"""

from dataclasses import dataclass

import numpy as np

from .partitions import ScorePartition

ROLE_FACTOR = 0.25


def role_factor(q):
    """Probability q(1 - q) of a given ordered (Can, Need) role assignment."""
    if not 0 < q < 1:
        raise ValueError(f"q must lie strictly between 0 and 1; got {q}.")
    return q * (1 - q)


@dataclass
class Rates:
    G: np.ndarray
    F: np.ndarray
    g: np.ndarray
    f: np.ndarray


class ClassStructure:
    """Score classes shared by a list of strategies (and optionally a norm's partition).

    Refined class c holds the scores in [lows[c], lows[c+1]). For n_scores scores:
    classes[j] is the refined class of score j, onehot[j, c] its indicator, and
    help[i, c, d] the probability that a Can of strategy i with score d helps a Need
    in refined class c.
    """

    def __init__(self, strategies, partitions=()):
        self.strategies = list(strategies)      # keeps them alive while cached by id
        cutoffs = sorted({x for s in self.strategies for x in s.partition.cutoffs}
                         | {x for p in partitions for x in p.cutoffs})
        self.partition = ScorePartition(cutoffs)
        self.lows = np.array([0] + cutoffs)
        self.capacity = 0
        self._grow(64)

    def _grow(self, n_scores):
        self.capacity = n_scores
        scores = np.arange(n_scores)
        self._classes = self.partition.classify(scores)
        self._onehot = np.eye(self.partition.n_classes)[self._classes]
        self._help = np.stack([s.donor_help(n_scores)[s.partition.classify(self.lows)] for s in self.strategies])

    def at(self, n_scores):
        if n_scores > self.capacity:
            self._grow(max(n_scores, 2 * self.capacity))
        return self._classes[:n_scores], self._onehot[:n_scores], self._help[:, :, :n_scores]


class _Cache:
    """A few ClassStructures, keyed by the strategies' identities."""

    def __init__(self, partitions=(), size=16):
        self.partitions = partitions
        self.size = size
        self._items = {}

    def __getstate__(self):
        return {"partitions": self.partitions, "size": self.size, "_items": {}}

    def get(self, strategies):
        key = tuple(id(s) for s in strategies)
        structure = self._items.get(key)
        if structure is None or any(a is not b for a, b in zip(structure.strategies, strategies)):
            if len(self._items) >= self.size:
                self._items.pop(next(iter(self._items)))
            structure = self._items[key] = ClassStructure(strategies, self.partitions)
        return structure


class TransitionRates:
    def __init__(self, norm, noise, role_factor=ROLE_FACTOR):
        self.norm = norm
        self.noise = noise
        self.role_factor = role_factor
        self._cache = _Cache((norm.partition,))
        self._rules = {}

    def __getstate__(self):
        return {**self.__dict__, "_rules": {}}

    def __call__(self, population, w):
        n, L = w.shape
        structure = self._cache.get(population.strategies)
        classes, onehot, help_ = structure.at(L)
        decline_cn, change_cn, decline_nc, change_nc, norm_onehot = self._rule_arrays(structure, L)

        rho = population.rho
        w_bar = rho @ w                                    # population-wide score distribution
        w_bar_class = w_bar @ onehot                       # ... per refined class

        # Focal player as Can (donor) at score j, meeting a Need from the whole population:
        # p(dS) = sum_c w_bar_class[c] (decline[c, j] + help[i, c, j] (give - decline)[c, j]).
        weighted = help_ * w_bar_class[:, None]                                   # [i, c, j]
        donor = (w_bar_class @ decline_cn)[:, None, :] + (weighted[None] * change_cn[:, None]).sum(axis=2)

        # Focal player as Need at score j, meeting Cans of each strategy (the same for every focal strategy).
        # helped[c, x]: mass of Cans in norm class x (over all strategies) who help a Need in class c.
        helped = (help_ * (rho[:, None] * w)[:, None, :]).sum(axis=0) @ norm_onehot
        receiver = decline_nc @ (w_bar @ norm_onehot) + (helped[classes][None] * change_nc).sum(axis=2)

        rates = self.role_factor * (donor + receiver[:, None, :])          # [up/down, i, j]
        G = np.zeros((n, L + 1))
        G[:, 1:] = rates[0]
        F = rates[1]
        F[:, 0] = 0.0

        eps, eps_up, eps_down = self.noise.eps, self.noise.eps_up, self.noise.eps_down
        g = (1 - eps) * G + eps * eps_up
        f = (1 - eps) * F + eps * eps_down
        g[:, 0] = 0.0
        f[:, 0] = 0.0
        return Rates(G=G, F=F, g=g, f=f)

    def _rule_arrays(self, structure, L):
        """Indicators of dS = +1 (index 0) and -1 (index 1) from the norm, per (refined class, score)."""
        key = id(structure)
        cached = self._rules.get(key)
        if cached is None or cached[0] is not structure or cached[1] < L:
            capacity = max(L, structure.capacity)
            nc = self.norm.partition.classify(np.arange(capacity))           # norm class of each score
            nc_of_class = self.norm.partition.classify(structure.lows)       # norm class of each refined class
            signs = np.array([1, -1])[:, None, None]
            CN, NC = self.norm.CN, self.norm.NC
            # CN[a, act, x]: Need in class a, Can in class x.
            give_cn = (CN[:, 1, :][None] == signs)[:, nc_of_class][:, :, nc]     # [s, c (Need), j (Can)]
            decline_cn = (CN[:, 0, :][None] == signs)[:, nc_of_class][:, :, nc]
            # NC[a, act, x]: Need at score j (class a), Can in norm class x.
            give_nc = (NC[:, 1, :][None] == signs)[:, nc, :]                      # [s, j (Need), x (Can)]
            decline_nc = (NC[:, 0, :][None] == signs)[:, nc, :]
            cached = (structure, capacity, decline_cn * 1.0, (give_cn * 1.0 - decline_cn),
                      decline_nc * 1.0, (give_nc * 1.0 - decline_nc), np.eye(self.norm.partition.n_classes)[nc])
            if len(self._rules) >= 16:
                self._rules.pop(next(iter(self._rules)))
            self._rules[key] = cached
        _, _, decline_cn, change_cn, decline_nc, change_nc, norm_onehot = cached
        return (decline_cn[:, :, :L], change_cn[:, :, :L], decline_nc[:, :L], change_nc[:, :L], norm_onehot[:L])
