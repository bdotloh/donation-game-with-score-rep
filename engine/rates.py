"""Transition rates of the score process.

For strategy i, g_j^(i) is the rate from score j-1 to j, and f_j^(i) the rate
from j to j-1. Interactions alone give G_j^(i) and F_j^(i): a random pair meets
with one as Can (donor) and one as Need (receiver), which happens with
probability 1/4 for a given ordered role assignment. Noise then gives
g = (1 - eps) G + eps eps_up and f = (1 - eps) F + eps eps_down.

Arrays use the document's indices: G[i, j] = G_j^(i) for j = 1..L and
F[i, j] = F_j^(i) for j = 1..L-1, where L is the number of scores tracked.
G[i, 0] and F[i, 0] are unused and set to 0.
"""

from dataclasses import dataclass

import numpy as np

ROLE_FACTOR = 0.25


@dataclass
class Rates:
    G: np.ndarray
    F: np.ndarray
    g: np.ndarray
    f: np.ndarray


class TransitionRates:
    def __init__(self, norm, noise):
        self.norm = norm
        self.noise = noise

    def __call__(self, population, w):
        n, L = w.shape
        scores = np.arange(L)
        partition = self.norm.partition
        nc = partition.classify(scores)                 # norm class of each score
        onehot = np.eye(partition.n_classes)[nc]        # (L, n_classes)
        w_bar = population.rho @ w                      # population-wide score distribution
        mass = w @ onehot                               # mass of each strategy in each norm class

        G = np.zeros((n, L + 1))
        F = np.zeros((n, L))
        for dS, out in ((1, G[:, 1:]), (-1, F)):
            receiver_term = self._as_receiver(population, w, nc, onehot, mass, dS)
            for i, strategy in enumerate(population.strategies):
                out[i] = ROLE_FACTOR * (self._as_donor(strategy, w_bar, nc, dS) + receiver_term)
        F[:, 0] = 0.0

        eps, eps_up, eps_down = self.noise.eps, self.noise.eps_up, self.noise.eps_down
        g = (1 - eps) * G + eps * eps_up
        f = (1 - eps) * F + eps * eps_down
        g[:, 0] = 0.0
        f[:, 0] = 0.0
        return Rates(G=G, F=F, g=g, f=f)

    def _as_donor(self, strategy, w_bar, nc, dS):
        """Pr(dS) for a focal Can of this strategy at each score j, summed over Need players."""
        L = len(nc)
        n_classes = self.norm.partition.n_classes
        rc = strategy.partition.classify(np.arange(L))
        B = strategy.partition.n_classes
        # Mass of receivers in (norm class a, strategy receiver class b).
        M = np.bincount(nc * B + rc, weights=w_bar, minlength=n_classes * B).reshape(n_classes, B)
        give = (self.norm.CN[:, 1, :] == dS)[:, nc]     # [a, j]: Need in class a, Can at score j
        decline = (self.norm.CN[:, 0, :] == dS)[:, nc]
        help_ = strategy.donor_help(L)                  # [b, j]
        return ((M.T @ give) * help_ + (M.T @ decline) * (1 - help_)).sum(axis=0)

    def _as_receiver(self, population, w, nc, onehot, mass, dS):
        """Pr(dS) for a focal Need at each score j, summed over Can players.

        This depends only on the donors' strategies, so it is shared by all focal strategies.
        """
        L = len(nc)
        give_rule = (self.norm.NC[:, 1, :] == dS)[nc]   # [j, x]: Need at score j, Can in class x
        decline_rule = (self.norm.NC[:, 0, :] == dS)[nc]
        total = np.zeros(L)
        for k, strategy in enumerate(population.strategies):
            help_ = strategy.donor_help(L)
            rc = strategy.partition.classify(np.arange(L))
            helped = ((help_ * w[k]) @ onehot)[rc]      # [j, x]: mass of k-Cans in class x who help
            declined = mass[k] - helped
            total += population.rho[k] * (helped * give_rule + declined * decline_rule).sum(axis=1)
        return total
