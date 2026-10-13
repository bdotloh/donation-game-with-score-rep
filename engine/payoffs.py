"""Expected payoffs in the donation game, given the score distributions."""

import numpy as np

from .rates import ROLE_FACTOR, _Cache


class Payoffs:
    def __init__(self, benefit=2.0, cost=1.0, role_factor=ROLE_FACTOR):
        self.benefit = benefit
        self.cost = cost
        self.role_factor = role_factor
        self._cache = _Cache()

    def help_probabilities(self, population, w):
        """H[i, k] = Pr(a Can of strategy i helps a Need of strategy k)."""
        _, onehot, help_ = self._cache.get(population.strategies).at(w.shape[1])
        return np.einsum("kc,icd,id->ik", w @ onehot, help_, w)

    def __call__(self, population, w):
        """Expected payoff of each strategy (per interaction, including the role factor)."""
        H = self.help_probabilities(population, w)
        rho = population.rho
        benefit = H.T @ rho     # helped by Cans of each strategy
        cost = H @ rho          # helping Needs of each strategy
        return self.role_factor * (self.benefit * benefit - self.cost * cost)
