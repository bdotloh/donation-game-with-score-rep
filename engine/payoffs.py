"""Expected payoffs in the donation game, given the score distributions."""

import numpy as np

from .rates import ROLE_FACTOR


class Payoffs:
    def __init__(self, benefit=2.0, cost=1.0, role_factor=ROLE_FACTOR):
        self.benefit = benefit
        self.cost = cost
        self.role_factor = role_factor

    def help_probabilities(self, population, w):
        """H[i, k] = Pr(a Can of strategy i helps a Need of strategy k)."""
        n, L = w.shape
        H = np.zeros((n, n))
        for i, strategy in enumerate(population.strategies):
            rc = strategy.partition.classify(np.arange(L))
            receiver_mass = np.stack(
                [np.bincount(rc, weights=w[k], minlength=strategy.partition.n_classes) for k in range(n)]
            )
            H[i] = receiver_mass @ strategy.donor_help(L) @ w[i]
        return H

    def __call__(self, population, w):
        """Expected payoff of each strategy (per interaction, including the role factor)."""
        H = self.help_probabilities(population, w)
        rho = population.rho
        benefit = H.T @ rho     # helped by Cans of each strategy
        cost = H @ rho          # helping Needs of each strategy
        return self.role_factor * (self.benefit * benefit - self.cost * cost)
