"""Evolutionarily stable strategies, following find_ESS in helper_funcs.jl.

Maynard Smith and Price: A is an ESS if, for every B != A, either
    E(A,A) >  E(B,A)                     (a rare B dies out in a population of A), or
    E(A,A) == E(B,A) and E(A,B) > E(B,B) (A would invade a population of B from rare).

To reproduce helper_funcs.jl, payoffs of a single-strategy population use the
distribution that generated the last rates (w_prev), while payoffs of a
two-strategy population use the final distribution (w).
"""

from .population import Population


class ESSFinder:
    def __init__(self, solver, payoffs, rho=0.99, fitness_threshold=1e-5, final_max_iter=100):
        """
        solver            StableScoreSolver used for every population.
        payoffs           Payoffs.
        rho               frequency of the resident A when tested against an invader B.
        fitness_threshold candidates whose 4 * payoff on their own is below this are skipped.
        final_max_iter    max_iter of the last check that an ESS is not a "boring" one
                          (helper_funcs.jl uses its default maxn of 100 here).
        """
        self.solver = solver
        self.final_solver = solver.with_options(max_iter=final_max_iter)
        self.payoffs = payoffs
        self.rho = rho
        self.fitness_threshold = fitness_threshold

    def payoff_alone(self, strategy, w_init, solver=None):
        population = Population([strategy])
        result = (solver or self.solver)(population, w_init)
        return self.payoffs(population, result.w_prev)[0]

    def payoffs_against(self, A, B, rho_A, w_init):
        population = Population([A, B], [rho_A, 1 - rho_A])
        result = self.solver(population, w_init)
        return self.payoffs(population, result.w)

    def is_ess(self, A, invaders, w_init):
        if 4 * self.payoff_alone(A, w_init) < self.fitness_threshold:
            return False
        for B in invaders:
            if B.name == A.name:
                continue
            fit_A, fit_B = self.payoffs_against(A, B, self.rho, w_init)
            if fit_A > fit_B:
                continue
            if fit_A < fit_B:
                return False
            # Equal when A is common: A must not lose when B is common.
            fit_A, fit_B = self.payoffs_against(A, B, 1 - self.rho, w_init)
            if fit_A < fit_B:
                return False
        return self.payoff_alone(A, w_init, self.final_solver) >= 1e-5

    def __call__(self, candidates, invaders, w_init):
        """The candidates that are ESS against every invader."""
        return [A for A in candidates if self.is_ess(A, invaders, w_init)]
