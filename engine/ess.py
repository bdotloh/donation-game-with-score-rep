"""Evolutionarily stable strategies.

Maynard Smith and Price: A is an ESS if, for every B != A, either
    E(A,A) >  E(B,A)                     (a rare B dies out in a population of A), or
    E(A,A) == E(B,A) and E(A,B) > E(B,B) (A would invade a population of B from rare).

Payoffs are always computed from the final score distribution (result.w).
"""

from .population import Population


class ESSFinder:
    def __init__(self, solver, payoffs, rho=0.99, fitness_threshold=1e-5, final_max_iter=100, tie_tol=0.0):
        """
        solver            StableScoreSolver used for every population.
        payoffs           Payoffs.
        rho               frequency of the resident A when tested against an invader B.
        fitness_threshold candidates whose 4 * payoff on their own is below this are skipped.
        final_max_iter    max_iter of the last check that an ESS is not a "boring" one
                          (default 100).
        tie_tol           two payoffs closer than this count as equal. 0 (the default) compares them exactly, so
                          when strategies that differ only in what they do in situations that never come up earn the
                          same payoff up to rounding (about 1e-16), rounding decides whether one invades the other;
                          a tolerance such as 1e-12 treats those as the ties they are.
        """
        self.solver = solver
        self.final_solver = solver.with_options(max_iter=final_max_iter)
        self.payoffs = payoffs
        self.rho = rho
        self.fitness_threshold = fitness_threshold
        self.tie_tol = tie_tol

    def payoff_alone(self, strategy, w_init, solver=None):
        population = Population([strategy])
        result = (solver or self.solver)(population, w_init)
        return self.payoffs(population, result.w)[0]

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
            if fit_A > fit_B + self.tie_tol:
                continue
            if fit_A < fit_B - self.tie_tol:
                return False
            # Equal when A is common: A must not lose when B is common.
            fit_A, fit_B = self.payoffs_against(A, B, 1 - self.rho, w_init)
            if fit_A < fit_B - self.tie_tol:
                return False
        return self.payoff_alone(A, w_init, self.final_solver) >= 1e-5

    def __call__(self, candidates, invaders, w_init):
        """The candidates that are ESS against every invader."""
        return [A for A in candidates if self.is_ess(A, invaders, w_init)]
