"""Replicator dynamics on top of the stable score distributions."""

import numpy as np

from engine import (
    BinaryStrategy,
    Defectors,
    Discriminators,
    Noise,
    Payoffs,
    PhiDisc,
    Population,
    ReplicatorDynamics,
    ScoringNorm,
    StableScoreSolver,
)


def test_payoff_matrix_gives_payoffs():
    norm = ScoringNorm.from_barcode("CN:00000-0+ NC:00000+-0")
    population = Population([PhiDisc(1), Defectors(), BinaryStrategy.from_name("0,g|0.5,g")], [0.5, 0.2, 0.3])
    solver = StableScoreSolver(norm, Noise(0.1, 0.3, 0.5))
    dynamics = ReplicatorDynamics(solver, Payoffs())
    w = solver(population, [0.5, 0.5]).w
    A = dynamics.payoff_matrix(population, w)
    np.testing.assert_allclose(A @ population.rho, Payoffs()(population, w), atol=1e-15)
    drho, _ = dynamics.velocity(population, w)
    np.testing.assert_allclose(drho.sum(), 0.0, atol=1e-15)


def test_cooperators_vs_defectors_analytic():
    """Unconditional strategies ignore scores: pi_C - pi_D = -cost/4, so rho_C is logistic."""
    norm = ScoringNorm.from_barcode("CN:00000-00 NC:00000000")
    dynamics = ReplicatorDynamics(StableScoreSolver(norm, Noise(0.1)), Payoffs(benefit=2, cost=1), dt=0.01)
    trajectory = dynamics.run([BinaryStrategy.from_name("g,g|g,g"), Defectors()], [0.9, 0.1], [0.5, 0.5], 2000)
    rho0 = 0.9
    expected = 1 / (1 + (1 - rho0) / rho0 * np.exp(0.25 * trajectory.t))
    np.testing.assert_allclose(trajectory.rho[:, 0], expected, atol=1e-3)
    np.testing.assert_allclose(trajectory.rho.sum(axis=1), 1.0, atol=1e-12)


def test_coupled_timescales_approach_quasi_static():
    norm = ScoringNorm.from_barcode("CN:00000-00 NC:00000000")
    solver = StableScoreSolver(norm, Noise(0.2, 0.1, 0.3))
    strategies, rho0 = [Discriminators(), Defectors()], [0.5, 0.5]
    quasi = ReplicatorDynamics(solver, Payoffs(), dt=0.5).run(strategies, rho0, [0.5, 0.5], 200)
    coupled = ReplicatorDynamics(solver, Payoffs(), dt=0.5, score_steps=50).run(strategies, rho0, [0.5, 0.5], 200)
    np.testing.assert_allclose(coupled.rho[-1], quasi.rho[-1], atol=1e-3)


def test_extinct_strategies_stay_extinct_and_rest_points_stop():
    norm = ScoringNorm.from_barcode("CN:00000-00 NC:00000000")
    dynamics = ReplicatorDynamics(StableScoreSolver(norm, Noise(0.2, 0.1, 0.3)), Payoffs(), dt=0.5)
    trajectory = dynamics.run([Discriminators(), Defectors()], [1.0, 0.0], [0.5, 0.5], 100)
    assert trajectory.at_rest and len(trajectory.t) == 1
    np.testing.assert_array_equal(trajectory.rho[-1], [1.0, 0.0])
