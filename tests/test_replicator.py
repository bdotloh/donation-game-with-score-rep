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
    solver = StableScoreSolver(norm, Noise(0.1, 0.3, 0.5), max_iter=500)
    dynamics = ReplicatorDynamics(solver, Payoffs())
    w = solver(population, [0.5, 0.5]).w
    A = dynamics.payoff_matrix(population, w)
    np.testing.assert_allclose(A @ population.rho, Payoffs()(population, w), atol=1e-15)
    drho, _ = dynamics.velocity(population, w)
    np.testing.assert_allclose(drho.sum(), 0.0, atol=1e-15)


def test_cooperators_vs_defectors_analytic():
    """Unconditional strategies ignore scores: pi_C - pi_D = -cost/4, so rho_C is logistic."""
    norm = ScoringNorm.from_barcode("CN:00000-00 NC:00000000")
    solver = StableScoreSolver(norm, Noise(0.1), max_iter=20)  # payoffs ignore scores here
    dynamics = ReplicatorDynamics(solver, Payoffs(benefit=2, cost=1), dt=0.01)
    trajectory = dynamics.run([BinaryStrategy.from_name("g,g|g,g"), Defectors()], [0.9, 0.1], [0.5, 0.5], 2000)
    rho0 = 0.9
    expected = 1 / (1 + (1 - rho0) / rho0 * np.exp(0.25 * trajectory.t))
    np.testing.assert_allclose(trajectory.rho[:, 0], expected, atol=1e-3)
    np.testing.assert_allclose(trajectory.rho.sum(axis=1), 1.0, atol=1e-12)


def test_restart_matches_warm_start_when_scores_settle():
    norm = ScoringNorm.from_barcode("CN:00000-00 NC:00000000")
    solver = StableScoreSolver(norm, Noise(0.2, 0.1, 0.3), tol=1e-8)
    strategies, rho0 = [Discriminators(), Defectors()], [0.5, 0.5]
    restart = ReplicatorDynamics(solver, Payoffs(), dt=1.0).run(strategies, rho0, [0.5, 0.5], 40)
    warm = ReplicatorDynamics(solver, Payoffs(), dt=1.0, warm_start=True).run(strategies, rho0, [0.5, 0.5], 40)
    assert restart.scores_converged.all()
    np.testing.assert_allclose(warm.rho, restart.rho, atol=1e-5)


def test_coupled_timescales_approach_quasi_static():
    norm = ScoringNorm.from_barcode("CN:00000-00 NC:00000000")
    solver = StableScoreSolver(norm, Noise(0.2, 0.1, 0.3))
    strategies, rho0 = [Discriminators(), Defectors()], [0.5, 0.5]
    quasi = ReplicatorDynamics(solver, Payoffs(), dt=0.5).run(strategies, rho0, [0.5, 0.5], 200)
    coupled = ReplicatorDynamics(solver.with_options(max_iter=50), Payoffs(), dt=0.5, warm_start=True)
    coupled = coupled.run(strategies, rho0, [0.5, 0.5], 200)
    np.testing.assert_allclose(coupled.rho[-1], quasi.rho[-1], atol=1e-3)


def test_restart_bounds_the_score_range():
    """Discriminators' scores drift upwards for ever under CN:00000-0+; each step restarts from w_init."""
    norm = ScoringNorm.from_barcode("CN:00000-0+ NC:00000000")
    solver = StableScoreSolver(norm, Noise(0.1, 0.2, 0.4), max_iter=200)
    dynamics = ReplicatorDynamics(solver, Payoffs(), dt=0.5)
    trajectory = dynamics.run([Discriminators(), Defectors()], [0.5, 0.5], [0.5, 0.5], 20, record_scores=True)
    assert not trajectory.scores_converged.any()
    assert max(w.shape[1] for w in trajectory.w) <= 2 + 200


def test_extinct_strategies_stay_extinct_and_rest_points_stop():
    norm = ScoringNorm.from_barcode("CN:00000-00 NC:00000000")
    dynamics = ReplicatorDynamics(StableScoreSolver(norm, Noise(0.2, 0.1, 0.3)), Payoffs(), dt=0.5)
    trajectory = dynamics.run([Discriminators(), Defectors()], [1.0, 0.0], [0.5, 0.5], 100)
    assert trajectory.at_rest and len(trajectory.t) == 1
    np.testing.assert_array_equal(trajectory.rho[-1], [1.0, 0.0])


def test_role_factor_scales_interaction_rates():
    """Without noise, rates and payoffs scale with q(1 - q)."""
    from engine.rates import role_factor

    norm = ScoringNorm.from_barcode("CN:00000-0+ NC:00000+-0")
    population = Population([Discriminators(), Defectors()], [0.6, 0.4])
    w = np.array([[0.3, 0.4, 0.3], [0.5, 0.25, 0.25]])
    base = StableScoreSolver(norm, Noise(0)).rates(population, w)
    q = 0.2
    scaled = StableScoreSolver(norm, Noise(0), role_factor=role_factor(q)).rates(population, w)
    np.testing.assert_allclose(scaled.G, base.G * q * (1 - q) / 0.25, atol=1e-15)
    np.testing.assert_allclose(scaled.F, base.F * q * (1 - q) / 0.25, atol=1e-15)
    pay = Payoffs(role_factor=role_factor(q))(population, w)
    np.testing.assert_allclose(pay, Payoffs()(population, w) * q * (1 - q) / 0.25, atol=1e-15)
