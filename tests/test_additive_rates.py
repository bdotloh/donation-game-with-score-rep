"""Additive rates g = G + eta_up(j), f = F + eta_down(j) (f_0 = 0), and action errors epsilon (engine/noise.py)."""

import independent_model as im
import numpy as np
import pytest

from engine import (BinaryStrategy, Defectors, Discriminators, Noise, Payoffs, PhiDisc, Population, ScoringNorm,
                    StableScoreSolver, TransitionRates, chain)

EPS, EPS_UP, EPS_DOWN, LAMBDA = 0.1, 0.49, 0.51, 0.25
NORMS = {                                       # the norms of the talk
    "Money": "CN:000000++ NC:000000--",
    "phi-Disc norm": "CN:00000-0+ NC:00000000",
    "Coercive money": "CN:00000-++ NC:00000+--",
}


def old_stationary(norm, strategies, rho, eps, eps_up, eps_down, steps=40_000):
    """The old mixture g = (1 - eps) G + eps eps_up, f = (1 - eps) F + eps eps_down, iterated to its fixed point."""
    population = Population(strategies, rho)
    rates = TransitionRates(norm, Noise())                  # without extra rates: g = G, f = F
    w = np.full((len(strategies), 2), 0.5)
    for _ in range(steps):
        w_ext = chain.extend(w)
        R = rates(population, w_ext)
        g, f = (1 - eps) * R.G + eps * eps_up, (1 - eps) * R.F + eps * eps_down
        f[:, 0] = 0.0
        w_new = w_ext + chain.change(w_ext, g, f)
        w_new /= w_new.sum(axis=1, keepdims=True)
        keep = np.flatnonzero((w_new > 1e-15).any(axis=0))[-1] + 1
        w_new = w_new[:, :max(keep, 2)]
        done = w_new.shape == w.shape and np.abs(w_new - w).max() < 1e-12
        w = w_new
        if done:
            break
    return w


def new_stationary(norm, strategies, rho, noise):
    solver = StableScoreSolver(norm, noise, tol=1e-12, max_iter=60_000, tail_tol=1e-15)
    result = solver(Population(strategies, rho), np.full((len(strategies), 2), 0.5))
    return result.w


@pytest.mark.parametrize("name", NORMS)
def test_from_old_has_the_old_stationary_distributions(name):
    """eta = eps eps_ud / (1 - eps): g_{j-1} / f_j, which alone fixes the stationary distribution, is unchanged."""
    noise = Noise.from_old(EPS, EPS_UP, EPS_DOWN)
    assert noise.eta_up == pytest.approx(0.0544, abs=5e-5) and noise.eta_down == pytest.approx(0.0567, abs=5e-5)
    norm = ScoringNorm.from_barcode(NORMS[name])
    population = Population([Discriminators(), Defectors()], [0.99, 0.01])
    w = np.random.default_rng(3).dirichlet(np.ones(2500), 2)         # any score distributions: the rates are fixed
    plain, new = TransitionRates(norm, Noise())(population, w), TransitionRates(norm, noise)(population, w)
    g_old, f_old = (1 - EPS) * plain.G + EPS * EPS_UP, (1 - EPS) * plain.F + EPS * EPS_DOWN
    for i in range(2):
        old_pi = chain.stationary(g_old[i, :2500], f_old[i, :2500])
        np.testing.assert_allclose(chain.stationary(new.g[i, :2500], new.f[i, :2500]), old_pi, atol=1e-13)
        assert old_pi[0] > 0


@pytest.mark.parametrize("name", NORMS)
def test_from_old_solver_has_the_old_stationary_distributions(name):
    """The same for the population's fixed point (a faster relaxing case than the talk's, to keep the test short)."""
    norm = ScoringNorm.from_barcode(NORMS[name])
    strategies, rho = [Discriminators(), Defectors()], [0.9, 0.1]
    old = old_stationary(norm, strategies, rho, EPS, 0.2, 0.8)
    new = new_stationary(norm, strategies, rho, Noise.from_old(EPS, 0.2, 0.8))
    n = min(old.shape[1], new.shape[1])
    np.testing.assert_allclose(new[:, :n], old[:, :n], atol=1e-9)


def test_from_old_rates_are_the_old_rates_divided_by_one_minus_eps():
    norm = ScoringNorm.from_barcode(NORMS["Coercive money"])
    population, w = Population([Discriminators(), Defectors()], [0.3, 0.7]), np.random.default_rng(0).dirichlet(np.ones(6), 2)
    plain = TransitionRates(norm, Noise())(population, w)
    new = TransitionRates(norm, Noise.from_old(EPS, EPS_UP, EPS_DOWN))(population, w)
    np.testing.assert_allclose(new.g[:, 1:], ((1 - EPS) * plain.G[:, 1:] + EPS * EPS_UP) / (1 - EPS), atol=1e-15)
    np.testing.assert_allclose(new.f[:, 1:], ((1 - EPS) * plain.F[:, 1:] + EPS * EPS_DOWN) / (1 - EPS), atol=1e-15)
    assert not new.f[:, 0].any()


def coercive_money(eta_up, eta_down, epsilon=0.01):
    norm = ScoringNorm.from_barcode(NORMS["Coercive money"])
    solver = StableScoreSolver(norm, Noise(eta_up, eta_down, epsilon), tol=1e-12, max_iter=100_000, tail_tol=1e-15)
    return solver(Population([Discriminators()]), np.array([[0.5, 0.5]]))


@pytest.mark.parametrize("eta_down, w0, mean", [(0.01, 0.5, 1.005), (0.0067, 0.254, 2.963)])
def test_discriminators_under_coercive_money(eta_down, w0, mean):
    """w_0 = 1 - eta_up / eta_down whatever lambda and epsilon are; w_j = r0 r1^(j-1) w_0."""
    eta_up, epsilon = 0.005, 0.01
    result = coercive_money(eta_up, eta_down, epsilon)
    assert not result.drifting.any()
    w = result.w[0]
    w0_exact = 1 - eta_up / eta_down
    alpha = LAMBDA * (1 - w0_exact) * (1 - epsilon) + eta_up
    beta = LAMBDA * (1 - w0_exact) + eta_up
    gamma = LAMBDA * (1 - epsilon * w0_exact) + eta_down
    r0, r1 = alpha / gamma, beta / gamma
    assert w[0] == pytest.approx(w0_exact, abs=1e-7) and w0_exact == pytest.approx(w0, abs=5e-4)
    j = np.arange(1, len(w))
    np.testing.assert_allclose(w[1:], r0 * r1 ** (j - 1) * w0_exact, atol=1e-8)
    assert (w * np.arange(len(w))).sum() == pytest.approx((1 - w0_exact) / (1 - r1), abs=1e-5)
    assert (w * np.arange(len(w))).sum() == pytest.approx(mean, abs=5e-4)


def test_w0_does_not_depend_on_lambda_or_epsilon():
    for epsilon in (0.0, 0.05):
        for role in (0.1, 0.25):
            norm = ScoringNorm.from_barcode(NORMS["Coercive money"])
            solver = StableScoreSolver(norm, Noise(0.005, 0.01, epsilon), tol=1e-12, max_iter=100_000, role_factor=role)
            assert solver(Population([Discriminators()]), np.array([[0.5, 0.5]])).w[0, 0] == pytest.approx(0.5, abs=1e-6)


def test_without_extra_rates_and_errors_the_results_are_the_old_no_noise_results():
    """Noise() is the old Noise(eps=0): the mixture reduces to g = G, f = F."""
    rng = np.random.default_rng(1)
    strategies = [BinaryStrategy.from_name("0,0|g,g"), BinaryStrategy.from_name("g,0|0,g")]
    norm_text = "CN:00+00-++ NC:00000+--"
    rho, w0 = [0.4, 0.6], rng.dirichlet(np.ones(4), 2)
    result = StableScoreSolver(ScoringNorm.from_barcode(norm_text), Noise(), tol=1e-300, max_iter=25, force_max_iter=True,
                               stop_when_transient=False)(Population(strategies, rho), w0)
    mine = [im.table_help(s.name) for s in strategies]
    expected, *_ = im.evolve(mine, im.parse_norm(norm_text), (0.0, 0.5, 0.5), rho, w0, 25)
    L = min(expected.shape[1], result.w.shape[1])
    np.testing.assert_allclose(result.w[:, :L], expected[:, :L], atol=1e-13)
    assert Noise() == Noise(0.0, 0.0, 0.0) and Noise().act(np.array([0.0, 0.3, 1.0])).tolist() == [0.0, 0.3, 1.0]


def test_rates_may_depend_on_the_score():
    norm = ScoringNorm.from_barcode(NORMS["Coercive money"])
    population, w = Population([Discriminators(), Defectors()], [0.3, 0.7]), np.random.default_rng(2).dirichlet(np.ones(7), 2)
    plain = TransitionRates(norm, Noise())(population, w)
    for eta_down in (lambda j: 0.01 * j, 0.01 * np.arange(7), [0.0, 0.01, 0.02]):
        R = TransitionRates(norm, Noise(eta_up=lambda j: 0.1 / (1 + j), eta_down=eta_down))(population, w)
        j = np.arange(7)
        table = eta_down(j) if callable(eta_down) else np.asarray(eta_down)[np.minimum(j, len(eta_down) - 1)]
        np.testing.assert_allclose(R.g[:, 1:], plain.g[:, 1:] + 0.1 / (1 + j), atol=1e-15)    # g[:, j] leaves score j - 1
        np.testing.assert_allclose(R.f[:, 1:], plain.f[:, 1:] + table[1:], atol=1e-15)
        assert not R.f[:, 0].any()


def test_extra_rates_are_checked():
    for bad in (-0.1, [0.1, -0.1], np.nan):
        with pytest.raises(ValueError):
            Noise(eta_up=bad)
    with pytest.raises(ValueError):
        Noise(epsilon=1.5)
    Noise(eta_up=3.0, eta_down=7.0)                          # rates, not probabilities: they need not sum to 1 or stay below 1
    with pytest.raises(ValueError):
        Noise(eta_down=lambda j: j - 3).down(6)


def test_there_is_no_cap_on_the_score():
    """With eta_up > eta_down the mass keeps moving up: the solver adds a score every step and reports drifting."""
    norm = ScoringNorm.from_barcode(NORMS["Coercive money"])
    result = StableScoreSolver(norm, Noise(0.02, 0.01), max_iter=300, force_max_iter=True)(Population([Discriminators()]),
                                                                                           np.array([[0.5, 0.5]]))
    assert result.w.shape[1] > 100 and result.drifting.all()
    assert not hasattr(StableScoreSolver(norm), "max_score")


def test_the_tail_cutoff_is_only_numerical():
    """Dropping the scores whose mass is below tail_tol changes the distribution by little, and less for a smaller one."""
    norm = ScoringNorm.from_barcode(NORMS["Coercive money"])
    kw = dict(tol=1e-13, max_iter=50_000)
    w0 = np.array([[0.5, 0.5]])
    exact = StableScoreSolver(norm, Noise(0.005, 0.01), **kw)(Population([Discriminators()]), w0).w[0]
    distance = {}
    for tail in (1e-15, 1e-12):
        cut = StableScoreSolver(norm, Noise(0.005, 0.01), tail_tol=tail, **kw)(Population([Discriminators()]), w0).w[0]
        assert abs(cut.sum() - 1) < 1e-14
        L = min(len(cut), len(exact))
        distance[tail] = np.abs(cut[:L] - exact[:L]).max()
    assert distance[1e-15] < 1e-9 and distance[1e-12] < 1e-6 and distance[1e-15] <= distance[1e-12]


def test_action_errors_change_the_help_probabilities_of_rates_and_payoffs():
    strategy, w = PhiDisc(1.0), np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])
    population = Population([strategy, Defectors()], [0.5, 0.5])
    epsilon = 0.2
    assert Noise(epsilon=epsilon).act(np.array([0.0, 0.25, 1.0])).tolist() == pytest.approx([0.2, 0.35, 0.8])
    norm = ScoringNorm.from_barcode(NORMS["Coercive money"])
    noisy = TransitionRates(norm, Noise(epsilon=epsilon))(population, w)
    plain = TransitionRates(norm, Noise())(population, w)
    assert not np.allclose(noisy.G, plain.G)
    H, H_noisy = Payoffs().help_probabilities(population, w), Payoffs(epsilon=epsilon).help_probabilities(population, w)
    np.testing.assert_allclose(H_noisy, (1 - epsilon) * H + epsilon * (1 - H), atol=1e-15)     # the weights sum to 1
    assert Payoffs(epsilon=0.0)(population, w) == pytest.approx(Payoffs()(population, w))
