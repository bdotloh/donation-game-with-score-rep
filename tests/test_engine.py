"""Tests of the engine against the closed forms in the notes and its own consistency."""

import math
from itertools import islice

import numpy as np
import pytest

from engine import (
    BinaryStrategy,
    Defectors,
    Discriminators,
    Noise,
    Payoffs,
    PhiDisc,
    Population,
    ScoringNorm,
    StableScoreSolver,
    TernaryStrategy,
    TransitionRates,
    all_norms,
    chain,
)

BARCODES = [
    "CN:00+00-++ NC:00000+--",
    "CN:++++0-++ NC:00000000",
    "CN:0+0+0-0+ NC:0000+-0-",
    "CN:00000-0+ NC:00000000",
]


def bisect(h, lo=0.0, hi=1.0, n=200):
    """Root of h on [lo, hi], h(lo) and h(hi) having opposite signs."""
    for _ in range(n):
        mid = (lo + hi) / 2
        if (h(lo) > 0) == (h(mid) > 0):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def tight_solver(norm, noise):
    return StableScoreSolver(norm, noise, tol=1e-14, max_iter=200_000)


# ---------------------------------------------------------------- norms and strategies


def test_barcode_round_trip():
    norm = ScoringNorm.from_barcode("CmeetsN:00+00-++ NmeetsC:00000+--")
    assert norm.barcode == "CN:00+00-++ NC:00000+--"
    assert ScoringNorm.from_barcode(norm.barcode) == norm


def test_score_zero_cannot_be_lost():
    with pytest.raises(ValueError):
        ScoringNorm.from_barcode("CN:-0000000 NC:00000000")  # Can with score 0
    with pytest.raises(ValueError):
        ScoringNorm.from_barcode("CN:00000000 NC:000-0000")  # Need with score 0


def test_ternary_barcode():
    norm = ScoringNorm.from_barcode("CN:00+00-++ NC:00000+--").to_ternary(tau=3)
    assert len(norm.barcode.split()[0]) == len("CN:") + 18
    assert ScoringNorm.from_barcode(norm.barcode, tau=3) == norm
    with pytest.raises(ValueError):
        ScoringNorm.from_barcode(norm.barcode)


def test_all_norms():
    norms = list(all_norms(use_NC=False))
    assert len(norms) == 2**4 * 3**4
    assert len({n.barcode for n in norms}) == len(norms)
    assert all(n.is_ternary for n in islice(all_norms(tau=2), 10))


def test_strategy_names():
    assert BinaryStrategy.from_name("0,0|0,g").name == "0,0|0,g"
    assert BinaryStrategy.from_name("0,0.25|1,g").name == "0,0.25|g,g"
    assert TernaryStrategy.from_name("0,0,0|0,g,g|0,0.5,g", tau=2).name == "0,0,0|0,g,g|0,0.5,g (tau=2)"
    with pytest.raises(ValueError):
        BinaryStrategy.from_name("0,0|0,1.5")


def test_phi_disc_help():
    help_ = PhiDisc(phi=2).donor_help(5)
    np.testing.assert_allclose(help_[0], 0)
    np.testing.assert_allclose(help_[1], [0, 1, 1 / 4, 1 / 9, 1 / 16])


# ---------------------------------------------------------------- closed forms from the notes


def test_defectors_vs_discriminators():
    """Section II.A: CN:00000-00, where w0 of each strategy is A_i(w0) with S -> infinity."""
    eps, eps_up, eps_down = 0.2, 0.1, 0.3
    rho = np.array([0.3, 0.7])
    norm = ScoringNorm.from_barcode("CN:00000-00 NC:00000000")
    population = Population([Defectors(), Discriminators()], rho)
    result = tight_solver(norm, Noise(eps, eps_up, eps_down))(population, [0.5, 0.5])

    w0_2 = (eps_down - eps_up) / eps_down                         # A_2

    def A_1(w0_1):
        q = rho[0] * (1 - w0_1) + rho[1] * (1 - w0_2)
        d_1 = (1 - eps) / 4 * q + eps * (eps_down - eps_up)
        return d_1 / (d_1 + eps * eps_up)

    w0_1 = bisect(lambda x: A_1(x) - x)
    np.testing.assert_allclose(result.w0, [w0_1, w0_2], atol=1e-9)

    # Every positive score shares the same rates, so the distributions are geometric.
    q = rho[0] * (1 - w0_1) + rho[1] * (1 - w0_2)
    ratios = [eps * eps_up / ((1 - eps) * q / 4 + eps * eps_down), eps_up / eps_down]
    for i, x in enumerate(ratios):
        L = result.w.shape[1]
        expected = result.w0[i] * x ** np.arange(L)
        np.testing.assert_allclose(result.w[i], expected, atol=1e-9)


def test_phi_disc_population_phi_1():
    """Section III.A: a population of 1-Disc, whose stable distribution is given by gamma functions."""
    eps, eps_up, eps_down = 0.5, 0.2, 0.4
    norm = ScoringNorm.from_barcode("CN:00000-0+ NC:00000000")
    population = Population([PhiDisc(phi=1)])
    result = tight_solver(norm, Noise(eps, eps_up, eps_down))(population, [0.5, 0.5])
    L = result.w.shape[1]

    et_up, et_down = eps * eps_up, eps * eps_down

    def gamma_form(pi_0, n_scores):
        B = (1 - eps) * (1 - pi_0) / 4
        a, b, r = B / et_up, et_down / (B + et_down), et_up / (B + et_down)
        pi_1 = pi_0 * et_up / et_down
        j = np.arange(1, n_scores)
        log_ratio = np.array(
            [(k - 1) * math.log(r) + math.log(k) + math.lgamma(k + a) + math.lgamma(1 + b)
             - math.lgamma(1 + a) - math.lgamma(k + b) for k in j]
        )
        return np.concatenate(([pi_0], pi_1 * np.exp(log_ratio)))

    # pi_0 satisfies a self-consistency equation: the distribution must sum to 1.
    pi_0 = bisect(lambda p: gamma_form(p, 2000).sum() - 1, 1e-9, 1 - 1e-9)
    expected = gamma_form(pi_0, L)
    np.testing.assert_allclose(result.w[0], expected, atol=1e-9)


def test_phi_disc_phi_0_is_discriminators():
    for barcode in BARCODES:
        solver = StableScoreSolver(ScoringNorm.from_barcode(barcode), Noise(0.05, 0.3, 0.6), max_iter=500)
        phi = solver(Population([PhiDisc(0), Defectors()], [0.6, 0.4]), [0.3, 0.4, 0.3])
        disc = solver(Population([Discriminators(), Defectors()], [0.6, 0.4]), [0.3, 0.4, 0.3])
        np.testing.assert_allclose(phi.w, disc.w, atol=1e-14)


# ---------------------------------------------------------------- consistency


@pytest.mark.parametrize("barcode", BARCODES)
@pytest.mark.parametrize("tau", [1, 2, 5])
def test_ternary_that_ignores_tau_matches_binary(barcode, tau):
    norm = ScoringNorm.from_barcode(barcode)
    binary = [BinaryStrategy.from_name("0,g|0,g"), BinaryStrategy.from_name("0,0|0.3,g")]
    ternary = [TernaryStrategy.from_binary(s, tau) for s in binary]
    noise = Noise(0.01, 0.4, 0.6)
    w_init = [0.2, 0.3, 0.5]

    expected = StableScoreSolver(norm, noise, max_iter=300)(Population(binary, [0.4, 0.6]), w_init)
    for strategies in (binary, ternary):
        for n in (norm, norm.to_ternary(tau)):
            population = Population(strategies, [0.4, 0.6])
            result = StableScoreSolver(n, noise, max_iter=300)(population, w_init)
            np.testing.assert_allclose(result.w, expected.w, atol=1e-13)
            np.testing.assert_allclose(Payoffs()(population, result.w), Payoffs()(population, expected.w), atol=1e-13)


def test_ternary_strategy_uses_tau():
    """Helping only receivers above tau differs from helping every receiver with a score."""
    norm = ScoringNorm.from_barcode("CN:00000-0+ NC:00000000").to_ternary(tau=2)
    solver = StableScoreSolver(norm, Noise(0.1, 0.3, 0.5), max_iter=2000)
    rich_only = TernaryStrategy.from_name("0,0,0|0,0,0|0,g,g", tau=2)
    everyone = TernaryStrategy.from_name("0,0,0|0,g,g|0,g,g", tau=2)
    w_rich = solver(Population([rich_only]), [0.5, 0.5]).w
    w_all = solver(Population([everyone]), [0.5, 0.5]).w
    assert not np.allclose(w_rich[0, :5], w_all[0, :5])


@pytest.mark.parametrize("barcode", BARCODES)
def test_rates_and_payoffs_are_linear_in_help_probabilities(barcode):
    """A decision point with probability p behaves as p * (gives) + (1 - p) * (declines)."""
    rates = TransitionRates(ScoringNorm.from_barcode(barcode), Noise(0.02, 0.5, 0.5))
    rng = np.random.default_rng(0)
    w = rng.random((2, 6))
    w /= w.sum(axis=1, keepdims=True)
    other = TernaryStrategy.from_name("0,g,0|0.2,g,0|g,0,0.7", tau=2)

    def evaluate(p):
        population = Population([BinaryStrategy([[0, 0.4], [p, 1]]), other], [0.35, 0.65])
        r = rates(population, w)
        return np.concatenate([r.g.ravel(), r.f.ravel(), Payoffs()(population, w)])

    p = 0.3
    np.testing.assert_allclose(evaluate(p), p * evaluate(1) + (1 - p) * evaluate(0), atol=1e-14)


def test_stable_distribution_satisfies_detailed_balance():
    norm = ScoringNorm.from_barcode("CN:0+0+0-0+ NC:0000+-0-")
    population = Population([PhiDisc(1.5), TernaryStrategy.from_name("0,0,0|0,g,0.5|g,g,g", tau=3)], [0.5, 0.5])
    result = tight_solver(norm, Noise(0.6, 0.1, 0.6))(population, [0.5, 0.5])
    assert result.converged
    for i in range(population.n):
        pi = chain.stationary(result.rates.g[i, :-1], result.rates.f[i])
        np.testing.assert_allclose(result.w[i], pi, atol=1e-10)


def test_probability_is_conserved():
    norm = ScoringNorm.from_barcode("CN:++++0-++ NC:00000+--")
    population = Population([PhiDisc(0.5), BinaryStrategy.from_name("0,g|0.5,g"), Defectors()], [0.2, 0.3, 0.5])
    result = StableScoreSolver(norm, Noise(0.01), max_iter=50, record_history=True)(population, [1.0])
    for w, _ in result.history:
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-13)
        assert np.all(w >= 0)


def test_drifting_chains_are_not_converged():
    """Money with eps_up = eps_down: Discriminators' scores spread out for ever (null recurrent)."""
    money = ScoringNorm.from_barcode("CN:00000-++ NC:00000+--")
    result = StableScoreSolver(money, Noise.from_noise_val(0.1), max_iter=20_000)(Population([Discriminators()]),
                                                                                  [0.5, 0.5])
    assert result.drifting.all() and not result.converged
    settled = StableScoreSolver(money, Noise(0.1, 0.2, 0.4), tol=1e-9, max_iter=20_000)(
        Population([Discriminators()]), [0.5, 0.5])
    assert settled.converged and not settled.drifting.any()
    np.testing.assert_allclose(settled.w0, [1 - 0.2 / 0.4], atol=1e-4)   # w0 = 1 - eps_up/eps_down for money


def test_concurrent_rate_evaluations():
    """Rates and payoffs can be computed from several threads sharing one solver."""
    import threading

    norm = ScoringNorm.from_barcode("CN:00000-0+ NC:00000+-0")
    solver = StableScoreSolver(norm, Noise(0.1, 0.3, 0.7))
    errors = []

    def work(seed):
        rng = np.random.default_rng(seed)
        for _ in range(40):
            population = Population([PhiDisc(rng.uniform(0, 2)), Discriminators()], [0.5, 0.5])
            w = rng.random((2, int(rng.integers(60, 300))))
            w /= w.sum(axis=1, keepdims=True)
            try:
                solver.rates(population, w)
                Payoffs().help_probabilities(population, w)
            except Exception as error:  # noqa: BLE001
                errors.append(error)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
