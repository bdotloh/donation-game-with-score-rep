"""The engine reproduces helper_funcs.jl (reference outputs from tests/reference/make_reference.jl)."""

import csv
import random
from pathlib import Path

import numpy as np
import pytest

from engine import BinaryStrategy, ESSFinder, Noise, Payoffs, Population, ScoringNorm, StableScoreSolver

REFERENCE = Path(__file__).parent / "reference"
STRATEGIES = [f"{a},{b}|{c},{d}" for a in "0g" for b in "0g" for c in "0g" for d in "0g"]


def read_tsv(name):
    with open(REFERENCE / name, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def floats(text):
    return np.array([float(x) for x in text.split()])


def padded(w, L):
    return np.pad(w, (0, L - len(w)))


def julia_trans(result, i):
    """alpha, beta, gamma of helper_funcs.jl: g_1, g_j (j >= 2) and f_j (j >= 1)."""
    if result.rates is None:
        return np.zeros(3)
    return np.array([result.rates.g[i, 1], result.rates.g[i, 2], result.rates.f[i, 1]])


STABLE_CASES = read_tsv("stable_cases.tsv")


@pytest.mark.parametrize("case", STABLE_CASES, ids=[str(i) for i in range(len(STABLE_CASES))])
def test_stable_scores(case):
    norm = ScoringNorm.from_barcode(case["barcode"])
    solver = StableScoreSolver(norm, Noise.from_noise_val(float(case["noise"])), max_iter=int(case["maxn"]))
    payoffs = Payoffs()

    if case["kind"] == "one":
        population = Population([BinaryStrategy.from_name(case["strategy_a"])])
        result = solver(population, floats(case["w_init_a"]))
        fit = payoffs(population, result.w_prev)
        expected_w = [floats(case["w_a"])]
        expected_fit = [float(case["fit_a"])]
        expected_trans = [floats(case["trans_a"])]
    else:
        strategies = [BinaryStrategy.from_name(case["strategy_a"]), BinaryStrategy.from_name(case["strategy_b"])]
        rho_a = float(case["rho_a"])
        population = Population(strategies, [rho_a, 1 - rho_a])
        result = solver(population, [floats(case["w_init_a"]), floats(case["w_init_b"])])
        fit = payoffs(population, result.w)
        expected_w = [floats(case["w_a"]), floats(case["w_b"])]
        expected_fit = [float(case["fit_a"]), float(case["fit_b"])]
        expected_trans = [floats(case["trans_a"]), floats(case["trans_b"])]

    for i in range(population.n):
        L = max(len(expected_w[i]), result.w.shape[1])
        np.testing.assert_allclose(padded(result.w[i], L), padded(expected_w[i], L), rtol=0, atol=1e-12)
        np.testing.assert_allclose(fit[i], expected_fit[i], rtol=0, atol=1e-12)
        np.testing.assert_allclose(julia_trans(result, i), expected_trans[i], rtol=0, atol=1e-12)


ESS_ROWS = read_tsv("ess_sweep.tsv")
ESS_SAMPLE = random.Random(0).sample(ESS_ROWS, 60)


def find_ess(barcode, noise):
    solver = StableScoreSolver(ScoringNorm.from_barcode(barcode), Noise.from_noise_val(noise), max_iter=100)
    finder = ESSFinder(solver, Payoffs(), rho=0.99, fitness_threshold=1e-5)
    strategies = [BinaryStrategy.from_name(s) for s in STRATEGIES]
    return [s.name for s in finder(strategies, strategies, [0.5, 0.5])]


@pytest.mark.parametrize("row", ESS_SAMPLE, ids=[f"{r['barcode']} noise={r['noise']}" for r in ESS_SAMPLE])
def test_ess(row):
    expected = row["ess"].split(";") if row["ess"] else []
    assert find_ess(row["barcode"], float(row["noise"])) == expected
