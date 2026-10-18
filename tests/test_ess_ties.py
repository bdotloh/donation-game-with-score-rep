"""ESSFinder's tie_tol, with the case that made it necessary, and docs/verification/compare_published.py."""

import sys
from pathlib import Path

import pytest

from engine import (ESSFinder, Noise, Payoffs, ScoringNorm, StableScoreSolver, all_binary_strategies,
                    strategy_from_name)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "verification"))
from compare_published import compare  # noqa: E402


def finder(tie_tol):
    # A norm in which everyone ends up helping: strategies that differ only in how they treat a receiver
    # without a score earn the same payoff, up to rounding (~1e-16).
    norm = ScoringNorm.from_barcode("CN:+++-0-++ NC:00000000")
    solver = StableScoreSolver(norm, Noise(0.01), tol=1e-5, max_iter=500)
    return ESSFinder(solver, Payoffs(), rho=0.99, fitness_threshold=0.01, final_max_iter=500, tie_tol=tie_tol)


def test_payoff_ties_are_decided_by_rounding_unless_a_tolerance_is_given():
    A, B = strategy_from_name("0,0|g,g"), strategy_from_name("0,g|g,g")
    exact = finder(0.0)
    fit_A, fit_B = exact.payoffs_against(A, B, 0.99, [0.5, 0.5])
    assert fit_A != fit_B and abs(fit_A - fit_B) < 1e-14          # the same payoff, apart from rounding
    strategies = all_binary_strategies()
    assert not exact.is_ess(A, strategies, [0.5, 0.5])             # rounding makes B "invade"
    assert finder(1e-12).is_ess(A, strategies, [0.5, 0.5])         # a tie, so A stands (as in the paper's code)
    assert finder(1e-12).is_ess(strategy_from_name("g,0|g,g"), strategies, [0.5, 0.5])


def test_a_tolerance_does_not_turn_real_invaders_into_ties():
    norm = ScoringNorm.from_barcode("CN:00000-00 NC:00000000")
    solver = StableScoreSolver(norm, Noise(0.01), tol=1e-5, max_iter=100)
    strategies = all_binary_strategies()
    for tol in (0.0, 1e-12):
        found = ESSFinder(solver, Payoffs(), rho=0.99, tie_tol=tol)(strategies, strategies, [0.5, 0.5])
        assert [s.name for s in found] == ["0,0|0,g"]


def test_comparison_with_a_published_table():
    from plots.ess_data import EssRow

    def row(barcode, strategy, cooperation):
        return EssRow(ScoringNorm.from_barcode(barcode), strategy_from_name(strategy), cooperation)

    a, b, c = "CN:00000-00 NC:00000000", "CN:+0000-00 NC:00000000", "CN:0+000-00 NC:00000000"
    engine = [row(a, "0,0|0,g", 0.5), row(b, "0,0|0,g", 0.7), row(c, "0,0|0,g", 0.2)]
    paper = [row(a, "0,0|0,g", 0.5004), row(b, "0,0|0,g", 0.7), row(a, "g,0|g,g", 0.9)]
    result = compare(engine, paper, threshold=0.25)
    assert (result["published"], result["engine"], result["common"]) == (3, 3, 2)
    assert result["missing"] == [(ScoringNorm.from_barcode(a).barcode, "g,0|g,g")]
    assert result["extra"] == [(ScoringNorm.from_barcode(c).barcode, "0,0|0,g")] and result["extra_above_threshold"] == []
    assert result["max_difference"] == pytest.approx(4e-4) and result["published_norms"] == 2
