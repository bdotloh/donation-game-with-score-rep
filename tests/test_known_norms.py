"""The Leading Eight: structure, and that each has its cooperative ESS in this engine."""

from itertools import product

import pytest

from engine import (LEADING_EIGHT, MONEY, BinaryStrategy, ESSFinder, Noise, Payoffs, Population, ScoringNorm,
                    StableScoreSolver, all_binary_strategies)

# Positions in a barcode: (recipient class, act, donor class), the donor's class fastest.
# Entries every Leading Eight norm shares (donor good = score >= 1, recipient good likewise):
FIXED = {
    7: "0",   # good donor helps a good recipient: stays good
    5: "-",   # good donor declines a good recipient: becomes bad
    1: "0",   # good donor declines a bad recipient: stays good (justified)
    6: "+",   # bad donor helps a good recipient: becomes good
    4: "0",   # bad donor declines a good recipient: stays bad
}
FREE = (3, 2, 0)    # good donor helps a bad recipient; bad donor helps a bad recipient; bad donor declines one


def test_they_are_eight_distinct_valid_norms():
    assert [n.label for n in LEADING_EIGHT] == [f"L{i}" for i in range(1, 9)]
    assert len({n.CN for n in LEADING_EIGHT}) == 8
    for named in LEADING_EIGHT:
        norm = ScoringNorm.from_barcode(named.barcode)
        assert norm.barcode == named.barcode
        assert set(named.NC) == {"0"}                       # only the donor is assessed
    ScoringNorm.from_barcode(MONEY.barcode)


def test_five_entries_are_shared_and_the_other_three_take_every_combination():
    for named in LEADING_EIGHT:
        for position, symbol in FIXED.items():
            assert named.CN[position] == symbol, (named.label, position)
    free = {tuple(named.CN[p] for p in FREE) for named in LEADING_EIGHT}
    assert free == set(product("0-", "0+", "0+"))


def test_named_ones():
    names = {n.label: n.name for n in LEADING_EIGHT}
    assert names["L3"] == "Simple Standing" and names["L6"] == "Stern Judging"
    simple = next(n for n in LEADING_EIGHT if n.label == "L3")
    assert [simple.CN[p] for p in FREE] == ["0", "+", "+"]  # the only two bad outcomes: defecting against the good
    stern = next(n for n in LEADING_EIGHT if n.label == "L6")
    assert [stern.CN[p] for p in FREE] == ["-", "0", "+"]   # right action = good, whoever the donor was


def test_lifted_to_ternary():
    named = LEADING_EIGHT[5]
    norm = named.norm(tau=3)
    assert norm.is_ternary and norm.partition.tau == 3


@pytest.mark.parametrize("named", LEADING_EIGHT, ids=lambda n: n.label)
def test_the_cooperative_strategy_is_an_ess_with_nearly_full_payoff(named):
    solver = StableScoreSolver(ScoringNorm.from_barcode(named.barcode), Noise.from_noise_val(0.01), max_iter=100)
    strategies = all_binary_strategies()
    ess = [s.name for s in ESSFinder(solver, Payoffs(), rho=0.99, fitness_threshold=1e-5)(strategies, strategies, [0.5, 0.5])]
    assert named.strategy in ess
    resident = Population([BinaryStrategy.from_name(named.strategy)])
    result = solver(resident, [0.5, 0.5])
    assert 4 * Payoffs()(resident, result.w_prev)[0] > 0.95      # b - c = 1 when everyone helps
