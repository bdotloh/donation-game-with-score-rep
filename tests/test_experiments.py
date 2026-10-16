"""The scripts in experiments/ reproduce their committed results (made with the old src/ code)."""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
ZERONESS = EXPERIMENTS / "verify-zeroness-one-strategy"


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def spread(rows, n):
    return [rows[k] for k in np.linspace(0, len(rows) - 1, n, dtype=int)]


def test_ess_sweep_matches_committed_csv():
    ess = load(EXPERIMENTS / "ESS" / "find_ESS_parellel.py")
    committed = read(EXPERIMENTS / "ESS" / "Results" / "tokens_CmeetsN_Noise0_B2.csv")
    norms = sorted(ess.bigb_name(norm) for norm in ess.all_norms(use_NC=False))
    with_ess = sorted({row["bb_name"] for row in committed})
    without_ess = sorted(set(norms) - set(with_ess))
    strategies = [f"{a},{b}|{c},{d}" for a in "0g" for b in "0g" for c in "0g" for d in "0g"]
    for name in spread(with_ess, 6) + spread(without_ess, 4):
        job = (name, strategies, strategies, np.array([0.5, 0.5]), 0.99, 100, 2, 1, 2, None, 0.01, ess.Noise())
        result = ess.run_one_bigb(job)
        expected = [row for row in committed if row["bb_name"] == name]
        assert result["ess"] == [row["ess_name"] for row in expected]
        for detail, row in zip(result["ess_details"], expected):
            got = [detail["fitness"], detail["w"][0], detail["w"][1], *detail["trans"]]
            want = [float(row[k]) for k in ("fitness", "w0", "w1", "alpha", "beta", "gamma")]
            # The committed file was made when a single strategy's payoff and rates came from the distribution
            # before the last step; they now come from the final one, which differs by that step's change
            # (at most the score tolerance, 1e-5 here). The ESS themselves are the same.
            np.testing.assert_allclose(got, want, atol=1e-4)


def test_grouping_matches_committed_combinations():
    group = load(ZERONESS / "group_chains_by_initial_transitions.py")
    rows = read(ZERONESS / "results" / "combinations.csv")
    numbers = ["initial_alpha", "initial_beta", "initial_gamma", "final_alpha", "final_beta", "final_gamma",
               "final_score_0"]
    for row in spread(rows, 60):
        result = group.run_combo((row["bb_string"], row["strategy"], np.array([0.5, 0.5]), 500, 1e-5))
        assert result["group"] == row["group"]
        np.testing.assert_allclose([result[k] for k in numbers], [float(row[k]) for k in numbers], atol=1e-12)
        if int(row["final_distribution_length"]) <= 100:       # the old cap at maxw = 100 didn't bind
            assert result["final_distribution_length"] == int(row["final_distribution_length"])
        else:
            assert result["final_distribution_length"] > 100


def test_maxw_is_rejected(monkeypatch, tmp_path):
    group = load(ZERONESS / "group_chains_by_initial_transitions.py")
    monkeypatch.setattr("sys.argv", ["group", "--maxw", "100", "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="maxw"):
        group.main()


def test_limit_checks_match_committed_results():
    limits = load(ZERONESS / "verify_transition_group_limits.py")
    committed = read(ZERONESS / "results" / "limit_checks" / "combination_limit_checks.csv")
    combos = {(row["bb_string"], row["strategy"]): row for row in read(ZERONESS / "results" / "combinations.csv")}
    # One quickly converging chain per condition.
    picked = {}
    for row in committed:
        if row["converged"] == "1" and int(row["steps"]) < 3000:
            picked.setdefault(row["condition"], row)
    assert len(picked) >= 6
    for condition, expected in picked.items():
        combo = combos[(expected["bb_string"], expected["strategy"])]
        result = limits.verify_combo((combo, 100, 100_000, 1e-12, 1e-6))
        assert result["condition"] == condition
        assert result["steps"] == int(expected["steps"])
        assert result["claim_passed"] == int(expected["claim_passed"])
        for key in ("final_p0", "final_p1", "final_pmax", "final_mean"):
            assert result[key] == pytest.approx(float(expected[key]), abs=1e-10)
