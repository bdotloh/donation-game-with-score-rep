"""scripts/run_markov_chain.py: the saved rates drive the saved score histories."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_markov_chain.py"
spec = importlib.util.spec_from_file_location("run_markov_chain", SCRIPT)
rmc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rmc)


def read_csv(path):
    with path.open() as handle:
        rows = list(csv.reader(handle))
    return rows[0], np.array(rows[1:], dtype=float)


def step(w, rates, k):
    """One step with class rates (alpha, beta1, gamma1, beta2, gamma2), classes {0}, {1..k-1}, {>=k}."""
    alpha, beta1, gamma1, beta2, gamma2 = rates
    j = np.arange(len(w))
    up = np.where(j == 0, alpha, np.where(j < k, beta1, beta2))
    down = np.where(j == 0, 0.0, np.where(j < k, gamma1, gamma2))
    new = w * (1 - up - down)
    new[1:] += (w * up)[:-1]
    new[:-1] += (w * down)[1:]
    return new


CASES = {
    "binary": ["--cmeetsn", "0+0-0-++", "--nmeetsc", "000+0+-0", "--strategy-a", "0,0|0,g",
               "--strategy-b", "g,0|g,g"],
    "ternary": ["--n-rep-states", "3", "--rich-threshold", "3", "--cmeetsn", "0+-0--+0-++-0-+0+-",
                "--nmeetsc", "000+0+0-0+-0-+0+0-", "--strategy-a", "0,0,0|0,g,g|0,0,g",
                "--strategy-b", "g,g,g|0,g,0|0,0,0"],
}


@pytest.mark.parametrize("case", CASES)
def test_histories_follow_the_saved_rates(case, tmp_path):
    args = rmc.build_parser().parse_args(
        CASES[case] + ["--rho-a", "0.6", "--initial-score-distribution-a", "0.4,0.3,0.3",
                       "--initial-score-distribution-b", "1", "--maxn", "40", "--results-dir", str(tmp_path)]
    )
    run_dir = rmc.run(args)
    details = json.loads((run_dir / "details.json").read_text())
    k = args.rich_threshold or 1
    assert details["parameters"]["tau"] == (None if case == "binary" else k - 1)
    for suffix in ("A", "B"):
        _, rates = read_csv(run_dir / f"evo_trans_{suffix}.csv")
        header, scores = read_csv(run_dir / f"score_trans_{suffix}.csv")
        rep_names, reps = read_csv(run_dir / f"rep_trans_{suffix}.csv")
        assert header == [f"score_{j}" for j in range(scores.shape[1])]
        assert len(rates) == len(scores) == details["results"]["iterations_completed"] == 40
        w = np.array(details["parameters"][f"initial_score_distribution_{suffix}"])
        for t in range(len(scores)):
            row = rates[t] if case == "ternary" else np.r_[rates[t], rates[t, 1:]]
            w = step(w, row, k)
            np.testing.assert_allclose(scores[t], w, atol=1e-12)
            w = scores[t]
        classes = [scores[:, :1], scores[:, 1:k], scores[:, k:]] if case == "ternary" else [scores[:, :1], scores[:, 1:]]
        np.testing.assert_allclose(reps, np.hstack([c.sum(axis=1, keepdims=True) for c in classes]), atol=1e-12)
        final = details["results"][f"strategy_{suffix}"]["final"]
        np.testing.assert_allclose(list(final["transition_probabilities"].values()), rates[-1])


@pytest.mark.parametrize("extra", [["--n-rep-states", "3", "--rich-threshold", "1"], ["--maxw", "100"]])
def test_rejected_arguments(extra, tmp_path):
    args = rmc.build_parser().parse_args(extra + ["--results-dir", str(tmp_path)])
    with pytest.raises(ValueError):
        rmc.run(args)
