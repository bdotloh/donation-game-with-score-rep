"""GUI logic that doesn't need a browser: input checks, settings, figures."""

import numpy as np
import pytest

from engine import BinaryStrategy, PhiDisc, TernaryStrategy
from gui import figures, spec, validation
from gui.pages.parameters import experiment_from_inputs


@pytest.mark.parametrize(
    "values",
    [["33.33", "33.33", "33.33"], ["0.3333", "0.3333", "0.3333"], ["1/3", "1/3", "1/3"], ["50", "50"],
     ["33.33%", "33.33%", "33.34%"]],
)
def test_shares_accept_rounded_fractions_and_percentages(values):
    shares, error = validation.shares(values)
    assert error == ""
    np.testing.assert_allclose(shares.sum(), 1.0)
    np.testing.assert_allclose(shares, np.full(len(values), 1 / len(values)), atol=1e-3)


def test_shares_mixed_types():
    np.testing.assert_allclose(validation.shares([0.25, "0.75"])[0], [0.25, 0.75])


@pytest.mark.parametrize("values", [["0.5", "0.4"], ["30", "30", "30"], ["0.5", "abc"], ["-0.5", "1.5"], ["", "1"]])
def test_shares_reject_bad_input(values):
    shares, error = validation.shares(values)
    assert shares is None and error


def test_score_distribution():
    entries = [{"score": 0, "mass": 0.25}, {"score": 3, "mass": 0.75}]
    w, error = validation.score_distribution(entries)
    assert error == ""
    np.testing.assert_allclose(w, [0.25, 0, 0, 0.75])
    assert validation.score_distribution([{"score": 0, "mass": 0.5}])[1]
    assert validation.score_entry("1.5", "0.2")[1]
    assert validation.score_entry("2", "-1")[1]
    assert validation.score_entry("2", "0.2") == ({"score": 2, "mass": 0.2}, "")


def test_noise():
    assert validation.noise("0.1", "0.3", "0.7")[1] == ""
    assert validation.noise("1.5", "0.5", "0.5")[1]
    assert validation.noise("0.1", "0.3", "0.6")[1]
    assert validation.noise("0.1", "0.333", "0.667")[0]["eps_up"] == pytest.approx(0.333)


def test_presets():
    assert BinaryStrategy(spec.preset_strategy("AllD-Disc")["P"]).name == "0,0|0,g"
    assert BinaryStrategy(spec.preset_strategy("Disc-Disc")["P"]).name == "0,0|g,g"
    assert BinaryStrategy(spec.preset_strategy("Cooperators")["P"]).name == "g,g|g,g"
    assert BinaryStrategy(spec.preset_strategy("Defectors")["P"]).name == "0,0|0,0"
    for barcode in spec.NORM_PRESETS.values():
        spec.norm_from_barcode(barcode, "binary")


def _experiment():
    strategies = [spec.preset_strategy("AllD-Disc"), spec.preset_strategy("Defectors"),
                  {"type": "phi", "name": "phi", "P": None, "tau": None, "phi": 1.0}]
    cn, nc = "00000-00", "00000000"
    return experiment_from_inputs(
        strategies, ["33.33", "33.33", "33.33"], [{"score": 0, "mass": 0.5}, {"score": 1, "mass": 0.5}],
        "binary", None, list(cn), list(nc), "0.1", "0.4", "0.6", "0.5", "2", "1", "0.5", "10", "200", "1e-6", "4")


def test_experiment_round_trip():
    experiment, error = _experiment()
    assert error == ""
    assert spec.check(experiment) == ""
    built = spec.build(experiment)
    np.testing.assert_allclose(built["rho0"], [1 / 3] * 3)
    np.testing.assert_allclose(built["w_init"], [0.5, 0.5])
    assert built["names"] == ["AllD-Disc", "Defectors", "phi"]


def test_experiment_errors():
    strategies = [spec.preset_strategy("AllD-Disc"), spec.preset_strategy("Defectors")]
    args = [["0.5", "0.5"], [{"score": 0, "mass": 1.0}], "binary", None, list("00000-00"), list("00000000"),
            "0.1", "0.4", "0.6", "0.5", "2", "1", "0.5", "10", "200", "1e-6", "4"]
    assert experiment_from_inputs(strategies, *args)[1] == ""
    bad = list(args)
    bad[0] = ["0.5", "0.6"]
    assert "Shares" in experiment_from_inputs(strategies, *bad)[1]
    bad = list(args)
    bad[7] = "0.3"
    assert "Noise" in experiment_from_inputs(strategies, *bad)[1]
    bad = list(args)
    bad[4] = list("-0000000")
    assert "Norm" in experiment_from_inputs(strategies, *bad)[1]
    assert "2 or 3" in experiment_from_inputs(strategies[:1], *args)[1]


@pytest.mark.parametrize("strategy", [BinaryStrategy.from_name("0,0|0,g"), TernaryStrategy.from_name(
    "0,0,0|0,g,0.5|0,g,g", tau=2), PhiDisc(1)])
def test_strategy_surface(strategy):
    values = figures.help_surface_values(strategy, 6)
    assert values.shape == (7, 7)
    assert np.all((values >= 0) & (values <= 1))
    fig = figures.strategy_surface(strategy, 6)
    assert fig.data[0].z.shape[0] == fig.data[0].z.shape[1]


def test_phi_disc_surface_values():
    values = figures.help_surface_values(PhiDisc(1), 4)   # [donor, receiver]
    np.testing.assert_allclose(values[:, 0], 0)          # receiver without a score is never helped
    np.testing.assert_allclose(values[:, 2], [0, 1, 1 / 2, 1 / 3, 1 / 4])


def test_simplex_figures():
    names = ["A", "B", "C"]
    grid = np.array([[1, 0, 0], [0.5, 0.5, 0], [0.2, 0.3, 0.5], [0, 0, 1]])
    velocities = np.array([[0, 0, 0], [0.1, -0.1, 0], [0.01, 0.02, -0.03], [0, 0, 0]])
    rest = [{"rho": [1, 0, 0], "kind": "stable", "eigenvalues": [-0.1, -0.2], "converged": True},
            {"rho": [0, 0, 1], "kind": "saddle", "eigenvalues": [-0.1, 0.2], "converged": False}]
    fig = figures.simplex_figure(names, grid, velocities, [True, True, False, True], rest,
                                 trajectory=grid[:3], current=grid[2])
    assert fig.data[-1].name == "selected"
    fig = figures.simplex_figure(names[:2], np.array([[1, 0], [0.5, 0.5], [0, 1]]),
                                 np.array([[0, 0], [0.1, -0.1], [0, 0]]), [True] * 3, [], current=[0.5, 0.5])
    assert fig.data[-1].name == "selected"
    np.testing.assert_allclose(figures.from_xy(*[c[0] for c in figures.to_xy([0.2, 0.3, 0.5])]), [0.2, 0.3, 0.5])


def test_histogram_and_heatmaps():
    w = np.array([[0.5, 0.3, 0.2, 0], [0.1, 0.2, 0.3, 0.4]])
    fig = figures.score_histogram(["A", "B"], [0.25, 0.75], w, hidden=["B"])
    assert {t.name for t in fig.data if t.showlegend is not False} == {"S1 · A", "population"}
    assert fig.layout.barmode == "overlay"
    figures.matrix_heatmap(np.array([[0.1, -0.2], [0.3, 0.0]]), ["A", "B"], "A", "strategy", "against", diverging=True)


def test_click_composition():
    from gui.pages.results import click_composition

    assert click_composition({"points": [{"customdata": [0.2, 0.8]}]}) == [0.2, 0.8]
    assert click_composition({"points": [{"customdata": {"1": 0.8, "0": 0.2}}]}) == [0.2, 0.8]
    assert click_composition({"points": [{}]}) is None
    assert click_composition({"points": [{"customdata": ["a"]}]}) is None


# ---------------------------------------------------------------- cases from the review


def test_grid_values_are_placed_by_id():
    from gui.pages.strategies import grid_values

    # Dash may send pattern-matching values out of row order.
    order = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    entries = [{"id": {"type": "s-p", "r": r, "c": c}, "value": f"{3 * r + c}"} for r, c in order]
    assert grid_values(entries, 3) == [["0", "1", "2"], ["3", "4", "5"], ["6", "7", "8"]]
    assert grid_values(entries[:4], 3) is None
    assert grid_values(entries[:4], 2) == [["0", "1"], ["3", "4"]]


@pytest.mark.parametrize("value", ["inf", "-inf", "nan", "1e400", float("nan"), float("inf"), True])
def test_non_finite_numbers_are_rejected(value):
    assert validation.parse_number(value) is None
    assert validation.positive_int(value, "Steps")[1]
    assert validation.non_negative(value, "The benefit")[1]
    assert validation.score_entry(value, "0.5")[1]
    assert validation.score_entry("1", value)[1]


def test_initial_scores_are_capped():
    assert validation.score_entry(validation.MAX_INITIAL_SCORE + 1, "1")[1]
    assert validation.score_distribution([{"score": -1, "mass": 1.0}])[1]
    assert validation.score_distribution([{"score": 0, "mass": 0.5}, {"score": -1, "mass": 0.2},
                                          {"score": 3, "mass": 0.3}])[1]
    assert validation.score_distribution([{"score": 10 ** 12, "mass": 1.0}])[1]


@pytest.mark.parametrize("change", [
    lambda e: [e],                                      # a list, e.g. rest_points.json
    lambda e: {k: v for k, v in e.items() if k != "n_steps"},
    lambda e: {k: v for k, v in e.items() if k != "resolution"},
    lambda e: {**e, "n_steps": "abc"},
    lambda e: {**e, "scores": [{"score": -1, "mass": 1.0}]},
    lambda e: {**e, "scores": [{"score": 10 ** 12, "mass": 1.0}]},
    lambda e: {**e, "benefit": float("nan")},
    lambda e: {**e, "strategies": e["strategies"][:1]},
    lambda e: {**e, "shares": [0.5, 0.5]},
    lambda e: {**e, "norm": {"type": "ternary", "tau": 2, "CN": "00000000", "NC": "00000000"}},
])
def test_bad_settings_files_are_rejected(change):
    experiment, error = _experiment()
    assert error == ""
    assert spec.check(change(experiment))


def test_settings_round_trip_keeps_digits():
    from gui.pages.strategies import params_from_experiment

    experiment, _ = _experiment()
    experiment["n_steps"] = 1234567
    experiment["tol"] = 1e-7
    params = params_from_experiment(experiment)
    assert params["advanced"]["n_steps"] == "1234567"
    assert validation.parse_number(params["advanced"]["tol"]) == 1e-7


def test_unusable_stored_norm_falls_back():
    from gui.pages.parameters import usable_norm

    fallback = {"type": "binary", "tau": None, "CN": "0" * 8, "NC": "0" * 8}
    assert usable_norm({"type": "ternary", "tau": 2, "CN": "0" * 8, "NC": "0" * 8}, fallback) is fallback
    assert usable_norm({"type": "ternary", "tau": None, "CN": "0" * 18, "NC": "0" * 18}, fallback)["tau"] == 2


def test_repeated_scores_are_merged_on_load():
    from gui.pages.strategies import merged_scores

    assert merged_scores([{"score": 1, "mass": 0.25}, {"score": 0, "mass": 0.5}, {"score": 1.0, "mass": 0.25}]) == [
        {"score": 0, "mass": 0.5}, {"score": 1, "mass": 0.5}]


def test_flow_lines_follow_the_field():
    """Flow lines of a field pointing at one corner run towards it and stay inside the triangle."""
    grid = np.array([(i, j, 10 - i - j) for i in range(11) for j in range(11 - i)]) / 10
    velocities = np.array([0, 0, 1.0]) - grid        # straight towards the top corner
    xs, ys = figures.flow_lines(grid, velocities)
    assert xs.count(None) >= 5                       # several lines
    points = [(x, y) for x, y in zip(xs, ys) if x is not None]
    rho = np.array([figures.from_xy(x, y) for x, y in points])
    assert rho.min() > -1e-6
    lines, current = [], []
    for x, y in zip(xs, ys):
        if x is None:
            lines.append(current)
            current = []
        else:
            current.append(y)
    assert all(line[-1] >= line[0] for line in lines)   # every line climbs towards the top corner


def test_absent_strategies_are_left_out():
    from gui.pages.results import present

    assert present([0.5, 0.0, 0.5]) == [0, 2]
    assert present([1.0, 0.0, 0.0]) == [0]
    w = np.array([[0.5, 0.5], [0.9, 0.1]])
    fig = figures.score_histogram(["A", "C"], [0.5, 0.5], w, labels=["S1", "S3"],
                                  colors=[figures.COLORS[0], figures.COLORS[2]])
    bars = [t for t in fig.data if t.type == "bar"]
    assert [b.name for b in bars] == ["S1 · A", "S3 · C"]
    assert bars[1].marker.color == figures.COLORS[2]
