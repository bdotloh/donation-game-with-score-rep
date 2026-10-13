"""The ESS figure and its readers."""

import importlib.util
import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from engine import LEADING_EIGHT, MONEY, BinaryStrategy, ScoringNorm, TernaryStrategy, strategy_from_name  # noqa: E402
from plots import EssRow, ess_figure, leading_label, read_paper_data, read_sweep  # noqa: E402
from plots.ess_plot import cell_weights, is_conservative, prepare, strategy_label  # noqa: E402


def row(cn, strategy, cooperation, nc="00000000", tau=None):
    return EssRow(ScoringNorm.from_barcode(f"CN:{cn} NC:{nc}", tau=tau), strategy_from_name(strategy, tau=tau), cooperation)


def leading_rows():
    return [row(n.CN, n.strategy, 0.99) for n in LEADING_EIGHT]


def test_cell_weights_follow_the_strategy():
    weights = cell_weights(BinaryStrategy.from_name("g,0|0,0.25"), 2)
    # [recipient class][act][donor class]; act 0 = declines, 1 = gives
    np.testing.assert_allclose(weights[0, 1], [1, 0])
    np.testing.assert_allclose(weights[0, 0], [0, 1])
    np.testing.assert_allclose(weights[1, 1], [0, 0.25])
    np.testing.assert_allclose(weights[1, 0], [1, 0.75])
    with pytest.raises(ValueError):
        cell_weights(BinaryStrategy.from_name("0,0|0,0"), 3)


def test_strategy_labels():
    assert strategy_label(strategy_from_name("0,0|0,g")) == "AllD-Disc"
    assert strategy_label(strategy_from_name("g,0|g,g")) == "AllC-Disc"
    assert strategy_label(strategy_from_name("0,g|g,0")) == "Disc-Anti"
    assert strategy_label(strategy_from_name("0,0|0.5,g")) == "0,0|0.5,g"
    assert strategy_label(strategy_from_name("0,0,0|0,g,0|0,g,0", tau=5)) == "0,0,0|0,g,0|0,g,0"
    assert strategy_label(strategy_from_name("phi:0.5")) == "phi-Disc(phi=0.5)"


def test_leading_eight_are_recognised():
    for named in LEADING_EIGHT:
        norm = ScoringNorm.from_barcode(named.barcode)
        assert leading_label(norm, strategy_from_name(named.strategy)) == named.label
        other = "0,0|0,g"
        assert leading_label(norm, strategy_from_name(other)) is None                    # wrong ESS strategy
        lifted = named.norm(tau=3)
        assert leading_label(lifted, TernaryStrategy.from_binary(strategy_from_name(named.strategy), 3)) == named.label
        assert leading_label(lifted, TernaryStrategy.from_binary(strategy_from_name(other), 3)) is None
    money = ScoringNorm.from_barcode(MONEY.barcode)
    assert leading_label(money, strategy_from_name(MONEY.strategy)) == "$"


def test_a_norm_with_extra_changes_that_cannot_matter_still_counts_as_a_leading_eight_norm():
    # '+' for a donor who has a score cannot happen in the two-reputation model: the paper's code ignores it
    stern = next(n for n in LEADING_EIGHT if n.label == "L6")
    tokens = stern.CN[:-1] + "+"
    norm = ScoringNorm.from_barcode(f"CN:{tokens} NC:00000000")
    assert leading_label(norm, strategy_from_name(stern.strategy)) == "L6"
    # but a change of the recipient's score does matter
    norm = ScoringNorm.from_barcode(f"CN:{stern.CN} NC:000000-0")
    assert leading_label(norm, strategy_from_name(stern.strategy)) is None


def test_conservative_norms():
    assert is_conservative(ScoringNorm.from_barcode(MONEY.barcode))
    assert is_conservative(ScoringNorm.from_barcode("CN:00000-++ NC:00000+--"))
    assert is_conservative(ScoringNorm.from_barcode("CN:00000000 NC:00000000"))
    assert not is_conservative(ScoringNorm.from_barcode(f"CN:{LEADING_EIGHT[0].CN} NC:00000000"))


def test_rows_are_sorted_by_cooperation_then_strategy():
    rows = [row("00000-+0", "0,0|g,g", 0.9), row("00000000", "0,0|0,0", 0.0), row("00000000", "0,0|0,g", 0.0)]
    data = prepare(rows)
    assert [r.strategy.name for r in data.rows] == ["0,0|0,0", "0,0|0,g", "0,0|g,g"]
    assert data.cn.shape == (3, 8) and data.weights.shape == (3, 8)
    with pytest.raises(ValueError):
        prepare([])
    with pytest.raises(ValueError):
        prepare([row("00000000", "0,0|0,0", 0.0), row("0" * 18, "0,0,0|0,0,0|0,0,0", 0.0, nc="0" * 18, tau=2)])


def ess_axes(fig):
    return next(ax for ax in fig.axes if ax.get_xlabel() == "Cooperation level")


def texts(ax):
    return [t.get_text() for t in ax.texts]


def test_binary_figure():
    rows = leading_rows() + [row("00000000", "0,0|0,g", 0.25), row("00000-00", "0,0|0,0", 0.0)]
    fig = ess_figure(rows, show_cons=True, title="test")
    ess_ax = ess_axes(fig)
    assert len(ess_ax.patches) == 10                                     # one bar per row
    assert sorted(t for t in texts(ess_ax) if t.startswith("L")) == [f"L{i}" for i in range(1, 9)]
    all_text = [t for ax in fig.axes for t in texts(ax)]
    assert "AllC-Disc" in all_text and "Disc-Disc" in all_text and "″" in all_text
    assert "Cons" in all_text and "✓" in all_text                   # the all-zero norm is conservative
    assert "Recvr" in all_text and "Action" in all_text and "Donor" in all_text
    fig.canvas.draw()


def test_leading_bars_are_black_and_the_others_grey():
    fig = ess_figure(leading_rows() + [row("00000000", "0,0|0,g", 0.25)])
    colours = [p.get_facecolor()[:3] for p in ess_axes(fig).patches]
    assert colours.count((0.0, 0.0, 0.0)) == 8 and colours.count((0.4, 0.4, 0.4)) == 1


def test_ternary_figure():
    rows = [row("0" * 18, "0,0,0|0,0,0|0,0,0", 0.0, nc="0" * 18, tau=3),
            row("0" * 8 + "+0" + "0" * 8, "0,0,0|0,g,0|0,g,g", 0.6, nc="0" * 18, tau=3)]
    fig = ess_figure(rows)
    all_text = [t for ax in fig.axes for t in texts(ax)]
    assert "1–3" in all_text and ">3" in all_text                  # class labels of the header
    assert "0,0,0|0,g,0|0,g,g" in all_text
    fig.canvas.draw()


def test_large_figures_are_drawn_as_images(monkeypatch):
    import plots.ess_plot as f

    rng = np.random.default_rng(0)
    codes = ["".join(rng.choice(list("0+"), 8)) for _ in range(300)]
    rows = [row(c, "0,0|g,g", float(rng.uniform())) for c in codes]
    monkeypatch.setattr(f, "VECTOR_CELLS", 100)
    fig = f.ess_figure(rows)
    assert any(len(ax.images) for ax in fig.axes[:1])                    # the norm is an image
    assert len(ess_axes(fig).images) == 1                                 # so are the bars (more than 200 rows)
    assert "300 rows in all" in texts(ess_axes(fig))
    fig.canvas.draw()


def test_image_and_vector_drawing_agree(monkeypatch):
    import plots.ess_plot as f

    rows = [row("+0+-0-+0", "g,0|g,g", 0.9), row("00000-00", "0,0|0,0.5", 0.1)]
    fig_vector = f.ess_figure(rows)
    monkeypatch.setattr(f, "VECTOR_CELLS", 1)
    fig_image = f.ess_figure(rows)
    for fig in (fig_vector, fig_image):
        fig.canvas.draw()
    # compare the norm columns pixel by pixel: same colours in the same places (up to the resampling)
    def grab(fig):
        ax = fig.axes[0]
        box = ax.get_window_extent().bounds
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)
        h = image.shape[0]
        x0, y0, w, hh = (int(round(v)) for v in box)
        return image[h - y0 - hh + 3:h - y0 - 3, x0 + 3:x0 + w - 3]
    a, b = grab(fig_vector), grab(fig_image)
    assert a.shape == b.shape
    assert np.mean(np.abs(a - b).sum(axis=2) > 60) < 0.05                # edges only


def write_sweep(directory, rows, config=None):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config or {"tau": None, "benefit": 2.0, "cost": 1.0}))
    lines = ["norm,ess,fitness,w0,w1,mean_score,alpha,beta,gamma,beta_top,gamma_top,converged"]
    lines += [f'{norm},"{ess}",{fit},0,0,0,0,0,0,,,1' for norm, ess, fit in rows]
    (directory / "ess.csv").write_text("\n".join(lines) + "\n")


def test_read_sweep_binary(tmp_path):
    write_sweep(tmp_path, [("CN:00+00-+0 NC:00000000", "g,0|g,g", 0.99), ("CN:00000000 NC:00000000", "0,0|0,0", 0.0)])
    rows = read_sweep(tmp_path)
    assert [r.strategy.name for r in rows] == ["g,0|g,g", "0,0|0,0"]
    assert rows[0].cooperation == pytest.approx(0.99) and not rows[0].norm.is_ternary
    assert read_sweep(tmp_path / "ess.csv")[0].cooperation == pytest.approx(0.99)


def test_read_sweep_scales_fitness_by_benefit_minus_cost(tmp_path):
    write_sweep(tmp_path, [("CN:00000000 NC:00000000", "0,0|g,g", 3.0)], {"tau": None, "benefit": 4.0, "cost": 1.0})
    assert read_sweep(tmp_path)[0].cooperation == pytest.approx(1.0)


def test_read_sweep_ternary(tmp_path):
    write_sweep(tmp_path, [(f"CN:{'0' * 18} NC:{'0' * 18}", "0,0,0|0,g,0|0,g,g (tau=4)", 0.5)],
                {"tau": 4, "benefit": 2.0, "cost": 1.0})
    (r,) = read_sweep(tmp_path)
    assert r.norm.is_ternary and r.norm.partition.tau == 4 and isinstance(r.strategy, TernaryStrategy)


def test_read_sweep_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_sweep(tmp_path)


PAPER = """# Reputation type: binary, noise value: 0.010, max iterations: 500
# BENEFIT 2, COST 1 # WSTART: [0.5, 0.0, 0.5],
#
# 0,0|0,0 being considered for ESS.
# BBname ...
+0+-0-00 00000000 0000 0,0|0,g 0.768 0.123 0.0 0.5 0.1 0.1 0.2 0.0
00+-0-+0 00000000 0000 g,0|g,g 0.9875 0.01 0.0 0.9 0.1 0.1 0.2 0.0
"""


def test_read_paper_data(tmp_path):
    path = tmp_path / "binary_CmeetsN_Noise0.01_B2"
    path.write_text(PAPER)
    rows = read_paper_data(path)
    assert [leading_label(r.norm, r.strategy) for r in rows] == [None, "L2"]
    assert rows[0].cooperation == pytest.approx(0.768)
    (tmp_path / "bad").write_text("# nothing\n00000000 00000000 0,0|0,0 0.5\n")
    with pytest.raises(ValueError):
        read_paper_data(tmp_path / "bad")


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "plot_ess.py"
    spec = importlib.util.spec_from_file_location("plot_ess", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_line(tmp_path):
    plot_ess = load_script()

    write_sweep(tmp_path / "sweep", [(f"CN:{n.CN} NC:{n.NC}", n.strategy, 0.98) for n in LEADING_EIGHT]
                + [("CN:00000000 NC:00000000", "0,0|0,0", 0.0)])
    plot_ess.main([str(tmp_path / "sweep"), "--out", str(tmp_path / "fig"), "--dpi", "40", "--threshold", "0.01"])
    assert (tmp_path / "fig.png").stat().st_size > 1000 and (tmp_path / "fig.pdf").stat().st_size > 1000
    with pytest.raises(SystemExit):
        plot_ess.main([str(tmp_path / "sweep"), "--threshold", "2"])
