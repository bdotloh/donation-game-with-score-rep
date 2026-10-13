"""The ESS figure: for every ESS found, the scoring norm as a strip of coloured cells and the cooperation level.

One row per (norm, ESS strategy), sorted by cooperation level, lowest at the top. On the left the norm is drawn
as in the paper (Frean and Marsland): one cell for each situation (recipient's class, whether the donor helped,
donor's class), green where the donor's (or recipient's) score goes up, blue where it goes down, grey where it
stays. A cell is wide when the ESS strategy takes that action in that situation and narrow when it doesn't (for
a strategy that helps with probability p, the width runs between the two in proportion), so the wide cells are
the ones that decide the score. On the right a bar shows the cooperation level, fitness / (benefit - cost);
the bars of Leading Eight norms with their ESS strategy are black and labelled L1 to L8.

The design follows plotting_funcs.py of https://github.com/bdotloh/origin_money, which is MIT licensed,
Copyright (c) 2026 Marcus Frean and Stephen Marsland (see NOTICE). This is a rewrite for this repository's
norms and strategies: any number of score classes (binary and ternary norms), and large numbers of rows.
"""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

from engine import LEADING_EIGHT, MONEY, BinaryStrategy, ClassStrategy, TernaryStrategy

COLORS = {1: (0.64, 0.90, 0.21), 0: (0.85, 0.85, 0.85), -1: (0.3, 0.6, 1.0)}   # score up, same, down
WIDE, NARROW = 0.8, 1 / 3                  # widths of a cell: action taken / not taken
LEADING_BAR, OTHER_BAR = (0.0, 0.0, 0.0), (0.4, 0.4, 0.4)
LABEL_ROWS = 50                            # up to this many rows: L1.. labels on the bars, lines between rows
STRATEGY_ROWS = 30                         # up to this many rows: the strategy names are written out
VECTOR_CELLS = 120_000                     # more cells than this are drawn as an image
CELL = 0.25                                # inches per cell of the norm
NICKNAMES = {("0", "0"): "AllD", ("0", "g"): "Disc", ("g", "0"): "Anti", ("g", "g"): "AllC"}


def strategy_label(strategy):
    """Binary strategies as in the paper, e.g. 'AllD-Disc': what a donor without a score does, then one with."""
    if isinstance(strategy, BinaryStrategy) and set(np.unique(strategy.P)) <= {0.0, 1.0}:
        a, b, c, d = ("g" if p else "0" for p in strategy.P.reshape(-1))
        return f"{NICKNAMES[(a, c)]}-{NICKNAMES[(b, d)]}"
    return strategy.name.split(" (tau=")[0]


def _symbols(norm_array):
    return norm_array.reshape(-1)


def _code(norm_array):
    return "".join("-0+"[v + 1] for v in _symbols(norm_array))


def is_conservative(norm):
    """Whatever the donor gains the recipient loses and the other way round (scores are only moved around)."""
    return bool(np.all(norm.CN == -norm.NC))


def leading_label(norm, strategy):
    """'L1' to 'L8' for a Leading Eight norm with its ESS strategy, '$' for the money norm, else None.

    A binary norm counts if it is the same once the entries that can't matter are zeroed ('-' for a donor
    without a score, '+' for one with a score: that is what a norm of the two-reputation model looks like
    here) and the recipient's score is never changed. A ternary norm must be a lifted binary one.
    """
    if norm.is_ternary:
        tau = norm.partition.tau
        if not isinstance(strategy, TernaryStrategy):
            return None
        for named in (*LEADING_EIGHT, MONEY):
            lifted = TernaryStrategy.from_binary(BinaryStrategy.from_name(named.strategy), tau)
            if norm == named.norm(tau) and np.array_equal(strategy.P, lifted.P):
                return named.label
        return None
    cn, nc = _code(norm.CN), _code(norm.NC)
    if (cn, nc, strategy.name) == (MONEY.CN, MONEY.NC, MONEY.strategy):
        return MONEY.label
    if set(nc) != {"0"}:
        return None
    reduced = "".join("0" if (i % 2 == 0 and ch == "-") or (i % 2 == 1 and ch == "+") else ch
                      for i, ch in enumerate(cn))
    for named in LEADING_EIGHT:
        if reduced == named.CN and strategy.name == named.strategy:
            return named.label
    return None


def cell_weights(strategy, n_classes):
    """Array (n, 2, n) of how often each situation (recipient class, act, donor class) is acted out by the strategy."""
    weights = np.ones((n_classes, 2, n_classes))
    if isinstance(strategy, ClassStrategy):
        if strategy.P.shape != (n_classes, n_classes):
            raise ValueError(f"The strategy {strategy.name!r} has {strategy.P.shape[0]} classes, the norm {n_classes}.")
        weights[:, 0, :] = 1 - strategy.P
        weights[:, 1, :] = strategy.P
    return weights


def prepare(rows):
    """Sorted rows and the arrays the figure is drawn from."""
    if not rows:
        raise ValueError("There are no ESS rows to plot.")
    rows = sorted(rows, key=lambda r: (r.cooperation, r.strategy.name, r.norm.barcode))
    n = rows[0].norm.partition.n_classes
    if any(r.norm.partition.n_classes != n for r in rows):
        raise ValueError("All rows must have the same number of score classes.")
    weights = np.array([cell_weights(r.strategy, n).reshape(-1) for r in rows])
    cn = np.array([_symbols(r.norm.CN) for r in rows])
    nc = np.array([_symbols(r.norm.NC) for r in rows])
    return SimpleNamespace(
        rows=rows, n=n, cn=cn, nc=nc, weights=weights,
        cooperation=np.array([r.cooperation for r in rows]),
        labels=[leading_label(r.norm, r.strategy) for r in rows],
        conservative=np.array([is_conservative(r.norm) for r in rows]),
    )


def _draw_vector(ax, symbols, weights):
    n_rows, n_cells = symbols.shape
    half = (NARROW + (WIDE - NARROW) * weights) / 2
    col = np.arange(n_cells)[None, :]
    row = np.arange(n_rows)[:, None]
    x0, x1 = col - half, col + half
    y0, y1 = row - 0.5 + 0 * x0, row + 0.5 + 0 * x0
    verts = np.stack([np.stack([x0, y0], -1), np.stack([x1, y0], -1), np.stack([x1, y1], -1), np.stack([x0, y1], -1)], -2)
    colors = np.array([COLORS[-1], COLORS[0], COLORS[1]])[symbols + 1]
    ax.add_collection(PolyCollection(verts.reshape(-1, 4, 2), facecolors=colors.reshape(-1, 3), edgecolors="none"))


def _draw_raster(ax, symbols, weights, resolution=24, chunk=4000):
    n_rows, n_cells = symbols.shape
    image = np.full((n_rows, n_cells * resolution, 3), 255, dtype=np.uint8)
    position = np.abs((np.arange(resolution) + 0.5) / resolution - 0.5)
    palette = (np.array([COLORS[-1], COLORS[0], COLORS[1]]) * 255).astype(np.uint8)
    for start in range(0, n_rows, chunk):
        sl = slice(start, start + chunk)
        half = (NARROW + (WIDE - NARROW) * weights[sl]) / 2
        inside = position[None, None, :] <= half[:, :, None]
        block = np.where(inside[..., None], palette[symbols[sl] + 1][:, :, None, :], 255)
        image[sl] = block.reshape(block.shape[0], n_cells * resolution, 3)
    ax.imshow(image, aspect="auto", interpolation="nearest", extent=(-0.5, n_cells - 0.5, n_rows - 0.5, -0.5))


def _draw_cells(ax, symbols, weights):
    n_rows, n_cells = symbols.shape
    ax.set_xlim(-0.5, n_cells - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    if n_rows * n_cells <= VECTOR_CELLS:
        _draw_vector(ax, symbols, weights)
    else:
        _draw_raster(ax, symbols, weights)
    if n_rows < LABEL_ROWS:
        for row in range(n_rows - 1):
            ax.axhline(row + 0.5, color="gray", linewidth=0.5, alpha=0.5)
    else:
        ax.axvline(n_cells / 2 - 0.5, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_header(ax, n, tau):
    """The three rows above the cells: recipient's class, whether help was given, donor's class."""
    cells = 2 * n * n
    ax.set_xlim(-0.5, cells - 0.5)
    ax.set_ylim(0, 3)
    ax.axis("off")
    classes = ["0", "1"] if n == 2 else ["0", f"1–{tau}", f">{tau}"]
    action = ["·", "☺"]                        # nothing given, help given
    rows = [("Donor", 1, classes * (2 * n)), ("Action", n, action * n), ("Recvr", 2 * n, classes)]
    font = 8 if n == 2 else 6
    for level, (name, span, texts) in enumerate(rows):
        for k, text in enumerate(texts):
            ax.add_patch(Rectangle((k * span - 0.5, level), span, 1, edgecolor="gray", facecolor="white"))
            ax.text(k * span - 0.5 + span / 2, level + 0.5, text, ha="center", va="center", color="gray", fontsize=font)
        ax.text(-0.7, level + 0.5, name, ha="right", va="center", color="gray", fontsize=8)


def _draw_bars(ax, cooperation, bar_colors):
    n_rows = len(cooperation)
    shown = np.clip(cooperation, 0, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(n_rows - 0.5, -0.5)
    if n_rows > 200:
        image = np.full((n_rows, 100, 3), 255, dtype=np.uint8)
        for row in range(n_rows):
            width = max(int(shown[row] * 100), 2 if shown[row] < 0.01 else 0)
            image[row, :width] = (np.array(bar_colors[row]) * 255).astype(np.uint8)
        ax.imshow(image, aspect="auto", interpolation="nearest", extent=(0, 1, n_rows - 0.5, -0.5))
    else:
        for row in range(n_rows):
            ax.add_patch(Rectangle((0, row - 0.45), max(shown[row], 0.02 if shown[row] < 0.01 else 0), 0.9,
                                   facecolor=bar_colors[row], edgecolor="none"))
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("Cooperation level", fontsize=11, labelpad=10)
    ax.set_yticks([])
    ax.spines[["left", "right", "bottom"]].set_visible(False)


def ess_figure(rows, title=None, show_cons=False, show_strategies=None, show_bb_labels=False, show_legend=True):
    """Figure of ESS rows (see ess_data.py). show_strategies defaults to on for up to 30 rows."""
    data = prepare(rows)
    n_rows, n = len(data.rows), data.n
    cells = 2 * n * n
    tau = data.rows[0].norm.partition.tau if n == 3 else None
    if show_strategies is None:
        show_strategies = n_rows <= STRATEGY_ROWS
    parts = [("CN", data.cn, "Donor's change")]
    if np.any(data.nc != 0):
        parts.append(("NC", data.nc, "Receiver's change"))

    is_leading = np.array([label is not None and label != MONEY.label for label in data.labels])
    bar_colors = [LEADING_BAR if flag else OTHER_BAR for flag in is_leading]
    names = [strategy_label(r.strategy) for r in data.rows]

    # Layout in inches.
    bb_width, gap = cells * CELL, 0.6
    strategy_width = 0.1 + 0.075 * max(len(name) for name in names)
    height = 0.2 * n_rows if n_rows <= STRATEGY_ROWS else (7.5 if n_rows <= 2000 else 10.0)
    left, top, bottom, ess_width = 1.0, 1.35, 0.5, 3.0
    x = left
    positions = {}
    for key, _, _ in parts:
        positions[key] = x
        x += bb_width + gap
    if show_strategies:
        positions["strategy"] = x
        x += strategy_width + 0.22
    if show_cons:
        positions["cons"] = x
        x += 0.3 + 0.22
    positions["ess"] = x
    width = x + ess_width + 0.3
    fig = plt.figure(figsize=(width, height + top + bottom))
    fig_w, fig_h = fig.get_size_inches()

    def add_axes(x_in, w_in, y_in=bottom, h_in=height):
        return fig.add_axes([x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h])

    for key, symbols, label in parts:
        ax = add_axes(positions[key], bb_width)
        _draw_cells(ax, symbols, data.weights)
        if show_bb_labels:
            ax.set_xlabel(label, fontsize=11, labelpad=10)
        if len(parts) > 1:
            ax.text(0.5, -0.02, label, transform=ax.transAxes, ha="center", va="top", fontsize=10, weight="bold")
    header = add_axes(positions["CN"], bb_width, y_in=bottom + height + 0.02, h_in=0.54)
    _draw_header(header, n, tau)

    if show_strategies:
        ax = add_axes(positions["strategy"], strategy_width)
        ax.set_xlim(0, 1)
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.axis("off")
        ax.text(0, -1, "ESS strategy", fontsize=8, color="gray", ha="left", va="center")
        previous = None
        mono = font_manager.FontProperties(family="monospace")
        for row, name in enumerate(names):
            shown = "″" if name == previous else name
            ax.text(0, row, shown, ha="left", va="center", fontsize=8, fontproperties=mono,
                    color=(0, 0, 0, 1) if is_leading[row] else (0.4, 0.4, 0.4, 1))
            previous = name

    if show_cons:
        ax = add_axes(positions["cons"], 0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.axis("off")
        ax.text(0.5, -1, "Cons", fontsize=8, color="gray", ha="center", va="center")
        for row in np.flatnonzero(data.conservative):
            money = data.labels[row] == MONEY.label
            ax.text(0.5, row, "$" if money else "✓", ha="center", va="center", fontsize=8 if n_rows > 100 else 10,
                    color="red" if money else "black")

    ax = add_axes(positions["ess"], ess_width)
    _draw_bars(ax, data.cooperation, bar_colors)
    if n_rows <= LABEL_ROWS:
        for row, label in enumerate(data.labels):
            if label is not None:
                ax.text(0.02, row, label, color="white", ha="left", va="center", fontsize=7, weight="bold")
    else:
        ax.text(0.45, n_rows * 0.25, f"{n_rows} rows in all", color="gray", ha="left", va="center", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="white", linewidth=0.5))

    if show_legend:
        legend = add_axes(positions["CN"], bb_width + 0.4, y_in=0.08, h_in=0.3)
        legend.set_xlim(0, 1)
        legend.set_ylim(0, 1)
        legend.axis("off")
        for k, (symbol, text) in enumerate(((1, "+1"), (0, "0"), (-1, "−1"))):
            legend.add_patch(Rectangle((0.02 + 0.13 * k, 0.15), 0.04, 0.7, facecolor=COLORS[symbol], edgecolor="none"))
            legend.text(0.07 + 0.13 * k, 0.5, text, ha="left", va="center", fontsize=7)
        legend.text(0.42, 0.5, "change of score; narrow = action not taken", ha="left", va="center", fontsize=7,
                    color="gray")
    if title:
        fig.text(left / fig_w, 1 - 0.12 / fig_h, title, ha="left", va="top", fontsize=11)
    return fig
