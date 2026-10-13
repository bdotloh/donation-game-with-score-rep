"""Plotly figures for the GUI. Pure functions of numbers, so they can be tested and exported."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = ["#3B6FB6", "#D9822B", "#3A9D5D"]
MARGINAL = "#333333"
REST_STYLE = {
    "stable": {"color": "black", "line": "black", "label": "stable"},
    "unstable": {"color": "white", "line": "black", "label": "unstable"},
    "saddle": {"color": "#9A9A9A", "line": "black", "label": "saddle"},
    "neutral": {"color": "white", "line": "#9A9A9A", "label": "neutral"},
}
SQRT3_2 = np.sqrt(3) / 2
FONT = dict(family="Inter, Helvetica, Arial, sans-serif", size=13)


def default_labels(n):
    return [f"S{i + 1}" for i in range(n)]


def _layout(fig, **kwargs):
    kwargs.setdefault("margin", dict(l=40, r=20, t=40, b=40))
    fig.update_layout(template="simple_white", font=FONT, **kwargs)
    return fig


# ---------------------------------------------------------------- strategies


def help_surface_values(strategy, max_score):
    """H[d, r] = p(help) for receiver score r and donor score d (rows: donor, as plotly's z)."""
    n = max_score + 1
    receiver_class = strategy.partition.classify(np.arange(n))
    help_ = strategy.donor_help(n)                       # [receiver class, donor score]
    return help_[receiver_class].T                        # [donor score, receiver score]


def strategy_surface(strategy, max_score=10, title=None):
    """p(help) over (receiver score, donor score), drawn as flat plateaus over each integer score."""
    values = help_surface_values(strategy, max_score)
    delta = 0.02
    edges = np.arange(max_score + 2) - 0.5
    coords = np.sort(np.concatenate([edges[:-1] + delta, edges[1:] - delta]))
    index = np.clip(np.round(coords).astype(int), 0, max_score)
    z = values[np.ix_(index, index)]
    fig = go.Figure(
        go.Surface(
            x=coords,
            y=coords,
            z=z,
            cmin=0,
            cmax=1,
            colorscale=[[0, "#cfdcf0"], [1, "#1f4e96"]],
            showscale=False,
            lighting=dict(ambient=0.75, diffuse=0.5),
            hovertemplate="receiver score %{x:.0f}<br>donor score %{y:.0f}<br>p(help) %{z:.3g}<extra></extra>",
        )
    )
    axis = dict(range=[-0.5, max_score + 0.5], dtick=max(1, max_score // 5))
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="receiver score", **axis),
            yaxis=dict(title="donor score", **axis),
            zaxis=dict(title="p(help)", range=[0, 1.05], dtick=0.25),
            camera=dict(eye=dict(x=-1.6, y=-1.6, z=1.0)),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.6),
        ),
        title=title,
        margin=dict(l=0, r=0, t=40 if title else 10, b=0),
        font=FONT,
        height=380,
    )
    return fig


# ---------------------------------------------------------------- simplex


def to_xy(rho):
    """Barycentric (rho_1, rho_2, rho_3) to the plane: strategy 1 bottom left, 2 bottom right, 3 top."""
    rho = np.atleast_2d(rho)
    return rho[:, 1] + 0.5 * rho[:, 2], SQRT3_2 * rho[:, 2]


def from_xy(x, y):
    rho3 = y / SQRT3_2
    rho2 = x - 0.5 * rho3
    return np.array([1 - rho2 - rho3, rho2, rho3])


def flow_lines(grid, velocities, dsep=0.03, step=0.004, max_steps=2000):
    """Evenly spaced flow lines of the velocity field on the triangle, as x and y lists with
    None between lines. The field is interpolated linearly between the grid points, and a new
    line stops where it comes within dsep of another one (Jobard and Lefer's method)."""
    from scipy.interpolate import LinearNDInterpolator

    x0, y0 = to_xy(grid)
    vx, vy = velocities[:, 1] + 0.5 * velocities[:, 2], SQRT3_2 * velocities[:, 2]
    speed_max = np.hypot(vx, vy).max()
    if speed_max == 0:
        return [], []
    field = LinearNDInterpolator(np.column_stack([x0, y0]), np.column_stack([vx, vy]))

    def direction(p):
        v = np.ravel(field(p[0], p[1]))
        speed = np.hypot(v[0], v[1])
        if not np.isfinite(speed) or speed < 1e-4 * speed_max:
            return None
        return v / speed

    def inside(p):
        return from_xy(p[0], p[1]).min() >= -1e-9

    def cell(p):
        return int(p[0] // dsep), int(p[1] // dsep)

    taken = set()

    def trace(seed, sign):
        points, own = [], set()
        p = np.array(seed, dtype=float)
        for _ in range(max_steps):
            d1 = direction(p)
            if d1 is None:
                break
            mid = p + 0.5 * step * sign * d1
            d2 = direction(mid) if inside(mid) else None
            if d2 is None:
                break
            q = p + step * sign * d2
            if not inside(q):
                break
            c = cell(q)
            if c in taken and c not in own:
                break
            own.add(c)
            points.append(q)
            if len(points) > 50 and np.hypot(q[0] - seed[0], q[1] - seed[1]) < step:
                break                           # back at the start: a closed orbit
            p = q
        return points, own

    xs, ys = [], []
    lattice = max(2, int(round(1 / dsep)))
    seeds = [to_xy(np.array([i, j, lattice - i - j]) / lattice) for i in range(lattice + 1)
             for j in range(lattice + 1 - i)]
    for sx, sy in seeds:
        seed = np.array([sx[0], sy[0]])
        if cell(seed) in taken:
            continue
        back, own_back = trace(seed, -1)
        forward, own_forward = trace(seed, 1)
        line = back[::-1] + [seed] + forward
        if len(line) < 8:
            continue
        taken.update(own_back | own_forward | {cell(seed)})
        xs += [p[0] for p in line] + [None]
        ys += [p[1] for p in line] + [None]
    return xs, ys


def _arrow_traces(x0, y0, dx, dy, magnitude, vmax, converged, hover, customdata, head_size=9):
    """Arrows coloured and sized by speed; non-converged arrows are drawn lighter."""
    traces = []
    colorscale = "Viridis"
    norm = magnitude / vmax if vmax > 0 else np.zeros_like(magnitude)
    bins = np.minimum((norm * 8).astype(int), 7)
    from plotly.colors import sample_colorscale

    palette = sample_colorscale(colorscale, [(b + 0.5) / 8 for b in range(8)])
    for ok in (True, False):
        for b in range(8):
            sel = (bins == b) & (converged == ok) & (magnitude > 0)
            if not sel.any():
                continue
            xs, ys = [], []
            for a, c, e, f in zip(x0[sel], y0[sel], dx[sel], dy[sel]):
                xs += [a, a + e, None]
                ys += [c, c + f, None]
            opacity = 1.0 if ok else 0.4
            traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=palette[b], width=2),
                                     opacity=opacity, hoverinfo="skip", showlegend=False))
            angle = np.degrees(np.arctan2(dx[sel], dy[sel]))
            traces.append(go.Scatter(x=x0[sel] + dx[sel], y=y0[sel] + dy[sel], mode="markers",
                                     marker=dict(symbol="triangle-up", size=head_size, angle=angle, color=palette[b]),
                                     opacity=opacity, hoverinfo="skip", showlegend=False))
    # Arrow bases carry the hover text and are clickable.
    traces.append(go.Scatter(x=x0, y=y0, mode="markers", marker=dict(size=4, color="rgba(0,0,0,0.25)"),
                             hovertext=hover, hoverinfo="text", customdata=np.asarray(customdata).tolist(),
                             showlegend=False, name="arrows"))
    # Colour bar for speed.
    traces.append(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(colorscale=colorscale, cmin=0, cmax=vmax, color=[0], showscale=True,
                                         colorbar=dict(title=dict(text="speed", side="right"), thickness=12,
                                                       len=0.6, tickformat=".2g"))))
    return traces


def _hover_point(rho, velocity, converged, names):
    lines = [f"{n}: {r:.3f} (d/dt {v:+.2g})" for n, r, v in zip(names, rho, velocity)]
    if not converged:
        lines.append("<i>scores didn't converge within max_iter</i>")
    return "<br>".join(lines)


def _rest_trace(points, xy, names):
    traces = []
    for kind, style in REST_STYLE.items():
        sel = [p for p in points if p["kind"] == kind]
        if not sel:
            continue
        x, y = xy([p["rho"] for p in sel])
        hover = []
        for p in sel:
            text = f"<b>{style['label']}</b><br>" + "<br>".join(f"{n}: {r:.3f}" for n, r in zip(names, p["rho"]))
            text += "<br>eigenvalues: " + ", ".join(f"{e:.2g}" for e in p["eigenvalues"])
            if not p.get("converged", True):
                text += "<br><i>scores didn't converge within max_iter</i>"
            hover.append(text)
        traces.append(go.Scatter(x=x, y=y, mode="markers", name=style["label"],
                                 marker=dict(size=13, color=style["color"], line=dict(color=style["line"], width=1.5)),
                                 hovertext=hover, hoverinfo="text",
                                 customdata=[np.asarray(p["rho"], dtype=float).tolist() for p in sel]))
    return traces


def simplex_figure(names, grid, velocities, converged, rest_points, trajectory=None, current=None, click_resolution=60,
                   labels=None, flow=True):
    """The simplex (a line for 2 strategies, a triangle for 3) with velocity arrows and rest points.

    grid: (m, n) points, velocities: (m, n), converged: (m,) bools, rest_points: list of
    {"rho", "kind", "eigenvalues", "converged"}, trajectory: (T, n) or None, current: (n,) or None.
    labels: short labels (default S1, S2, S3), shown with the names at the corners. flow: draw
    flow lines (triangle only). The last trace is the current point, so it can be moved with a Patch.
    """
    n = len(names)
    labels = labels or default_labels(n)
    grid, velocities, converged = np.asarray(grid), np.asarray(velocities), np.asarray(converged, dtype=bool)
    hover = [_hover_point(r, v, c, labels) for r, v, c in zip(grid, velocities, converged)]
    if not converged.any():
        converged = np.ones_like(converged)     # fading every arrow would say nothing; the note says it
    if n == 2:
        return _line_figure(names, labels, grid, velocities, converged, rest_points, trajectory, current, hover,
                            click_resolution)
    return _triangle_figure(names, labels, grid, velocities, converged, rest_points, trajectory, current, hover,
                            click_resolution, flow)


def _triangle_figure(names, labels, grid, velocities, converged, rest_points, trajectory, current, hover,
                     click_resolution, flow):
    fig = go.Figure()
    # Invisible dense layer that makes every part of the triangle clickable.
    k = click_resolution
    lattice = np.array([(i, j, k - i - j) for i in range(k + 1) for j in range(k + 1 - i)]) / k
    lx, ly = to_xy(lattice)
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="markers", marker=dict(size=10, opacity=0), hoverinfo="none",
                             customdata=lattice.tolist(), showlegend=False, name="click"))
    fig.add_trace(go.Scatter(x=[0, 1, 0.5, 0], y=[0, 0, SQRT3_2, 0], mode="lines", line=dict(color="black", width=1.5),
                             hoverinfo="skip", showlegend=False))
    if flow:
        fx, fy = flow_lines(grid, velocities)
        fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines", line=dict(color="rgba(90,90,90,0.45)", width=1),
                                 hoverinfo="skip", name="flow lines"))
    x0, y0 = to_xy(grid)
    vx, vy = velocities[:, 1] + 0.5 * velocities[:, 2], SQRT3_2 * velocities[:, 2]
    magnitude = np.hypot(vx, vy)
    vmax = magnitude.max() if magnitude.size else 0
    spacing = 1 / max(1, round(np.sqrt(2 * len(grid))) - 1)
    scale = 0.8 * spacing / vmax if vmax > 0 else 0
    head = 9 if spacing >= 0.1 else 6
    for t in _arrow_traces(x0, y0, vx * scale, vy * scale, magnitude, vmax, converged, hover, grid, head):
        fig.add_trace(t)
    if trajectory is not None and len(trajectory):
        tx, ty = to_xy(trajectory)
        fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines", line=dict(color="#C0392B", width=2.5),
                                 name="trajectory", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=tx[:1], y=ty[:1], mode="markers", name="start",
                                 marker=dict(symbol="square", size=9, color="#C0392B"), hoverinfo="skip"))
    for t in _rest_trace(rest_points, to_xy, labels):
        fig.add_trace(t)
    pad = 0.07
    corners = [(0, -pad), (1, -pad), (0.5, SQRT3_2 + pad)]
    for label, name, (x, y) in zip(labels, names, corners):
        fig.add_annotation(x=x, y=y, text=f"<b>{label}</b><br><span style='font-size:11px;color:#555'>{name}</span>",
                           showarrow=False, font=dict(size=15))
    cx, cy = to_xy(current) if current is not None else ([None], [None])
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="markers", name="selected",
                             marker=dict(symbol="x", size=12, color="#C0392B", line=dict(width=1, color="white")),
                             hoverinfo="skip"))
    fig.update_xaxes(visible=False, range=[-0.1, 1.1])
    fig.update_yaxes(visible=False, range=[-0.14, SQRT3_2 + 0.13], scaleanchor="x", scaleratio=1)
    return _layout(fig, height=600, hovermode="closest", clickmode="event",
                   legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center"))


def _line_figure(names, labels, grid, velocities, converged, rest_points, trajectory, current, hover,
                 click_resolution):
    """Two strategies: x = share of the second strategy; top panel d rho_2/dt, bottom panel arrows on the line."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
    x = grid[:, 1]
    v = velocities[:, 1]
    order = np.argsort(x)
    fig.add_trace(go.Scatter(x=x[order], y=v[order], mode="lines+markers", line=dict(color="#555", width=1.5),
                             marker=dict(size=5, color=np.where(converged[order], "#555", "#BBB")),
                             hovertext=np.array(hover, dtype=object)[order], hoverinfo="text",
                             customdata=grid[order].tolist(), showlegend=False, name="velocity"), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="#999", width=1), row=1, col=1)
    # Clickable layer along the line.
    fine = np.linspace(0, 1, click_resolution + 1)
    fig.add_trace(go.Scatter(x=fine, y=np.zeros_like(fine), mode="markers", marker=dict(size=14, opacity=0),
                             hoverinfo="none", customdata=np.column_stack([1 - fine, fine]).tolist(), showlegend=False,
                             name="click"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 0], mode="lines", line=dict(color="black", width=1.5),
                             hoverinfo="skip", showlegend=False), row=2, col=1)
    magnitude = np.abs(v)
    vmax = magnitude.max() if magnitude.size else 0
    spacing = 1 / max(1, len(grid) - 1)
    scale = 0.8 * spacing / vmax if vmax > 0 else 0
    for t in _arrow_traces(x, np.zeros_like(x), v * scale, np.zeros_like(x), magnitude, vmax, converged, hover, grid):
        fig.add_trace(t, row=2, col=1)
    if trajectory is not None and len(trajectory):
        fig.add_trace(go.Scatter(x=[trajectory[0, 1]], y=[0], mode="markers", name="start",
                                 marker=dict(symbol="square", size=9, color="#C0392B"), hoverinfo="skip"), row=2, col=1)
    for t in _rest_trace(rest_points, lambda r: (np.asarray(r)[:, 1], np.zeros(len(r))), labels):
        fig.add_trace(t, row=2, col=1)
    cx = [current[1]] if current is not None else [None]
    fig.add_trace(go.Scatter(x=cx, y=[0] if current is not None else [None], mode="markers", name="selected",
                             marker=dict(symbol="x", size=12, color="#C0392B"), hoverinfo="skip"), row=2, col=1)
    fig.update_yaxes(title_text=f"d(share of {labels[1]})/dt", row=1, col=1, zeroline=False)
    fig.update_yaxes(visible=False, range=[-1, 1], row=2, col=1)
    fig.update_xaxes(range=[-0.05, 1.05], row=2, col=1,
                     title_text=f"share of {labels[1]}   ({labels[0]}: {names[0]},  {labels[1]}: {names[1]})",
                     tickvals=[0, 0.25, 0.5, 0.75, 1],
                     ticktext=[f"all {labels[0]}", "0.25", "0.5", "0.75", f"all {labels[1]}"])
    return _layout(fig, height=480, hovermode="closest", clickmode="event",
                   legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"))


# ---------------------------------------------------------------- scores and matrices


def score_histogram(names, rho, w, hidden=(), show_marginal=True, labels=None):
    """Conditional score distributions w[i] for each strategy (overlaid, semi-transparent), and the
    population's marginal sum_i rho_i w[i]."""
    labels = labels or default_labels(len(names))
    w = np.asarray(w)
    rho = np.asarray(rho)
    marginal = rho @ w
    scores = np.arange(w.shape[1])
    tail = np.cumsum(w[:, ::-1], axis=1)[:, ::-1].max(axis=0)
    upper = max(5, int(np.nonzero(tail > 1e-4)[0].max()) + 1 if (tail > 1e-4).any() else 5)
    upper = min(upper, w.shape[1] - 1)
    fig = go.Figure()
    ymax = 0
    for i, name in enumerate(names):
        if name in hidden:
            continue
        mean = scores @ w[i]
        ymax = max(ymax, w[i, : upper + 1].max())
        label = f"{labels[i]} · {name}"
        fig.add_trace(go.Bar(x=scores[: upper + 1], y=w[i, : upper + 1], name=label, legendgroup=name,
                             marker=dict(color=COLORS[i], line=dict(color=COLORS[i], width=1.5)), opacity=0.45,
                             hovertemplate=f"{labels[i]}<br>score %{{x}}<br>probability %{{y:.4f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[mean, mean], y=[0, 1], yaxis="y2", mode="lines", legendgroup=name,
                                 line=dict(color=COLORS[i], dash="dash", width=2), showlegend=False,
                                 hovertemplate=f"{labels[i]}: mean score {mean:.3f}<extra></extra>"))
    if show_marginal:
        mean = scores @ marginal
        ymax = max(ymax, marginal[: upper + 1].max())
        fig.add_trace(go.Scatter(x=scores[: upper + 1], y=marginal[: upper + 1], mode="lines+markers",
                                 line=dict(color=MARGINAL, shape="hvh", width=2), marker=dict(size=4),
                                 name="population", legendgroup="population",
                                 hovertemplate="population<br>score %{x}<br>probability %{y:.4f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[mean, mean], y=[0, 1], yaxis="y2", mode="lines", legendgroup="population",
                                 line=dict(color=MARGINAL, dash="dot", width=2), showlegend=False,
                                 hovertemplate=f"population: mean score {mean:.3f}<extra></extra>"))
    fig.update_layout(barmode="overlay", bargap=0.05,
                      xaxis=dict(title="score  (dashed lines: mean scores)", range=[-0.6, upper + 0.6]),
                      yaxis=dict(title="probability", rangemode="tozero"),
                      yaxis2=dict(overlaying="y", range=[0, 1], visible=False, fixedrange=True),
                      legend=dict(orientation="h", y=1.08, x=0, xanchor="left"))
    return _layout(fig, height=380, hovermode="closest")


def matrix_heatmap(M, names, title, row_title, col_title, diverging=False, fmt=".3f"):
    M = np.asarray(M)
    if diverging:
        bound = float(np.abs(M).max()) or 1.0
        colors = dict(colorscale="RdBu", zmid=0, zmin=-bound, zmax=bound)
    else:
        colors = dict(colorscale="Blues", zmin=0, zmax=1)
    text = [[format(x, fmt) for x in row] for row in M]
    fig = go.Figure(go.Heatmap(z=M, x=names, y=names, text=text, texttemplate="%{text}", textfont=dict(size=14),
                               showscale=False, hovertemplate=f"{row_title} %{{y}}<br>{col_title} %{{x}}<br>%{{z:.4g}}<extra></extra>",
                               **colors))
    fig.update_yaxes(autorange="reversed", title=row_title)
    fig.update_xaxes(side="top", title=col_title)
    return _layout(fig, height=300, title=dict(text=title, y=0.98), margin=dict(l=60, r=20, t=80, b=20))
