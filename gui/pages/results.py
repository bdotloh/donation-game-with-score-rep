"""Page 3: the simplex with arrows and rest points, the trajectory, and what happens at a chosen point."""

import io
import json
import zipfile

import numpy as np
from dash import ALL, Input, Output, Patch, State, ctx, dcc, html, no_update

from .. import figures, spec, validation
from ..compute import RUNS
from ..components import GRAPH_CONFIG, card, field, status
from .strategies import small_surface

GRAPHS = ["r-simplex", "r-hist", "r-payoff", "r-help"]


def layout():
    return html.Div(
        [
            html.Div(
                [
                    html.H1("Results"),
                    html.Div(
                        [
                            html.Button("← Parameters", id="r-back", className="secondary", n_clicks=0),
                            html.Span("Figures as", className="hint"),
                            dcc.RadioItems(id="r-format", value="png", inline=True,
                                           options=[{"label": "PNG", "value": "png"},
                                                    {"label": "SVG", "value": "svg"}]),
                            html.Button("Download data", id="r-download-data", className="secondary", n_clicks=0),
                            html.Button("Save settings", id="r-download-settings", className="secondary", n_clicks=0),
                            dcc.Download(id="r-download"),
                        ],
                        className="toolbar",
                    ),
                ],
                className="title-row",
            ),
            dcc.Interval(id="r-poll", interval=400),
            dcc.Interval(id="r-player", interval=120, disabled=True),
            dcc.Store(id="r-selection"),
            html.Div(id="r-progress"),
            html.Div(results_body(), id="r-body", style={"display": "none"}),
        ]
    )


def results_body():
    return [
        card("Strategies", [
            dcc.Checklist(id="r-surfaces", inline=True, className="checks",
                          options=[{"label": "Show the p(help) surfaces", "value": "on"}], value=[]),
            html.Div(id="r-strategies", className="strategy-strip"),
        ]),
        html.Div(
            [
                card(None, [
                    dcc.Graph(id="r-simplex", config=GRAPH_CONFIG),
                    html.Div(id="r-simplex-note", className="note"),
                ], className="card simplex-card"),
                card("At the selected point", [
                    html.Div(id="r-composition", className="row"),
                    html.Div([html.Button("Show", id="r-show", className="secondary", n_clicks=0),
                              html.Span("or click the simplex", className="hint")], className="row"),
                    html.Div(id="r-composition-error"),
                    html.Hr(),
                    html.Div([
                        html.Button("▶ Play", id="r-play", className="primary", n_clicks=0),
                        html.Span(id="r-time", className="time"),
                    ], className="row"),
                    dcc.Slider(id="r-frame", min=0, max=1, step=1, value=0, marks=None, updatemode="drag"),
                    html.Div(id="r-summary"),
                ], className="card side"),
            ],
            className="results-top",
        ),
        card("Score distribution", [
            dcc.Checklist(id="r-show-strategies", inline=True, className="checks"),
            html.Div([
                field("x from", dcc.Input(id="r-xmin", type="text", placeholder="auto", inputMode="decimal", className="num")),
                field("x to", dcc.Input(id="r-xmax", type="text", placeholder="auto", inputMode="decimal", className="num")),
                field("y from", dcc.Input(id="r-ymin", type="text", placeholder="auto", inputMode="decimal", className="num")),
                field("y to", dcc.Input(id="r-ymax", type="text", placeholder="auto", inputMode="decimal", className="num")),
                html.Div(id="r-range-error", className="status error"),
            ], className="row bottom"),
            dcc.Graph(id="r-hist", config=GRAPH_CONFIG),
        ]),
        html.Div(
            [
                card(None, dcc.Graph(id="r-payoff", config=GRAPH_CONFIG)),
                card(None, dcc.Graph(id="r-help", config=GRAPH_CONFIG)),
            ],
            className="two-col",
        ),
    ]


def register(app):
    @app.callback(
        Output("r-progress", "children"),
        Output("r-body", "style"),
        Output("r-poll", "disabled"),
        Output("r-simplex", "figure"),
        Output("r-simplex-note", "children"),
        Output("r-composition", "children"),
        Output("r-frame", "max"),
        Output("r-show-strategies", "options"),
        Output("r-show-strategies", "value"),
        Output("r-selection", "data"),
        Input("r-poll", "n_intervals"),
        State("store-run", "data"),
    )
    def poll(_, run_id):
        run = RUNS.get(run_id)
        nothing = (no_update,) * 7                      # the outputs after the first three
        if run is None:
            message = html.P(["Nothing has been run yet. ", html.A("Set the parameters", href="/parameters"), "."])
            return (message, {"display": "none"}, True) + nothing
        if run.state == "running":
            bar = html.Div([html.Div(run.message), html.Progress(value=f"{run.progress:.3f}", max="1")],
                           className="progress")
            return (bar, {"display": "none"}, False) + nothing
        if run.state == "error":
            return (status(f"The computation failed: {run.error}", False), {"display": "none"}, True) + nothing
        converged = np.array([r.converged for r in run.grid_results])
        fig = run.figure                    # drawn once, at the end of the computation
        note = []
        if not converged.any():
            note.append("The score chains don't converge anywhere here (no stable score distribution), so all "
                        "values are those after the set number of score iterations.")
        elif not converged.all() or not run.trajectory.scores_converged.all():
            note.append("Lighter arrows: the score chains didn't converge within the maximum number of iterations, "
                        "so values there depend on that maximum and on the initial scores.")
        if run.trajectory.at_rest:
            note.append(f"The trajectory reached a rest point at t = {run.trajectory.t[-1]:g}.")
        composition = [field(f"{label} · {name}",
                             dcc.Input(id={"type": "r-share", "index": i}, type="text", inputMode="decimal",
                                       value=f"{run.rho0[i]:.4g}", className="num"))
                       for i, (label, name) in enumerate(zip(run.labels, run.names))]
        options = ([{"label": f"{l} · {n}", "value": n} for l, n in zip(run.labels, run.names)]
                   + [{"label": "population", "value": "population"}])
        values = [o["value"] for o in options]
        selection = {"kind": "frame", "frame": 0}
        return (None, {}, True, fig, " ".join(note), composition, len(run.trajectory.t) - 1, options, values,
                selection)

    @app.callback(
        Output("r-strategies", "children"),
        Input("r-poll", "n_intervals"),
        Input("r-surfaces", "value"),
        State("store-run", "data"),
        State("r-strategies", "children"),
    )
    def show_strategies(_, surfaces, run_id, shown):
        run = RUNS.get(run_id)
        if run is None or (ctx.triggered_id == "r-poll" and shown):     # drawn once, and again for the checkbox
            return no_update
        return strategy_cards(run, surfaces=bool(surfaces))

    @app.callback(
        Output("r-selection", "data", allow_duplicate=True),
        Output("r-composition-error", "children"),
        Output("r-player", "disabled", allow_duplicate=True),
        Output("r-play", "children", allow_duplicate=True),
        Input("r-simplex", "clickData"),
        Input("r-show", "n_clicks"),
        Input("r-frame", "value"),
        State({"type": "r-share", "index": ALL}, "value"),
        State("r-player", "disabled"),
        prevent_initial_call=True,
    )
    def select(click, show, frame, shares, player_disabled):
        trigger = ctx.triggered_id
        if trigger == "r-frame":
            return {"kind": "frame", "frame": frame}, None, no_update, no_update
        pause = (True, "▶ Play")
        if trigger == "r-show":
            rho, error = validation.shares(shares)
            if error:
                return no_update, status(error, False), no_update, no_update
            return ({"kind": "point", "rho": rho.tolist()}, None) + pause
        if trigger == "r-simplex" and click:
            rho = click_composition(click)
            if rho is None:
                return no_update, None, no_update, no_update
            return ({"kind": "point", "rho": rho}, None) + pause
        return no_update, None, no_update, no_update

    @app.callback(
        Output("r-hist", "figure"),
        Output("r-payoff", "figure"),
        Output("r-help", "figure"),
        Output("r-summary", "children"),
        Output("r-time", "children"),
        Output("r-simplex", "figure", allow_duplicate=True),
        Output("r-range-error", "children"),
        Input("r-selection", "data"),
        Input("r-show-strategies", "value"),
        Input("r-xmin", "value"),
        Input("r-xmax", "value"),
        Input("r-ymin", "value"),
        Input("r-ymax", "value"),
        State("store-run", "data"),
        prevent_initial_call=True,
    )
    def show_selection(selection, shown, xmin, xmax, ymin, ymax, run_id):
        run = RUNS.get(run_id)
        if run is None or run.state != "done" or not selection:
            return (no_update,) * 7
        x_range, y_range, range_error = axis_ranges(xmin, xmax, ymin, ymax)
        if selection["kind"] == "frame":
            state = run.frame(min(selection["frame"], len(run.trajectory.t) - 1))
        else:
            state = run.point(selection["rho"])
        shown = shown or []
        p = present(state["rho"])                    # strategies with share 0 are left out of these panels
        names, labels = [run.names[i] for i in p], [run.labels[i] for i in p]
        hidden = [n for n in names if n not in shown]
        hist = figures.score_histogram(names, np.asarray(state["rho"])[p], np.asarray(state["w"])[p], hidden=hidden,
                                       labels=labels, colors=[figures.COLORS[i] for i in p],
                                       show_marginal="population" in shown, x_range=x_range, y_range=y_range)
        payoff = figures.matrix_heatmap(np.asarray(state["payoff_matrix"])[np.ix_(p, p)], labels,
                                        "Payoff matrix A  (π = Aρ)", "strategy", "against", diverging=True)
        help_ = figures.matrix_heatmap(np.asarray(state["help_matrix"])[np.ix_(p, p)], labels, "Helping rates H",
                                       "donor", "receiver")
        time = f"t = {state['time']:g}" if state["time"] is not None else "selected point"
        patch = Patch()
        if run.n == 3:
            x, y = figures.to_xy(state["rho"])
            patch["data"][-1]["x"], patch["data"][-1]["y"] = list(x), list(y)
        else:
            patch["data"][-1]["x"], patch["data"][-1]["y"] = [state["rho"][1]], [0]
        return hist, payoff, help_, summary(run, state, p), time, patch, status(range_error, False)

    @app.callback(
        Output("r-player", "disabled"),
        Output("r-play", "children"),
        Output("r-frame", "value", allow_duplicate=True),
        Input("r-play", "n_clicks"),
        State("r-player", "disabled"),
        State("r-frame", "value"),
        State("r-frame", "max"),
        prevent_initial_call=True,
    )
    def play(n_clicks, disabled, frame, last):
        if not n_clicks:
            return no_update, no_update, no_update
        if disabled:
            start = 0 if frame is None or frame >= last else frame
            return False, "❚❚ Pause", start
        return True, "▶ Play", no_update

    @app.callback(
        Output("r-frame", "value", allow_duplicate=True),
        Output("r-player", "disabled", allow_duplicate=True),
        Output("r-play", "children", allow_duplicate=True),
        Input("r-player", "n_intervals"),
        State("r-frame", "value"),
        State("r-frame", "max"),
        prevent_initial_call=True,
    )
    def advance(_, frame, last):
        frame = (frame or 0) + 1
        if frame >= last:
            return last, True, "▶ Play"
        return frame, no_update, no_update

    @app.callback(
        *[Output(g, "config") for g in GRAPHS],
        Input("r-format", "value"),
    )
    def figure_format(fmt):
        config = {**GRAPH_CONFIG, "toImageButtonOptions": {"format": fmt, "scale": 2 if fmt == "png" else 1}}
        return [config] * len(GRAPHS)

    @app.callback(Output("url", "pathname", allow_duplicate=True), Input("r-back", "n_clicks"),
                  prevent_initial_call=True)
    def back(n_clicks):
        return "/parameters" if n_clicks else no_update

    @app.callback(
        Output("r-download", "data"),
        Input("r-download-data", "n_clicks"),
        Input("r-download-settings", "n_clicks"),
        State("store-run", "data"),
        State("r-selection", "data"),
        prevent_initial_call=True,
    )
    def download(data_clicks, settings_clicks, run_id, selection):
        run = RUNS.get(run_id)
        if run is None or not ctx.triggered[0]["value"]:
            return no_update
        if ctx.triggered_id == "r-download-settings":
            return dict(content=json.dumps(run.experiment, indent=2), filename="replicator_settings.json")
        if run.state != "done":
            return no_update
        return dcc.send_bytes(lambda buffer: buffer.write(results_zip(run, selection)), "replicator_results.zip")


def click_composition(click):
    """The shares stored with a clicked point (Plotly may send them as a list or as {"0": .., "1": ..})."""
    data = click["points"][0].get("customdata") if click.get("points") else None
    if isinstance(data, dict):
        data = [data[k] for k in sorted(data, key=int)]
    if not isinstance(data, (list, tuple)) or not data:
        return None
    try:
        rho = [float(x) for x in data]
    except (TypeError, ValueError):
        return None
    return rho if min(rho) >= 0 and sum(rho) > 0 else None


def strategy_cards(run, surfaces=False):
    """One card per strategy of the run: its label and colour, name, initial share and what it does."""
    cards = []
    for i, cfg in enumerate(run.experiment["strategies"]):
        color = figures.COLORS[i]
        children = [
            html.Div([html.Span(run.labels[i], className="chip", style={"background": color}), html.B(cfg["name"])],
                     className="strategy-head"),
            html.Div(f"{spec.describe_strategy(cfg)} · initial share {run.rho0[i]:.4g}", className="hint"),
            strategy_table(cfg),
        ]
        if surfaces:
            children.append(dcc.Graph(figure=small_surface(cfg), config={"displayModeBar": False}, className="mini"))
        cards.append(html.Div(children, className="strategy-card", style={"borderLeftColor": color}))
    return cards


def strategy_table(cfg):
    """The decision points of a strategy: p(help) for each class of receiver and donor (phi-Disc: by donor score)."""
    if cfg["type"] == "phi":
        scores = [1, 2, 3, 5, 10]
        table = html.Table(
            [html.Tr([html.Th("donor score")] + [html.Th(str(j)) for j in scores]),
             html.Tr([html.Th("p(help)")] + [html.Td(f"{j ** -cfg['phi']:.3g}", style=shade(j ** -cfg["phi"]))
                                             for j in scores])],
            className="grid pgrid")
        return html.Div([table, html.Div("Only helps a receiver with a score, and never when its own score is 0.",
                                         className="hint")])
    labels = spec.class_labels(cfg["type"], cfg.get("tau"))
    header = html.Tr([html.Th("receiver ↓  donor →", className="corner")] + [html.Th(f"score {c}") for c in labels])
    rows = [html.Tr([html.Th(f"score {labels[r]}")]
                    + [html.Td(f"{p:g}", style=shade(p)) for p in row])
            for r, row in enumerate(cfg["P"])]
    return html.Table([header] + rows, className="grid pgrid")


def shade(p):
    """Cell colour that grows with the probability of helping."""
    return {"background": f"rgba(47, 95, 167, {0.06 + 0.42 * p:.2f})"}


def axis_ranges(xmin, xmax, ymin, ymax):
    """((x low, x high), (y low, y high), error): typed axis limits of the score distribution, None for auto.
    A blank box is automatic; limits that can't be used are ignored with a message."""
    values = []
    for text, what in ((xmin, "x from"), (xmax, "x to"), (ymin, "y from"), (ymax, "y to")):
        if text is None or str(text).strip() == "":
            values.append(None)
            continue
        number = validation.parse_number(text)
        if number is None:
            return (None, None), (None, None), f"“{text}” is not a number ({what}); the axes are automatic."
        values.append(number)
    x_range, y_range = tuple(values[:2]), tuple(values[2:])
    if x_range[1] is not None and x_range[1] > 5000:
        return (None, None), (None, None), "x to can be at most 5000; the axes are automatic."
    if y_range[0] is not None and y_range[0] < 0:
        return (None, None), (None, None), "y from can't be negative; the axes are automatic."
    for low, high, what in ((*x_range, "x"), (*y_range, "y")):
        if low is not None and high is not None and low >= high:
            return (None, None), (None, None), f"{what} from must be below {what} to; the axes are automatic."
    return x_range, y_range, ""


def present(rho, tol=1e-9):
    """Indices of the strategies with a positive share."""
    return [i for i, r in enumerate(rho) if r > tol]


def summary(run, state, shown=None):
    rho, pi, v, w = state["rho"], state["payoffs"], state["velocity"], state["w"]
    scores = np.arange(w.shape[1])
    header = html.Tr([html.Th(""), html.Th("share"), html.Th("payoff π"), html.Th("dρ/dt"), html.Th("w₀"),
                      html.Th("mean score")])
    rows = [
        html.Tr([html.Th(html.Span([html.B(run.labels[i]), f" {name}"], style={"color": figures.COLORS[i]})),
                 html.Td(f"{rho[i]:.4f}"),
                 html.Td(f"{pi[i]:.4f}"), html.Td(f"{v[i]:+.2g}"), html.Td(f"{w[i, 0]:.4f}"),
                 html.Td(f"{scores @ w[i]:.3f}")])
        for i, name in enumerate(run.names) if shown is None or i in shown
    ]
    rows.append(html.Tr([html.Th("mean"), html.Td(""), html.Td(f"{rho @ pi:.4f}"), html.Td(""), html.Td(""),
                         html.Td("")]))
    children = [html.Table([header] + rows, className="list summary")]
    if not state["converged"]:
        children.append(html.Div("Scores didn't converge within the maximum number of iterations here.",
                                 className="note"))
    return children


def results_zip(run, selection):
    """Settings, trajectory, vector field, rest points and the selected point, as CSV/JSON in a zip."""
    names = run.labels                     # column names S1, S2, S3; settings.json has the strategies
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("settings.json", json.dumps(run.experiment, indent=2))
        t = run.trajectory
        cols = (["t"] + [f"share_{n}" for n in names] + [f"payoff_{n}" for n in names]
                + [f"w0_{n}" for n in names] + [f"mean_score_{n}" for n in names] + ["scores_converged"])
        lines = [",".join(cols)]
        for k in range(len(t.t)):
            row = [t.t[k], *t.rho[k], *t.payoffs[k], *t.w0[k], *t.mean_score[k], int(t.scores_converged[k])]
            lines.append(",".join(f"{x:.10g}" for x in row))
        z.writestr("trajectory.csv", "\n".join(lines))
        lines = [",".join([f"share_{n}" for n in names] + [f"velocity_{n}" for n in names]
                          + [f"payoff_{n}" for n in names] + ["scores_converged"])]
        for p, r in zip(run.grid, run.grid_results):
            lines.append(",".join(f"{x:.10g}" for x in [*p, *r.velocity, *r.payoffs, int(r.converged)]))
        z.writestr("vector_field.csv", "\n".join(lines))
        z.writestr("rest_points.json", json.dumps(run.rest_point_dicts(), indent=2, default=str))
        if selection:
            state = run.frame(selection["frame"]) if selection["kind"] == "frame" else run.point(selection["rho"])
            z.writestr("selected_point.json", json.dumps({
                "strategies": dict(zip(run.labels, run.names)),
                "share": list(state["rho"]),
                "payoff": list(state["payoffs"]),
                "velocity": list(state["velocity"]),
                "payoff_matrix": np.asarray(state["payoff_matrix"]).tolist(),
                "help_matrix": np.asarray(state["help_matrix"]).tolist(),
                "scores_converged": state["converged"],
                "time": state["time"],
            }, indent=2, default=float))
            w = state["w"]
            lines = [",".join(["score"] + names + ["population"])]
            marginal = state["rho"] @ w
            for j in range(w.shape[1]):
                lines.append(",".join(f"{x:.10g}" for x in [j, *w[:, j], marginal[j]]))
            z.writestr("selected_score_distribution.csv", "\n".join(lines))
    return buffer.getvalue()
