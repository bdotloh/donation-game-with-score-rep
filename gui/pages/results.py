"""Page 3: the simplex with arrows and rest points, the trajectory, and what happens at a chosen point."""

import base64
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from dash import ALL, Input, Output, Patch, State, ctx, dcc, html, no_update

from .. import figures, saves, spec, validation
from ..compute import RUNS
from ..components import GRAPH_CONFIG, card, field, status
from .strategies import params_from_experiment, small_surface

GRAPHS = ["r-simplex", "r-hist", "r-payoff", "r-help"]
PLACEHOLDER = html.Div("Click a point on the simplex, or type its shares and press Show; then evolve the dynamics "
                       "from it.", className="hint")


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
            saves_card(),
            dcc.Interval(id="r-poll", interval=400),
            dcc.Interval(id="r-player", interval=120, disabled=True),
            dcc.Interval(id="r-evolve-poll", interval=400, disabled=True),
            dcc.Store(id="r-selection"),
            html.Div(id="r-progress"),
            html.Div(results_body(), id="r-body", style={"display": "none"}),
        ]
    )


def save_options():
    return [{"label": entry["label"], "value": entry["file"]} for entry in saves.list_saves()]


def saves_card():
    return card("Saved results", [
        html.Div([
            field("Save these results as", dcc.Input(id="r-save-name", type="text", placeholder="a name",
                                                     maxLength=saves.MAX_NAME)),
            html.Button("Save", id="r-save", className="secondary", n_clicks=0),
            field("Load a saved result", dcc.Dropdown(id="r-load-choice", options=save_options(),
                                                      placeholder="none saved yet" if not save_options() else "choose one"),
                  className="field wide"),
            html.Button("Load", id="r-load", className="secondary", n_clicks=0),
            html.Button("Delete", id="r-delete", className="link", n_clicks=0),
            dcc.ConfirmDialog(id="r-confirm-delete", message=""),
            field("Rename the chosen save to", dcc.Input(id="r-rename-name", type="text", placeholder="a new name",
                                                         maxLength=saves.MAX_NAME)),
            html.Button("Rename", id="r-rename", className="secondary", n_clicks=0),
        ], className="row bottom"),
        html.Div([
            html.Button("Download file", id="r-download-file", className="secondary", n_clicks=0),
            html.Button("Download as CSV", id="r-download-csv", className="secondary", n_clicks=0),
            dcc.Upload(html.Button("Upload a saved file…", className="secondary"), id="r-upload", accept=".sqlite",
                       max_size=saves.MAX_FILE_BYTES),
            dcc.Download(id="r-save-download"),
            html.Span("(of the save chosen above)", className="hint"),
        ], className="row bottom"),
        html.Div(id="r-save-status"),
        html.Div(f"Saves are SQLite files in the folder {saves.default_folder()}/ (next to where the GUI was started): "
                 "tables of the points on the simplex with their shares, velocities, payoffs, payoff and helping "
                 "matrices and score distributions, which any Python (sqlite3, pandas) or R can read. A save brings "
                 "back the plots and rest points without computing them again, with all its settings (evolutions from a "
                 "point are not saved). "
                 "The far tail of each score distribution (probabilities below 1e-12) is not stored.",
                 className="hint"),
    ])


def results_body():
    return [
        card("Score update norm", html.Div(id="r-norm")),
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
                        html.Button("Evolve from point", id="r-evolve", className="primary", n_clicks=0,
                                    disabled=True),
                        html.Span(id="r-evolve-status", className="time"),
                    ], className="row"),
                    html.Details([
                        html.Summary("Advanced"),
                        html.Div([
                            field("Step size dt", dcc.Input(id="r-dt", type="text", inputMode="decimal",
                                                            value=f"{spec.DEFAULT_ADVANCED['dt']:g}", className="num")),
                            field("Steps", dcc.Input(id="r-steps", type="text", inputMode="numeric",
                                                     value=f"{spec.DEFAULT_ADVANCED['n_steps']:g}", className="num")),
                        ], className="row wrap"),
                    ], className="advanced"),
                    html.Div([
                        html.Div([
                            html.Button("▶ Play", id="r-play", className="primary", n_clicks=0),
                            html.Span(id="r-time", className="time"),
                        ], className="row"),
                        dcc.Slider(id="r-frame", min=0, max=1, step=1, value=0, marks=None, updatemode="drag"),
                    ], id="r-player-box", style={"display": "none"}),
                    html.Div(PLACEHOLDER, id="r-summary"),
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
    register_saves(app)
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
        Output("r-player-box", "style"),
        Output("r-dt", "value"),
        Output("r-steps", "value"),
        Output("r-evolve-status", "children"),
        Input("r-poll", "n_intervals"),
        State("store-run", "data"),
    )
    def poll(_, run_id):
        run = RUNS.get(run_id)
        nothing = (no_update,) * 11                     # the outputs after the first three
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
        elif not converged.all():
            note.append("Lighter arrows: the score chains didn't converge within the maximum number of iterations, "
                        "so values there depend on that maximum and on the initial scores.")
        composition = [field(f"{label} · {name}",
                             dcc.Input(id={"type": "r-share", "index": i}, type="text", inputMode="decimal",
                                       value="", className="num"))
                       for i, (label, name) in enumerate(zip(run.labels, run.names))]
        options = ([{"label": f"{l} · {n}", "value": n} for l, n in zip(run.labels, run.names)]
                   + [{"label": "population", "value": "population"}])
        values = [o["value"] for o in options]
        trajectory = run.trajectory                 # only a save made before evolutions were separate has one
        selection = None if trajectory is None else {"kind": "frame", "frame": 0}
        player = {"display": "none"} if trajectory is None else {}
        return (None, {}, True, fig, " ".join(note), composition, 1 if trajectory is None else len(trajectory.t) - 1,
                options, values, selection, player, f"{run.experiment.get('dt', spec.DEFAULT_ADVANCED['dt']):g}",
                f"{run.experiment.get('n_steps', spec.DEFAULT_ADVANCED['n_steps']):g}", "")

    @app.callback(
        Output("r-strategies", "children"),
        Output("r-norm", "children"),
        Input("r-poll", "n_intervals"),
        Input("r-surfaces", "value"),
        State("store-run", "data"),
        State("r-strategies", "children"),
    )
    def show_strategies(_, surfaces, run_id, shown):
        run = RUNS.get(run_id)
        if run is None or (ctx.triggered_id == "r-poll" and shown):     # drawn once, and again for the checkbox
            return no_update, no_update
        return strategy_cards(run, surfaces=bool(surfaces)), norm_view(run.experiment["norm"])

    @app.callback(
        Output("r-selection", "data", allow_duplicate=True),
        Output("r-composition-error", "children"),
        Output("r-player", "disabled", allow_duplicate=True),
        Output("r-play", "children", allow_duplicate=True),
        Output({"type": "r-share", "index": ALL}, "value"),
        Input("r-simplex", "clickData"),
        Input("r-show", "n_clicks"),
        Input("r-frame", "value"),
        State({"type": "r-share", "index": ALL}, "value"),
        State("r-player", "disabled"),
        prevent_initial_call=True,
    )
    def select(click, show, frame, shares, player_disabled):
        trigger = ctx.triggered_id
        keep = [no_update] * len(shares)             # a wildcard output takes a list, one value per share box
        if trigger == "r-frame":
            return {"kind": "frame", "frame": frame}, None, no_update, no_update, keep
        pause = (True, "▶ Play")
        if trigger == "r-show":
            rho, error = validation.shares(shares)
            if error:
                return no_update, status(error, False), no_update, no_update, keep
            return ({"kind": "point", "rho": rho.tolist()}, None) + pause + (keep,)
        if trigger == "r-simplex" and click:
            rho = click_composition(click)
            if rho is None:
                return no_update, None, no_update, no_update, keep
            return ({"kind": "point", "rho": rho}, None) + pause + ([f"{x:.4g}" for x in rho],)
        return no_update, None, no_update, no_update, keep

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
        if run is None or run.state != "done":
            return (no_update,) * 7
        if not selection:                            # nothing selected yet: empty panels and a hint
            return {}, {}, {}, PLACEHOLDER, "", no_update, ""
        x_range, y_range, range_error = axis_ranges(xmin, xmax, ymin, ymax)
        if selection["kind"] == "frame" and run.trajectory is not None:
            state = run.frame(min(selection["frame"], len(run.trajectory.t) - 1))
        else:
            state = run.point(selection["rho"] if selection["kind"] == "point" else run.frame(0)["rho"])
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
        Output("r-evolve", "disabled"),
        Input("r-selection", "data"),
        Input("r-evolve-poll", "disabled"),
        State("store-run", "data"),
    )
    def evolve_button(selection, poll_disabled, run_id):
        """Evolving needs a selected point, and only one evolution runs at a time."""
        run = RUNS.get(run_id)
        return not selection or run is None or run.evolve_state == "running"

    @app.callback(
        Output("r-evolve-poll", "disabled"),
        Output("r-evolve-status", "children", allow_duplicate=True),
        Input("r-evolve", "n_clicks"),
        State("r-selection", "data"),
        State("r-steps", "value"),
        State("r-dt", "value"),
        State("store-run", "data"),
        prevent_initial_call=True,
    )
    def evolve(n_clicks, selection, steps, dt, run_id):
        run = RUNS.get(run_id)
        if not n_clicks or run is None or run.state != "done":
            return no_update, no_update
        rho = start_point(run, selection)
        if rho is None:
            return no_update, status("Select a point first: click the simplex, or type its shares and press Show.",
                                     False)
        n_steps, error = validation.positive_int(steps, "Steps")
        if error:
            return no_update, status(error, False)
        dt_value, error = validation.non_negative(dt, "The step size")
        if error or dt_value == 0:
            return no_update, status(error or "The step size must be positive.", False)
        if not run.start_evolve(rho, n_steps, dt_value):
            return no_update, no_update
        return False, "Evolving…"

    @app.callback(
        Output("r-evolve-poll", "disabled", allow_duplicate=True),
        Output("r-evolve-status", "children", allow_duplicate=True),
        Output("r-simplex", "figure", allow_duplicate=True),
        Output("r-frame", "max"),
        Output("r-frame", "value", allow_duplicate=True),
        Output("r-player-box", "style", allow_duplicate=True),
        Output("r-selection", "data", allow_duplicate=True),
        Input("r-evolve-poll", "n_intervals"),
        State("store-run", "data"),
        prevent_initial_call=True,
    )
    def evolve_poll(_, run_id):
        run = RUNS.get(run_id)
        nothing = (no_update,) * 5
        if run is None or run.evolve_state == "idle":
            return (True, no_update) + nothing
        if run.evolve_state == "running":
            bar = html.Progress(value=f"{run.evolve_progress:.3f}", max="1")
            return (no_update, bar) + nothing
        if run.evolve_state == "error":
            return (True, status(f"The evolution failed: {run.evolve_error}", False)) + nothing
        t = run.trajectory
        message = f"Evolved {len(t.t) - 1} steps to t = {t.t[-1]:g}."
        if t.at_rest:
            message += " It reached a rest point."
        return (True, message, run.figure, len(t.t) - 1, 0, {}, {"kind": "frame", "frame": 0})

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


def register_saves(app):
    @app.callback(
        Output("r-save-status", "children"),
        Output("r-load-choice", "options"),
        Output("r-save-name", "value"),
        Output("r-load-choice", "value", allow_duplicate=True),
        Input("r-save", "n_clicks"),
        State("r-save-name", "value"),
        State("store-run", "data"),
        prevent_initial_call=True,
    )
    def save(n_clicks, name, run_id):
        if not n_clicks:
            return no_update, no_update, no_update, no_update
        try:
            path = saves.save_run(RUNS.get(run_id), name)
        except ValueError as error:
            return status(str(error), False), no_update, no_update, no_update
        return status(f"Saved as “{name.strip()}” ({path}).", True), save_options(), "", path.name

    @app.callback(
        Output("store-run", "data", allow_duplicate=True),
        Output("store-strategies", "data", allow_duplicate=True),
        Output("store-params", "data", allow_duplicate=True),
        Output("r-save-status", "children", allow_duplicate=True),
        Output("r-poll", "disabled", allow_duplicate=True),
        Output("r-strategies", "children", allow_duplicate=True),
        Input("r-load", "n_clicks"),
        State("r-load-choice", "value"),
        prevent_initial_call=True,
    )
    def load(n_clicks, choice):
        keep = (no_update,) * 3
        if not n_clicks:
            return keep + (no_update,) * 3
        if not choice:
            return keep + (status("Choose a saved result to load.", False), no_update, no_update)
        try:
            run = saves.load_run(saves.default_folder() / Path(choice).name)
        except ValueError as error:
            return keep + (status(str(error), False), no_update, no_update)
        experiment = run.experiment
        message = status("Loaded. The parameters page now shows its settings.", True)
        return (RUNS.add(run), experiment["strategies"], params_from_experiment(experiment), message,
                False, None)                                # polling again draws the loaded run

    @app.callback(
        Output("r-confirm-delete", "displayed"),
        Output("r-confirm-delete", "message"),
        Output("r-save-status", "children", allow_duplicate=True),
        Input("r-delete", "n_clicks"),
        State("r-load-choice", "value"),
        prevent_initial_call=True,
    )
    def ask_delete(n_clicks, choice):
        if not n_clicks:
            return no_update, no_update, no_update
        if not choice:
            return False, no_update, status("Choose a saved result to delete.", False)
        return True, f"Delete the saved result “{Path(choice).stem}”? This can't be undone.", None

    @app.callback(
        Output("r-save-status", "children", allow_duplicate=True),
        Output("r-load-choice", "options", allow_duplicate=True),
        Output("r-load-choice", "value"),
        Input("r-confirm-delete", "submit_n_clicks"),
        State("r-load-choice", "value"),
        prevent_initial_call=True,
    )
    def delete(submitted, choice):
        if not submitted or not choice:
            return no_update, no_update, no_update
        try:
            saves.delete_save(choice)
        except ValueError as error:
            return status(str(error), False), save_options(), None
        return status("Deleted.", True), save_options(), None


    @app.callback(
        Output("r-save-status", "children", allow_duplicate=True),
        Output("r-load-choice", "options", allow_duplicate=True),
        Output("r-load-choice", "value", allow_duplicate=True),
        Output("r-rename-name", "value"),
        Input("r-rename", "n_clicks"),
        State("r-load-choice", "value"),
        State("r-rename-name", "value"),
        prevent_initial_call=True,
    )
    def rename(n_clicks, choice, new_name):
        if not n_clicks:
            return (no_update,) * 4
        if not choice:
            return status("Choose a saved result to rename.", False), no_update, no_update, no_update
        try:
            file = saves.rename_save(choice, new_name)
        except ValueError as error:
            return status(str(error), False), no_update, no_update, no_update
        return status(f"Renamed to “{new_name.strip()}”.", True), save_options(), file, ""

    @app.callback(
        Output("r-save-download", "data"),
        Output("r-save-status", "children", allow_duplicate=True),
        Input("r-download-file", "n_clicks"),
        Input("r-download-csv", "n_clicks"),
        State("r-load-choice", "value"),
        prevent_initial_call=True,
    )
    def download_save(file_clicks, csv_clicks, choice):
        if not ctx.triggered[0]["value"]:
            return no_update, no_update
        if not choice:
            return no_update, status("Choose a saved result to download.", False)
        try:
            if ctx.triggered_id == "r-download-csv":
                return dcc.send_bytes(saves.export_csv(choice), f"{Path(choice).stem}_csv.zip"), None
            return dcc.send_bytes(saves.read_file(choice), Path(choice).name), None
        except ValueError as error:
            return no_update, status(str(error), False)

    @app.callback(
        Output("r-save-status", "children", allow_duplicate=True),
        Output("r-load-choice", "options", allow_duplicate=True),
        Output("r-load-choice", "value", allow_duplicate=True),
        Input("r-upload", "contents"),
        prevent_initial_call=True,
    )
    def upload_save(contents):
        if not contents:
            return no_update, no_update, no_update
        try:
            content = base64.b64decode(contents.split(",", 1)[1])
            name = saves.import_file(content)
        except ValueError as error:
            return status(str(error), False), no_update, no_update
        return status("Added to the saved results. Press Load to open it.", True), save_options(), name


def start_point(run, selection):
    """The shares of the selected point (for a step of an evolution, the shares at that step), or None."""
    if not selection:
        return None
    if selection["kind"] == "point":
        return selection["rho"]
    if run.trajectory is None:
        return None
    return run.frame(min(selection["frame"], len(run.trajectory.t) - 1))["rho"].tolist()


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


ACTIONS = ["declines", "gives"]
CHANGE = {"+": ("+1", "up"), "0": ("0", "same"), "-": ("\u22121", "down")}


def norm_view(norm):
    """The scoring norm of a run: its barcode, how to read it, and each rule as a table of score changes."""
    n = 3 if norm["type"] == "ternary" else 2
    labels = spec.class_labels(norm["type"], norm.get("tau"))
    barcode = f"CN:{norm['CN']} NC:{norm['NC']}"
    classes = ", ".join(f"{k} = score {label}" for k, label in enumerate(labels))
    tables = []
    for rule, title in (("CN", "Donor's score change (CN)"), ("NC", "Receiver's score change (NC)")):
        code = norm[rule]
        header = html.Tr([html.Th("receiver, action ↓  donor →", className="corner")]
                         + [html.Th(f"score {label}") for label in labels])
        rows = []
        for need in range(n):
            for act in range(2):
                cells = []
                for can in range(n):
                    text, kind = CHANGE[code[need * 2 * n + act * n + can]]
                    cells.append(html.Td(text, className=f"change {kind}"))
                rows.append(html.Tr([html.Th(f"score {labels[need]}, {ACTIONS[act]}")] + cells))
        tables.append(html.Div([html.Label(title), html.Table([header] + rows, className="grid norm-view")]))
    convention = [
        html.Div([html.Code(barcode, className="barcode-text"),
                  html.Span(f"  τ = {norm['tau']}" if n == 3 else "", className="hint")]),
        html.Div(
            f"Reading the barcode: CN and NC are each {2 * n * n} symbols, one for every case, in the order receiver's "
            f"class, then whether the donor declined or gave, then donor's class (the donor's class changes fastest). "
            f"CN is the change to the donor's score, NC the change to the receiver's: + adds one, − takes one "
            f"(a player with score 0 can't lose) and 0 leaves it. Classes: {classes}. "
            f"Noise ε adds its own ±1 moves on top.", className="hint"),
    ]
    return convention + [html.Div(tables, className="two-col tight")]


def strategy_cards(run, surfaces=False):
    """One card per strategy of the run: its label and colour, name and what it does."""
    cards = []
    for i, cfg in enumerate(run.experiment["strategies"]):
        color = figures.COLORS[i]
        children = [
            html.Div([html.Span(run.labels[i], className="chip", style={"background": color}), html.B(cfg["name"])],
                     className="strategy-head"),
            html.Div(spec.describe_strategy(cfg), className="hint"),
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
    """Settings, the last evolution, vector field, rest points and the selected point, as CSV/JSON in a zip."""
    names = run.labels                     # column names S1, S2, S3; settings.json has the strategies
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("settings.json", json.dumps(run.experiment, indent=2))
        t = run.trajectory
        if t is not None:                  # only after an evolution from a point
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
            state = (run.frame(selection["frame"]) if selection["kind"] == "frame" and run.trajectory is not None
                     else run.point(selection["rho"] if selection["kind"] == "point" else run.frame(0)["rho"]))
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
