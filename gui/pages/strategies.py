"""Page 1: choose 2 or 3 strategies."""

import base64
import json

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from .. import figures, spec, validation
from ..components import GRAPH_CONFIG, card, field

TYPES = [
    {"label": "Binary (score 0 / score ≥ 1)", "value": "binary"},
    {"label": "Ternary (score 0 / 1 to τ / above τ)", "value": "ternary"},
    {"label": "φ-Disc", "value": "phi"},
]


def layout():
    return html.Div(
        [
            html.H1("Choose strategies"),
            html.P("Pick 2 or 3.", className="lead"),
            html.Div(
                [
                    card(
                        "New strategy",
                        [
                            field("Type", dcc.Dropdown(id="s-type", options=TYPES, value="binary", clearable=False)),
                            field("Preset", dcc.Dropdown(id="s-preset", options=list(spec.STRATEGY_PRESETS),
                                                         placeholder="none")),
                            field("Name", dcc.Input(id="s-name", type="text", placeholder="optional",
                                                    debounce=False)),
                            html.Div(field("τ", dcc.Input(id="s-tau", type="text", value="2", inputMode="numeric", className="num")),
                                     id="s-tau-row", style={"display": "none"}),
                            html.Div(
                                field("φ", dcc.Input(id="s-phi", type="text", value="1", inputMode="decimal", className="num"),
                                      hint="Never helps when its own score is 0; otherwise helps a receiver with a "
                                           "score with probability j^(−φ), j being its own score."),
                                id="s-phi-row", style={"display": "none"}),
                            html.Div(id="s-grid"),
                            html.Div(id="s-error", className="status error"),
                            html.Button("Add strategy", id="s-add", className="primary", n_clicks=0),
                        ],
                    ),
                    card(
                        "Preview",
                        [
                            dcc.Graph(id="s-preview", config=GRAPH_CONFIG),
                            field("Scores shown", dcc.Input(id="s-range", type="text", value="10", inputMode="numeric", className="num"),
                                  className="field inline"),
                        ],
                    ),
                ],
                className="two-col",
            ),
            card("Selected", html.Div(id="s-selected", className="selected")),
            dcc.Store(id="s-current"),
            html.Div(
                [
                    dcc.Upload(html.Button("Load settings…", className="secondary"), id="s-upload",
                               accept=".json,application/json"),
                    html.Div(id="s-upload-error", className="status error"),
                    html.Button("Next →", id="s-next", className="primary", n_clicks=0, disabled=True),
                ],
                className="actions",
            ),
        ]
    )


def grid(kind, P, tau):
    labels = spec.class_labels(kind, tau)
    header = html.Tr([html.Th("receiver ↓  donor →", className="corner")] + [html.Th(f"score {c}") for c in labels])
    rows = [
        html.Tr(
            [html.Th(f"score {labels[r]}")]
            + [
                html.Td(dcc.Input(id={"type": "s-p", "r": r, "c": c}, type="text", inputMode="decimal",
                                  value=f"{P[r][c]:g}", className="cell"))
                for c in range(len(labels))
            ]
        )
        for r in range(len(labels))
    ]
    return html.Div([html.Label("p(help)"), html.Table([header] + rows, className="grid")])


def grid_values(entries, n):
    """The n x n grid from pattern-matching entries [{"id": {"r", "c"}, "value"}], placed by id
    (Dash does not keep them in row order), or None if the grid shown is not n x n."""
    cells = {(e["id"]["r"], e["id"]["c"]): e.get("value") for e in entries}
    if set(cells) != {(r, c) for r in range(n) for c in range(n)}:
        return None
    return [[cells[(r, c)] for c in range(n)] for r in range(n)]


def register(app):
    @app.callback(
        Output("s-grid", "children"),
        Output("s-tau-row", "style"),
        Output("s-phi-row", "style"),
        Output("s-type", "value"),
        Output("s-name", "value"),
        Output("s-preset", "value"),
        Input("s-type", "value"),
        Input("s-preset", "value"),
        Input("s-tau", "value"),
        State({"type": "s-p", "r": ALL, "c": ALL}, "value"),
        State("s-name", "value"),
    )
    def show_inputs(kind, preset, tau, values, name):
        trigger = ctx.triggered_id
        if trigger == "s-preset" and preset:
            cfg = spec.preset_strategy(preset)
            kind, P, name = "binary", cfg["P"], preset
        else:
            n = 3 if kind == "ternary" else 2
            current = grid_values(ctx.states_list[0], n)
            P = [[validation.parse_number(x) or 0.0 for x in row] for row in current] if current else spec.default_P(kind)
            if trigger == "s-type":
                preset = None
                if name in spec.STRATEGY_PRESETS:
                    name = ""
        tau_value, _ = validation.positive_int(tau, "τ")
        hide, show = {"display": "none"}, {}
        grid_children = grid(kind, P, tau_value or "τ") if kind != "phi" else None
        return (grid_children, show if kind == "ternary" else hide, show if kind == "phi" else hide, kind, name,
                preset)

    @app.callback(
        Output("s-preview", "figure"),
        Output("s-error", "children"),
        Output("s-current", "data"),
        Input("s-type", "value"),
        Input({"type": "s-p", "r": ALL, "c": ALL}, "value"),
        Input("s-tau", "value"),
        Input("s-phi", "value"),
        Input("s-name", "value"),
        Input("s-range", "value"),
    )
    def preview(kind, values, tau, phi, name, max_score):
        matrix = grid_values(ctx.inputs_list[1], 3 if kind == "ternary" else 2)
        cfg, error = current_strategy(kind, matrix, tau, phi, name)
        max_score, _ = validation.positive_int(max_score, "Scores shown")
        if error:
            return no_update, error, None
        strategy = spec.build_strategy(cfg)
        return figures.strategy_heatmap(strategy, min(max_score or 10, 60), height=440, font_size=13,
                                        annotate_up_to=14), "", cfg

    @app.callback(
        Output("store-strategies", "data"),
        Input("s-add", "n_clicks"),
        Input({"type": "s-remove", "index": ALL}, "n_clicks"),
        State("store-strategies", "data"),
        State("s-current", "data"),
        prevent_initial_call=True,
    )
    def change_selection(add_clicks, remove_clicks, selected, current):
        selected = list(selected or [])
        trigger = ctx.triggered_id
        if trigger == "s-add":
            if not add_clicks or current is None or len(selected) >= spec.MAX_STRATEGIES:
                return no_update
            cfg = dict(current)
            preset = cfg["name"] if cfg["name"] in spec.STRATEGY_PRESETS else None
            if preset and (cfg["type"] != "binary" or cfg["P"] != spec.preset_strategy(preset)["P"]):
                cfg["name"] = ""            # edited after choosing the preset: it is no longer that strategy
            cfg["name"] = unique_name(cfg["name"] or f"Strategy {len(selected) + 1}", [s["name"] for s in selected])
            return selected + [cfg]
        if isinstance(trigger, dict) and trigger.get("type") == "s-remove":
            if not ctx.triggered[0]["value"]:
                return no_update
            return [s for i, s in enumerate(selected) if i != trigger["index"]]
        return no_update

    @app.callback(
        Output("s-selected", "children"),
        Output("s-next", "disabled"),
        Output("s-add", "disabled"),
        Input("store-strategies", "data"),
        Input("s-current", "data"),
    )
    def show_selection(selected, current):
        selected = selected or []
        if not selected:
            items = html.P("None yet.", className="hint")
        else:
            items = [
                html.Div(
                    [
                        html.Div([html.B(s["name"]), html.Span(spec.describe_strategy(s), className="hint")],
                                 className="selected-title"),
                        dcc.Graph(figure=small_surface(s), config={"displayModeBar": False}, className="mini"),
                        html.Button("Remove", id={"type": "s-remove", "index": i}, className="link", n_clicks=0),
                    ],
                    className="selected-item",
                )
                for i, s in enumerate(selected)
            ]
        full = len(selected) >= spec.MAX_STRATEGIES
        return items, not 2 <= len(selected) <= spec.MAX_STRATEGIES, full or current is None

    @app.callback(Output("url", "pathname", allow_duplicate=True), Input("s-next", "n_clicks"),
                  prevent_initial_call=True)
    def next_page(n_clicks):
        return "/parameters" if n_clicks else no_update

    @app.callback(
        Output("store-strategies", "data", allow_duplicate=True),
        Output("store-params", "data", allow_duplicate=True),
        Output("url", "pathname", allow_duplicate=True),
        Output("s-upload-error", "children"),
        Input("s-upload", "contents"),
        prevent_initial_call=True,
    )
    def load_settings(contents):
        if not contents:
            return no_update, no_update, no_update, ""
        try:
            experiment = json.loads(base64.b64decode(contents.split(",", 1)[1]))
        except Exception:  # noqa: BLE001
            return no_update, no_update, no_update, "That file isn't a settings file."
        error = spec.check(experiment)
        if error:
            return no_update, no_update, no_update, error
        try:
            params = params_from_experiment(experiment)
        except Exception as error:  # noqa: BLE001
            return no_update, no_update, no_update, f"These settings can't be used: {error}"
        return experiment["strategies"], params, "/parameters", ""


def current_strategy(kind, matrix, tau, phi, name):
    """Strategy cfg from the editor (matrix: the p(help) grid as typed), or an error message."""
    name = (name or "").strip()
    if kind == "phi":
        phi_value, error = validation.non_negative(phi, "φ")
        if error:
            return None, error
        return {"type": "phi", "name": name or f"φ-Disc (φ={phi_value:g})", "P": None, "tau": None,
                "phi": phi_value}, ""
    n = 3 if kind == "ternary" else 2
    tau_value = None
    if kind == "ternary":
        tau_value, error = validation.positive_int(tau, "τ")
        if error:
            return None, error
    if matrix is None:
        return None, ""
    P = []
    for row in matrix:
        out = []
        for x in row:
            p, error = validation.probability(x)
            if error:
                return None, error
            out.append(p)
        P.append(out)
    return {"type": kind, "name": name, "P": P, "tau": tau_value, "phi": None}, ""


def unique_name(name, taken):
    if name not in taken:
        return name
    k = 2
    while f"{name} ({k})" in taken:
        k += 1
    return f"{name} ({k})"


def small_surface(cfg):
    """The figure on a strategy card: p(help) as a heat map over receiver and donor score."""
    return figures.strategy_heatmap(spec.build_strategy(cfg), max_score=6)


def merged_scores(entries):
    """Score entries with the masses of repeated scores added up, sorted by score."""
    masses = {}
    for e in entries:
        score = int(validation.parse_number(e["score"]))
        masses[score] = masses.get(score, 0.0) + validation.parse_number(e["mass"])
    return [{"score": score, "mass": mass} for score, mass in sorted(masses.items())]


def params_from_experiment(experiment):
    """The parameters page's stored inputs, from a loaded settings file."""
    fmt = spec.fmt
    number = validation.parse_number
    return {
        "scores": merged_scores(experiment["scores"]),
        "norm": experiment["norm"],
        "noise": {k: fmt(number(experiment["noise"][k])) for k in ("eps", "eps_up", "eps_down")},
        "interactions": {"p_can": fmt(number(experiment.get("p_can", 0.5))), "benefit": fmt(number(experiment["benefit"])),
                         "cost": fmt(number(experiment["cost"]))},
        "advanced": {**{k: fmt(number(experiment[k])) for k in ("max_iter", "tol", "resolution")},
                     "full_steps": bool(experiment.get("full_steps", False))},
    }
