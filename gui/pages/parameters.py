"""Page 2: initial shares, initial scores, scoring norm, noise and the rest."""

import json

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from .. import spec, validation
from ..compute import RUNS
from ..components import card, field, status

ACTS = ["declines", "gives"]
SYMBOLS = [("−", "-"), ("0", "0"), ("+", "+")]

# Inputs whose values make up an experiment, in the order experiment_from_inputs takes them.
FORM = [
    ({"type": "p-share", "index": ALL}, "value"),
    ("p-scores", "data"),
    ("p-norm-type", "value"),
    ("p-norm-tau", "value"),
    ({"type": "p-bit", "rule": "CN", "index": ALL}, "value"),
    ({"type": "p-bit", "rule": "NC", "index": ALL}, "value"),
    ("p-eps", "value"),
    ("p-eps-up", "value"),
    ("p-eps-down", "value"),
    ("p-q", "value"),
    ("p-benefit", "value"),
    ("p-cost", "value"),
    ("p-dt", "value"),
    ("p-steps", "value"),
    ("p-maxiter", "value"),
    ("p-tol", "value"),
    ("p-resolution", "value"),
    ("p-full", "value"),
]


def layout():
    return html.Div([html.H1("Parameters"), dcc.Store(id="p-scores"), html.Div(id="p-body")])


def actions():
    return html.Div(
        [
            html.Button("← Strategies", id="p-back", className="secondary", n_clicks=0),
            html.Button("Save settings", id="p-save", className="secondary", n_clicks=0),
            dcc.Download(id="p-download"),
            html.Div(id="p-run-error", className="status error"),
            html.Button("Run", id="p-run", className="primary", n_clicks=0),
        ],
        className="actions",
    )


def defaults(n):
    share = f"{1 / n:.4g}"
    return {
        "shares": [share] * n,
        "scores": [{"score": 0, "mass": 0.5}, {"score": 1, "mass": 0.5}],
        "norm": spec.norm_from_barcode(list(spec.NORM_PRESETS.values())[0], "binary"),
        "noise": {"eps": "0.1", "eps_up": "0.5", "eps_down": "0.5"},
        "interactions": {"p_can": "0.5", "benefit": f"{spec.DEFAULT_ADVANCED['benefit']:g}",
                         "cost": f"{spec.DEFAULT_ADVANCED['cost']:g}"},
        "advanced": {**{k: f"{spec.DEFAULT_ADVANCED[k]:g}" for k in ("dt", "n_steps", "max_iter", "tol", "resolution")},
                     "full_steps": spec.DEFAULT_ADVANCED["full_steps"]},
    }


def text_input(id_, value, mode="decimal", width=None):
    style = {"width": width} if width else None
    return dcc.Input(id=id_, type="text", value=value, inputMode=mode, className="num", style=style)


def form(strategies, params):
    n = len(strategies)
    base = defaults(n)
    params = {**base, **(params or {})}
    if len(params["shares"]) != n:
        params["shares"] = base["shares"]
    norm = usable_norm(params["norm"], base["norm"])
    noise, inter, adv = params["noise"], params["interactions"], params["advanced"]
    return [
        html.Div(
            [
                card("Initial population shares", [
                    html.Div([field(s["name"], text_input({"type": "p-share", "index": i}, params["shares"][i]))
                              for i, s in enumerate(strategies)], className="row"),
                    html.Div(id="p-shares-status"),
                ]),
                card("Noise", [
                    html.Div([
                        field("ε", text_input("p-eps", noise["eps"])),
                        field("ε↑", text_input("p-eps-up", noise["eps_up"])),
                        field("ε↓", text_input("p-eps-down", noise["eps_down"])),
                    ], className="row"),
                    html.Div(id="p-noise-status"),
                ]),
            ],
            className="two-col",
        ),
        html.Div(
            [
                card("Initial score distribution", [
                    html.Div([
                        field("Score", text_input("p-score", "", mode="numeric")),
                        field("Mass", text_input("p-mass", "")),
                        html.Button("Add", id="p-add-score", className="secondary", n_clicks=0),
                        html.Button("Clear", id="p-clear-scores", className="link", n_clicks=0),
                    ], className="row bottom"),
                    html.Div(id="p-score-error", className="status error"),
                    html.Div(id="p-score-table"),
                ]),
                card("Interactions", [
                    html.Div([
                        field("q", text_input("p-q", inter["p_can"]),
                              hint="Probability an individual is able to help; a donor meets a receiver with "
                                   "probability q(1−q) for each order."),
                        field("Benefit b", text_input("p-benefit", inter["benefit"])),
                        field("Cost c", text_input("p-cost", inter["cost"])),
                    ], className="row"),
                    html.Div(id="p-inter-status"),
                    html.Details([
                        html.Summary("Advanced"),
                        html.Div([
                            field("Step size dt", text_input("p-dt", adv["dt"])),
                            field("Steps", text_input("p-steps", adv["n_steps"], mode="numeric")),
                            field("Max score iterations", text_input("p-maxiter", adv["max_iter"], mode="numeric")),
                            field("Score tolerance", text_input("p-tol", adv["tol"])),
                            field("Arrow grid", text_input("p-resolution", adv["resolution"], mode="numeric"),
                                  hint="Points per side of the simplex."),
                        ], className="row wrap"),
                        dcc.Checklist(id="p-full", className="checks",
                                      options=[{"label": "Run every score chain for all the iterations",
                                                "value": "full"}],
                                      value=["full"] if adv.get("full_steps", True) else []),
                    ], className="advanced"),
                ]),
            ],
            className="two-col",
        ),
        card("Scoring norm", [
            html.Div([
                field("Type", dcc.RadioItems(id="p-norm-type", value=norm["type"], inline=True,
                                             options=[{"label": "Binary", "value": "binary"},
                                                      {"label": "Ternary", "value": "ternary"}])),
                html.Div(field("τ", text_input("p-norm-tau", str(norm.get("tau") or 2), mode="numeric")),
                         id="p-norm-tau-row", style={} if norm["type"] == "ternary" else {"display": "none"}),
                field("Preset", dcc.Dropdown(id="p-norm-preset", options=list(spec.NORM_PRESETS),
                                             placeholder="none"), className="field wide"),
            ], className="row bottom"),
            html.Div(norm_grids(norm), id="p-norm-grid"),
            html.Div([
                field("Barcode", dcc.Input(id="p-barcode", type="text", className="barcode")),
                html.Button("Apply", id="p-apply", className="secondary", n_clicks=0),
            ], className="row bottom"),
            html.Div(id="p-norm-status"),
        ]),
    ]


def usable_norm(norm, fallback):
    """A stored norm cfg that can be drawn, with tau 2 if a ternary norm has none; fallback if invalid."""
    try:
        norm = dict(norm)
        if norm["type"] == "ternary" and not norm.get("tau"):
            norm["tau"] = 2
        spec.build_norm(norm)
        return norm
    except Exception:  # noqa: BLE001
        return fallback


def norm_grids(norm):
    n = 3 if norm["type"] == "ternary" else 2
    labels = spec.class_labels(norm["type"], norm.get("tau"))
    tables = []
    for rule, title in (("CN", "Donor's score change"), ("NC", "Receiver's score change")):
        code = norm[rule]
        header = html.Tr([html.Th("", className="corner")] + [html.Th(f"donor {c}") for c in labels])
        rows = []
        for need in range(n):
            for act in range(2):
                cells = []
                for can in range(n):
                    k = need * 2 * n + act * n + can
                    affected_has_zero = can == 0 if rule == "CN" else need == 0
                    options = [{"label": lab, "value": val, "disabled": val == "-" and affected_has_zero}
                               for lab, val in SYMBOLS]
                    cells.append(html.Td(dcc.Dropdown(id={"type": "p-bit", "rule": rule, "index": k}, options=options,
                                                      value=code[k], clearable=False, searchable=False,
                                                      className="bit")))
                rows.append(html.Tr([html.Th(f"receiver {labels[need]}, {ACTS[act]}")] + cells))
        tables.append(html.Div([html.Label(f"{title} ({rule})"), html.Table([header] + rows, className="grid norm")]))
    return html.Div(tables, className="two-col tight")


def register(app):
    @app.callback(
        Output("p-body", "children"),
        Output("p-scores", "data"),
        Input("url", "pathname"),
        State("store-strategies", "data"),
        State("store-params", "data"),
    )
    def build(pathname, strategies, params):
        if pathname != "/parameters":
            return no_update, no_update
        if not strategies or len(strategies) < 2:
            return html.P(["Choose strategies first. ", dcc.Link("Back to strategies", href="/")]), []
        params = params or defaults(len(strategies))
        return form(strategies, params) + [actions()], params.get("scores", defaults(2)["scores"])

    @app.callback(
        Output("store-params", "data", allow_duplicate=True),
        *[Input(i, p) for i, p in FORM],
        prevent_initial_call=True,
    )
    def remember(shares, scores, norm_type, tau, cn, nc, eps, eps_up, eps_down, q, b, c, dt, steps, maxiter, tol, res,
                 full):
        if not shares:
            return no_update
        bits = "".join(cn or [])
        return {
            "shares": shares,
            "scores": scores or [],
            "norm": {"type": "ternary" if len(bits) == 18 else "binary", "tau": validation.positive_int(tau, "τ")[0],
                     "CN": bits, "NC": "".join(nc or [])},
            "noise": {"eps": eps, "eps_up": eps_up, "eps_down": eps_down},
            "interactions": {"p_can": q, "benefit": b, "cost": c},
            "advanced": {"dt": dt, "n_steps": steps, "max_iter": maxiter, "tol": tol, "resolution": res,
                         "full_steps": bool(full)},
        }

    @app.callback(Output("p-shares-status", "children"), Input({"type": "p-share", "index": ALL}, "value"))
    def check_shares(values):
        shares, error = validation.shares(values)
        return status(error, False) if error else status("Adds up to 1.", True)

    @app.callback(Output("p-noise-status", "children"),
                  Input("p-eps", "value"), Input("p-eps-up", "value"), Input("p-eps-down", "value"))
    def check_noise(eps, up, down):
        _, error = validation.noise(eps, up, down)
        return status(error, False) if error else None

    @app.callback(Output("p-inter-status", "children"),
                  Input("p-q", "value"), Input("p-benefit", "value"), Input("p-cost", "value"))
    def check_interactions(q, b, c):
        _, error = interactions(q, b, c)
        return status(error, False) if error else None

    @app.callback(
        Output("p-scores", "data", allow_duplicate=True),
        Output("p-score-error", "children"),
        Output("p-score", "value"),
        Output("p-mass", "value"),
        Input("p-add-score", "n_clicks"),
        Input("p-clear-scores", "n_clicks"),
        Input({"type": "p-score-remove", "index": ALL}, "n_clicks"),
        State("p-score", "value"),
        State("p-mass", "value"),
        State("p-scores", "data"),
        prevent_initial_call=True,
    )
    def edit_scores(add, clear, removes, score, mass, entries):
        entries = list(entries or [])
        trigger = ctx.triggered_id
        if not ctx.triggered[0]["value"]:
            return no_update, no_update, no_update, no_update
        if trigger == "p-clear-scores":
            return [], "", no_update, no_update
        if isinstance(trigger, dict):
            return [e for e in entries if e["score"] != trigger["index"]], "", no_update, no_update
        entry, error = validation.score_entry(score, mass)
        if error:
            return no_update, error, no_update, no_update
        entries = [e for e in entries if e["score"] != entry["score"]] + [entry]
        return sorted(entries, key=lambda e: e["score"]), "", "", ""

    @app.callback(Output("p-score-table", "children"), Input("p-scores", "data"))
    def show_scores(entries):
        entries = entries or []
        w, error = validation.score_distribution(entries)
        rows = [
            html.Tr([html.Td(e["score"]), html.Td(f"{e['mass']:g}"),
                     html.Td(html.Button("×", id={"type": "p-score-remove", "index": e["score"]}, className="link",
                                         n_clicks=0))])
            for e in entries
        ]
        table = html.Table([html.Tr([html.Th("score"), html.Th("mass"), html.Th("")])] + rows, className="list") \
            if rows else None
        return [table, status(error, False) if error else status("Adds up to 1.", True)]

    @app.callback(
        Output("p-norm-grid", "children"),
        Output("p-norm-status", "children"),
        Output("p-norm-type", "value"),
        Output("p-norm-tau-row", "style"),
        Output("p-norm-preset", "value"),
        Output("p-norm-tau", "value"),
        Input("p-norm-type", "value"),
        Input("p-norm-tau", "value"),
        Input("p-norm-preset", "value"),
        Input("p-apply", "n_clicks"),
        State({"type": "p-bit", "rule": "CN", "index": ALL}, "value"),
        State({"type": "p-bit", "rule": "NC", "index": ALL}, "value"),
        State("p-barcode", "value"),
        prevent_initial_call=True,
    )
    def edit_norm(kind, tau, preset, apply, cn, nc, barcode):
        """The grid's bits are the norm; the radio and tau follow them, so they can't disagree."""
        trigger = ctx.triggered_id
        grid_type = "ternary" if len(cn) == 18 else "binary"
        tau_value, tau_error = validation.positive_int(tau, "τ")
        show, hide = {}, {"display": "none"}
        keep = no_update
        try:
            if trigger == "p-norm-preset":
                if not preset:
                    return keep, None, keep, keep, keep, keep
                norm = spec.norm_from_barcode(spec.NORM_PRESETS[preset], "binary")
                return norm_grids(norm), None, "binary", hide, preset, keep
            if trigger == "p-apply":
                if kind == "ternary" and tau_error:
                    return keep, status(tau_error, False), keep, keep, keep, keep
                norm = spec.norm_from_barcode(barcode or "", kind, tau_value)
                return norm_grids(norm), status("Applied.", True), keep, keep, None, keep
            if trigger == "p-norm-type":
                if kind == grid_type:
                    return keep, None, keep, show if kind == "ternary" else hide, keep, keep
                if kind == "ternary":
                    tau_value = tau_value or 2
                    current = {"type": "binary", "tau": None, "CN": "".join(cn), "NC": "".join(nc)}
                    lifted = spec.build_norm(current).to_ternary(tau_value)
                    norm = spec.norm_from_barcode(lifted.barcode, "ternary", tau_value)
                    return norm_grids(norm), None, keep, show, None, str(tau_value)
                return norm_grids(spec.default_norm("binary")), None, keep, hide, None, keep
            # tau changed: the same bits with new labels.
            if grid_type != "ternary":
                return keep, None, keep, keep, keep, keep
            if tau_error:
                return keep, status(tau_error, False), keep, keep, keep, keep
            norm = {"type": "ternary", "tau": tau_value, "CN": "".join(cn), "NC": "".join(nc)}
            return norm_grids(norm), None, keep, keep, keep, keep
        except (ValueError, KeyError) as error:
            return keep, status(f"Not a valid barcode: {error}", False), keep, keep, keep, keep

    @app.callback(
        Output("p-barcode", "value"),
        Input({"type": "p-bit", "rule": "CN", "index": ALL}, "value"),
        Input({"type": "p-bit", "rule": "NC", "index": ALL}, "value"),
    )
    def show_barcode(cn, nc):
        if not cn:
            return no_update
        return f"CN:{''.join(cn)} NC:{''.join(nc)}"

    @app.callback(Output("url", "pathname", allow_duplicate=True), Input("p-back", "n_clicks"),
                  prevent_initial_call=True)
    def back(n_clicks):
        return "/" if n_clicks else no_update

    @app.callback(
        Output("store-run", "data"),
        Output("url", "pathname", allow_duplicate=True),
        Output("p-run-error", "children"),
        Input("p-run", "n_clicks"),
        State("store-strategies", "data"),
        *[State(i, p) for i, p in FORM],
        prevent_initial_call=True,
    )
    def run(n_clicks, strategies, *values):
        if not n_clicks:
            return no_update, no_update, ""
        experiment, error = experiment_from_inputs(strategies, *values)
        if error:
            return no_update, no_update, error
        return RUNS.start(experiment), "/results", ""

    @app.callback(
        Output("p-download", "data"),
        Output("p-run-error", "children", allow_duplicate=True),
        Input("p-save", "n_clicks"),
        State("store-strategies", "data"),
        *[State(i, p) for i, p in FORM],
        prevent_initial_call=True,
    )
    def save(n_clicks, strategies, *values):
        if not n_clicks:
            return no_update, ""
        experiment, error = experiment_from_inputs(strategies, *values)
        if error:
            return no_update, error
        return dict(content=json.dumps(experiment, indent=2), filename="replicator_settings.json"), ""


def interactions(q, b, c):
    q_value = validation.parse_number(q)
    if q_value is None or not 0 < q_value < 1:
        return None, "q must lie strictly between 0 and 1."
    b_value, error = validation.non_negative(b, "The benefit")
    if error:
        return None, error
    c_value, error = validation.non_negative(c, "The cost")
    if error:
        return None, error
    return {"p_can": q_value, "benefit": b_value, "cost": c_value}, ""


def experiment_from_inputs(strategies, shares, scores, norm_type, tau, cn, nc, eps, eps_up, eps_down, q, b, c,
                           dt, steps, maxiter, tol, resolution, full=("full",)):
    """A validated experiment dict, or the first error message."""
    if not strategies or not 2 <= len(strategies) <= spec.MAX_STRATEGIES:
        return None, "Choose 2 or 3 strategies first."
    rho, error = validation.shares(shares)
    if error:
        return None, f"Shares: {error}"
    w, error = validation.score_distribution(scores or [])
    if error:
        return None, f"Score distribution: {error}"
    noise, error = validation.noise(eps, eps_up, eps_down)
    if error:
        return None, f"Noise: {error}"
    inter, error = interactions(q, b, c)
    if error:
        return None, error
    norm_type = "ternary" if len(cn) == 18 else "binary"     # the grid's bits decide the type
    tau_value = None
    if norm_type == "ternary":
        tau_value, error = validation.positive_int(tau, "τ")
        if error:
            return None, f"Norm: {error}"
    norm = {"type": norm_type, "tau": tau_value, "CN": "".join(cn), "NC": "".join(nc)}
    try:
        spec.build_norm(norm)
    except ValueError as error:
        return None, f"Norm: {error}"
    dt_value, error = validation.non_negative(dt, "The step size")
    if error or dt_value == 0:
        return None, error or "The step size must be positive."
    tol_value, error = validation.non_negative(tol, "The tolerance")
    if error:
        return None, error
    numbers = {}
    for key, value, what in (("n_steps", steps, "Steps"), ("max_iter", maxiter, "Max score iterations"),
                             ("resolution", resolution, "The arrow grid")):
        numbers[key], error = validation.positive_int(value, what)
        if error:
            return None, error
    total = sum(float(e["mass"]) for e in scores)
    experiment = {
        "version": spec.VERSION,
        "strategies": strategies,
        "shares": rho.tolist(),
        "scores": [{"score": int(e["score"]), "mass": float(e["mass"]) / total} for e in scores],
        "norm": norm,
        "noise": noise,
        **inter,
        "dt": dt_value,
        "tol": tol_value,
        **numbers,
        "full_steps": bool(full),
    }
    return experiment, ""
