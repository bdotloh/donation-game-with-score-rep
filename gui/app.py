"""Dash app: strategies -> parameters -> results."""

from dash import Dash, Input, Output, dcc, html

from .pages import parameters, results, strategies

PAGES = {"/": strategies, "/parameters": parameters, "/results": results}
STEPS = [("/", "1  Strategies"), ("/parameters", "2  Parameters"), ("/results", "3  Results")]


def create_app():
    app = Dash(__name__, title="Replicator dynamics", suppress_callback_exceptions=True)
    app.layout = html.Div(
        [
            dcc.Location(id="url"),
            dcc.Store(id="store-strategies", storage_type="session", data=[]),
            dcc.Store(id="store-params", storage_type="session", data=None),
            dcc.Store(id="store-run", storage_type="session", data=None),
            html.Header(
                [
                    html.Div("Donation game · replicator dynamics", className="brand"),
                    html.Nav(id="steps", className="steps"),
                ],
                className="topbar",
            ),
            html.Main(id="page", className="page"),
        ]
    )

    @app.callback(Output("page", "children"), Output("steps", "children"), Input("url", "pathname"))
    def show_page(pathname):
        page = PAGES.get(pathname, strategies)
        steps = [
            html.Span(label, className="step active" if path == (pathname if pathname in PAGES else "/") else "step")
            for path, label in STEPS
        ]
        return page.layout(), steps

    for page in PAGES.values():
        page.register(app)
    return app
