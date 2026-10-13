"""Small layout helpers shared by the pages."""

from dash import html

GRAPH_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}


def card(title, children, className="card", **kwargs):
    header = [html.H3(title)] if title else []
    return html.Section(header + (children if isinstance(children, list) else [children]), className=className, **kwargs)


def field(label, control, hint=None, className="field"):
    parts = [html.Label(label), control]
    if hint:
        parts.append(html.Div(hint, className="hint"))
    return html.Div(parts, className=className)


def status(message, ok):
    return html.Div(message, className="status ok" if ok else "status error") if message else None
