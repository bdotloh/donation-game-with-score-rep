"""Experiment settings as plain JSON-able dicts, and their conversion to engine objects.

strategy: {"type": "binary" | "ternary" | "phi", "name": str, "P": [[p, ...], ...], "tau": int, "phi": float}
norm:     {"type": "binary" | "ternary", "tau": int, "CN": "8 or 18 of -0+", "NC": "..."}
experiment: {"version": 1, "strategies": [...], "shares": [...], "scores": [{"score", "mass"}, ...],
             "norm": {...}, "noise": {"eps", "eps_up", "eps_down"}, "p_can", "benefit", "cost", "dt",
             "n_steps", "max_iter", "tol", "resolution", "full_steps"}
full_steps: run every score chain for max_iter steps instead of stopping at the tolerance.
"""

import numpy as np

from engine import (
    LEADING_EIGHT,
    MONEY,
    BinaryStrategy,
    Noise,
    Payoffs,
    PhiDisc,
    ReplicatorDynamics,
    ScoringNorm,
    StableScoreSolver,
    TernaryStrategy,
)
from engine.rates import role_factor

VERSION = 1
MAX_STRATEGIES = 3

STRATEGY_PRESETS = {
    # Names read as "behaviour when the donor has score 0" - "behaviour when the donor has a score".
    "Defectors": "0,0|0,0",
    "Cooperators": "g,g|g,g",
    "AllD-Disc": "0,0|0,g",
    "Disc-Disc": "0,0|g,g",
    "AllC-Disc": "g,0|g,g",
}

NORM_PRESETS = {
    "Lose a score for not helping someone with a score": "CN:00000-00 NC:00000000",
    "φ-Disc norm (notes, p. 7)": "CN:00000-0+ NC:00000000",
    "Money": "CN:00000-++ NC:00000+--",
    "Money (only when help is given)": MONEY.barcode,
    **{f"Leading Eight {n.title}": n.barcode for n in LEADING_EIGHT},
}

DEFAULT_ADVANCED = {
    "benefit": 2.0,
    "cost": 1.0,
    "dt": 1.0,
    "n_steps": 100,
    "max_iter": 2000,
    "tol": 1e-6,
    "resolution": 20,
    "full_steps": True,
}
TAIL_TOL = 1e-15        # scores whose mass falls below this everywhere are dropped (keeps long runs fast)


def class_labels(kind, tau=None):
    if kind == "ternary":
        return ["0", f"1–{tau}", f">{tau}"]
    return ["0", "≥1"]


def default_P(kind):
    n = 3 if kind == "ternary" else 2
    return [[0.0] * n for _ in range(n)]


def preset_strategy(name):
    P = BinaryStrategy.from_name(STRATEGY_PRESETS[name]).P
    return {"type": "binary", "name": name, "P": P.tolist(), "tau": None, "phi": None}


def build_strategy(cfg):
    if cfg["type"] == "binary":
        return BinaryStrategy(cfg["P"])
    if cfg["type"] == "ternary":
        return TernaryStrategy(cfg["P"], cfg["tau"])
    if cfg["type"] == "phi":
        return PhiDisc(cfg["phi"])
    raise ValueError(f"Unknown strategy type {cfg['type']!r}.")


def describe_strategy(cfg):
    if cfg["type"] == "phi":
        return f"φ-Disc, φ = {cfg['phi']:g}"
    strategy = build_strategy(cfg)
    text = strategy.name
    return f"{'ternary' if cfg['type'] == 'ternary' else 'binary'}: {text}"


def build_norm(cfg):
    barcode = f"CN:{cfg['CN']} NC:{cfg['NC']}"
    if cfg["type"] == "ternary":
        return ScoringNorm.from_barcode(barcode, tau=cfg["tau"])
    return ScoringNorm.from_barcode(barcode)


def norm_from_barcode(barcode, kind, tau=None):
    """Parse a pasted barcode into a norm cfg (raises ValueError if it is not valid)."""
    norm = ScoringNorm.from_barcode(barcode, tau=tau if kind == "ternary" else None)
    cn, nc = norm.barcode.split()
    return {"type": kind, "tau": tau, "CN": cn[3:], "NC": nc[3:]}


def default_norm(kind="binary", tau=2):
    n = 18 if kind == "ternary" else 8
    return {"type": kind, "tau": tau if kind == "ternary" else None, "CN": "0" * n, "NC": "0" * n}


def w_init(scores):
    w = np.zeros(max(int(e["score"]) for e in scores) + 1)
    for e in scores:
        w[int(e["score"])] += float(e["mass"])
    return w / w.sum()


def build(experiment):
    """Engine objects for an experiment: strategies, rho0, w_init, ReplicatorDynamics."""
    strategies = [build_strategy(s) for s in experiment["strategies"]]
    norm = build_norm(experiment["norm"])
    noise = Noise(**experiment["noise"])
    roles = role_factor(experiment.get("p_can", 0.5))
    solver = StableScoreSolver(norm, noise, tol=experiment["tol"], max_iter=experiment["max_iter"], role_factor=roles,
                               force_max_iter=bool(experiment.get("full_steps", False)), tail_tol=TAIL_TOL)
    payoffs = Payoffs(benefit=experiment["benefit"], cost=experiment["cost"], role_factor=roles)
    dynamics = ReplicatorDynamics(solver, payoffs, dt=experiment["dt"])
    return {
        "strategies": strategies,
        "names": [s["name"] for s in experiment["strategies"]],
        "rho0": np.asarray(experiment["shares"], dtype=float),
        "w_init": w_init(experiment["scores"]),
        "dynamics": dynamics,
    }


def fmt(x):
    """A number as text for an input box, without losing digits."""
    return f"{x:.15g}"


def check(experiment):
    """Error message for an experiment loaded from a file, or "" if it can be run."""
    from . import validation

    try:
        if not isinstance(experiment, dict):
            return "That file isn't a settings file."
        if experiment.get("version") != VERSION:
            return "This settings file is from a different version."
        strategies = experiment["strategies"]
        if not isinstance(strategies, list) or not 2 <= len(strategies) <= MAX_STRATEGIES:
            return "Choose 2 or 3 strategies."
        for s in strategies:
            if not isinstance(s, dict) or not isinstance(s.get("name"), str) or s.get("type") not in ("binary", "ternary", "phi"):
                return "A strategy in the file is not valid."
            if s["type"] == "phi" and validation.non_negative(s.get("phi"), "φ")[1]:
                return "φ must be a number, 0 or more."
            if s["type"] != "phi" and any(validation.probability(p)[1] for row in s.get("P") or [] for p in row):
                return "Each p(help) must be a number between 0 and 1."
        shares, error = validation.shares(experiment["shares"])
        if error or len(shares) != len(strategies):
            return error or "There must be one share per strategy."
        scores = experiment["scores"]
        if not isinstance(scores, list) or not all(isinstance(e, dict) for e in scores):
            return "The score distribution is not valid."
        _, error = validation.score_distribution(scores)
        if error:
            return error
        _, error = validation.noise(*(experiment["noise"].get(k) for k in ("eps", "eps_up", "eps_down")))
        if error:
            return error
        q = validation.parse_number(experiment.get("p_can", 0.5))
        if q is None or not 0 < q < 1:
            return "q must lie strictly between 0 and 1."
        for key in ("benefit", "cost", "tol"):
            if validation.non_negative(experiment[key], key)[1]:
                return f"{key} must be a number, 0 or more."
        dt = validation.parse_number(experiment["dt"])
        if dt is None or dt <= 0:
            return "The step size must be positive."
        for key in ("n_steps", "max_iter", "resolution"):
            if validation.positive_int(experiment[key], key)[1]:
                return f"{key} must be a whole number, 1 or more."
        build(experiment)
    except Exception as error:  # noqa: BLE001  (shown to the user)
        return f"These settings can't be used: {error}"
    return ""
