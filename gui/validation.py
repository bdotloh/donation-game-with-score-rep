"""Checks on what the user types. Each function returns the cleaned value and an error
message ("" when the input is fine), so the pages can show the message next to the input."""

import numpy as np

# Shares and masses may be typed as fractions (0.3333) or percentages (33.33).
FRACTION_TOL = 1e-3
PERCENT_TOL = 0.1
MAX_INITIAL_SCORE = 1000


def parse_number(value):
    """float from a number or a string (accepts "33.33", "1/3", "33.33%"); None if it isn't one."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().rstrip("%").strip()
        if not text:
            return None
        try:
            if "/" in text:
                num, den = text.split("/", 1)
                number = float(num) / float(den)
            else:
                number = float(text)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None
    return number if np.isfinite(number) else None


def normalise(values, what):
    """Non-negative numbers that add up to 1 (fractions) or 100 (percentages), rescaled to sum to exactly 1."""
    numbers = [parse_number(v) for v in values]
    if any(x is None for x in numbers):
        return None, f"Every {what} must be a number."
    numbers = np.array(numbers)
    if np.any(numbers < 0):
        return None, f"{what.capitalize()}s cannot be negative."
    total = numbers.sum()
    if abs(total - 1) <= FRACTION_TOL or abs(total - 100) <= PERCENT_TOL:
        return numbers / total, ""
    return None, f"{what.capitalize()}s add up to {total:g}; they must add up to 1 (or 100%)."


def shares(values):
    return normalise(values, "share")


def score_distribution(entries):
    """entries: list of {"score": int, "mass": number}. Returns w_init (index = score)."""
    if not entries:
        return None, "Add at least one score."
    for e in entries:
        _, error = score_entry(e.get("score"), e.get("mass"))
        if error:
            return None, error
    masses, error = normalise([e["mass"] for e in entries], "mass")
    if error:
        return None, error
    scores = [int(parse_number(e["score"])) for e in entries]
    w = np.zeros(max(scores) + 1)
    for j, m in zip(scores, masses):
        w[j] += m
    return w, ""


def score_entry(score, mass):
    """A new (score, mass) row; the score must be a whole number >= 0 and the mass a number >= 0."""
    s, m = parse_number(score), parse_number(mass)
    if s is None or s < 0 or s != int(s):
        return None, "The score must be a whole number, 0 or more."
    if s > MAX_INITIAL_SCORE:
        return None, f"The score must be at most {MAX_INITIAL_SCORE}."
    if m is None or m < 0:
        return None, "The mass must be a number, 0 or more."
    return {"score": int(s), "mass": m}, ""


def noise(eta_up, eta_down, epsilon):
    values = [parse_number(x) for x in (eta_up, eta_down, epsilon)]
    if any(x is None for x in values):
        return None, "η↑, η↓ and ε must be numbers."
    eta_up, eta_down, epsilon = values
    if eta_up < 0 or eta_down < 0:
        return None, "η↑ and η↓ are rates; they must be 0 or more."
    if not 0 <= epsilon <= 1:
        return None, "ε must lie between 0 and 1."
    return {"eta_up": eta_up, "eta_down": eta_down, "epsilon": epsilon}, ""


def probability(value):
    p = parse_number(value)
    if p is None or not 0 <= p <= 1:
        return None, "Each p(help) must be a number between 0 and 1."
    return p, ""


def positive_int(value, what):
    x = parse_number(value)
    if x is None or x < 1 or x != int(x):
        return None, f"{what} must be a whole number, 1 or more."
    return int(x), ""


def tau(value):
    """The ternary cut-off: classes 0 / 1..tau-1 / >= tau, so tau is at least 2."""
    x, error = positive_int(value, "τ")
    if not error and x >= 2:
        return x, ""
    return None, "τ must be a whole number, 2 or more."


def non_negative(value, what):
    x = parse_number(value)
    if x is None or x < 0:
        return None, f"{what} must be a number, 0 or more."
    return x, ""
