"""Checks on what the user types. Each function returns the cleaned value and an error
message ("" when the input is fine), so the pages can show the message next to the input."""

import numpy as np

# Shares and masses may be typed as fractions (0.3333) or percentages (33.33).
FRACTION_TOL = 1e-3
PERCENT_TOL = 0.1


def parse_number(value):
    """float from a number or a string (accepts "33.33", "1/3", "33.33%"); None if it isn't one."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if np.isfinite(value) else None
    text = str(value).strip().rstrip("%").strip()
    if not text:
        return None
    try:
        if "/" in text:
            num, den = text.split("/", 1)
            return float(num) / float(den)
        return float(text)
    except (ValueError, ZeroDivisionError):
        return None


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
    masses, error = normalise([e["mass"] for e in entries], "mass")
    if error:
        return None, error
    w = np.zeros(max(e["score"] for e in entries) + 1)
    for e, m in zip(entries, masses):
        w[e["score"]] += m
    return w, ""


def score_entry(score, mass):
    """A new (score, mass) row; the score must be a whole number >= 0 and the mass a number >= 0."""
    s, m = parse_number(score), parse_number(mass)
    if s is None or s < 0 or s != int(s):
        return None, "The score must be a whole number, 0 or more."
    if m is None or m < 0:
        return None, "The mass must be a number, 0 or more."
    return {"score": int(s), "mass": m}, ""


def noise(eps, eps_up, eps_down):
    values = [parse_number(x) for x in (eps, eps_up, eps_down)]
    if any(x is None for x in values):
        return None, "ε, ε↑ and ε↓ must be numbers."
    eps, eps_up, eps_down = values
    if not 0 <= eps <= 1:
        return None, "ε must lie between 0 and 1."
    if not (0 <= eps_up <= 1 and 0 <= eps_down <= 1):
        return None, "ε↑ and ε↓ must lie between 0 and 1."
    if abs(eps_up + eps_down - 1) > FRACTION_TOL:
        return None, f"ε↑ + ε↓ = {eps_up + eps_down:g}; it must be 1."
    total = eps_up + eps_down
    return {"eps": eps, "eps_up": eps_up / total, "eps_down": eps_down / total}, ""


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


def non_negative(value, what):
    x = parse_number(value)
    if x is None or x < 0:
        return None, f"{what} must be a number, 0 or more."
    return x, ""
