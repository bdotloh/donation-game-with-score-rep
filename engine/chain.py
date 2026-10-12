"""The birth-death chain on scores (Fig. 1), for n strategies at once.

w has shape (n, L): w[i, j] = w_j^(i). Rates follow rates.py: g[:, j] is the
rate from j-1 to j and f[:, j] the rate from j to j-1.
"""

import numpy as np


def extend(w):
    """Append an empty top score if any chain has mass there, so mass can always move up."""
    if np.any(w[:, -1] != 0.0):
        w = np.pad(w, ((0, 0), (0, 1)))
    return w


def change(w, g, f):
    """dw from one step of the chain. Needs w[:, -1] == 0 (see extend)."""
    up = g[:, 1:]       # up[:, j]: rate from j to j+1
    down = f            # down[:, j]: rate from j to j-1
    dw = -w * (up + down)
    dw[:, 1:] += w[:, :-1] * up[:, :-1]
    dw[:, :-1] += w[:, 1:] * down[:, 1:]
    return dw


def stationary(g, f):
    """Stationary distribution of a fixed birth-death chain on {0, ..., S} (Eqs. 5-6).

    g and f are 1-D arrays of length S+1 holding g_j, f_j for j = 1..S (index 0 unused);
    all of them must be positive.
    """
    g, f = np.asarray(g, dtype=float)[1:], np.asarray(f, dtype=float)[1:]
    log_ratio = np.concatenate(([0.0], np.cumsum(np.log(g) - np.log(f))))
    pi = np.exp(log_ratio - log_ratio.max())
    return pi / pi.sum()
