"""Scoring norms, written in Frean and Marsland's barcode convention.

A norm gives the score change dS in {-1, 0, +1} of a player, depending on the
role, the classes of both players and whether help was given:

    CN[c_need, act, c_can]   change to the donor (Can) when meeting a Need player
    NC[c_need, act, c_can]   change to the receiver (Need) when meeting a Can player

with act = 0 (declines) or 1 (gives). The barcode lists each array in that
index order, e.g. "CN:00000-00 NC:00000000" for 2 classes (8 characters each).

Binary norms use the classes s = 0 / s >= 1. Ternary norms use
s = 0 / 0 < s <= tau / s > tau, with 18 characters each. A player in class 0
cannot lose a score.
"""

from itertools import product

import numpy as np

from .partitions import PossessionPartition, TernaryPartition

_SYMBOLS = {"-": -1, "0": 0, "+": 1}
_CHARS = {v: k for k, v in _SYMBOLS.items()}
_PREFIXES = {"CN": "CN", "CmeetsN": "CN", "NC": "NC", "NmeetsC": "NC"}


class ScoringNorm:
    def __init__(self, CN, NC, partition=None):
        partition = partition or PossessionPartition()
        n = partition.n_classes
        CN = np.asarray(CN, dtype=int)
        NC = np.asarray(NC, dtype=int)
        for label, rule in (("CN", CN), ("NC", NC)):
            if rule.shape != (n, 2, n):
                raise ValueError(f"{label} must have shape {(n, 2, n)}; got {rule.shape}.")
            if not np.isin(rule, (-1, 0, 1)).all():
                raise ValueError(f"{label} entries must be -1, 0 or 1.")
        if (CN[:, :, 0] == -1).any():
            raise ValueError("CN cannot decrease a donor in class 0 (score 0).")
        if (NC[0, :, :] == -1).any():
            raise ValueError("NC cannot decrease a receiver in class 0 (score 0).")
        self.CN = CN
        self.NC = NC
        self.partition = partition

    @classmethod
    def from_barcode(cls, barcode, tau=None):
        """Parse "CN:xxxxxxxx NC:xxxxxxxx" (binary) or 18-character codes (ternary, needs tau).

        The older "CmeetsN:... NmeetsC:..." spelling is also accepted.
        """
        codes = {}
        for part in barcode.split():
            label, code = part.split(":", 1)
            codes[_PREFIXES[label]] = code
        length = {len(code) for code in codes.values()}
        if len(length) != 1 or set(codes) != {"CN", "NC"}:
            raise ValueError(f"Expected CN and NC codes of equal length in {barcode!r}.")
        length = length.pop()
        if length == 8:
            if tau is not None:
                raise ValueError("A binary barcode takes no tau; use .to_ternary(tau) to lift it.")
            partition = PossessionPartition()
        elif length == 18:
            if tau is None:
                raise ValueError("A ternary barcode needs tau.")
            partition = TernaryPartition(tau)
        else:
            raise ValueError(f"Barcodes have 8 (binary) or 18 (ternary) characters; got {length}.")
        n = partition.n_classes

        def to_array(code):
            return np.array([_SYMBOLS[x] for x in code]).reshape((n, 2, n))

        return cls(to_array(codes["CN"]), to_array(codes["NC"]), partition)

    @property
    def barcode(self):
        def to_code(rule):
            return "".join(_CHARS[x] for x in rule.flat)

        return f"CN:{to_code(self.CN)} NC:{to_code(self.NC)}"

    @property
    def is_ternary(self):
        return self.partition.n_classes == 3

    def to_ternary(self, tau):
        """Ternary norm in which both positive classes act like s >= 1."""
        if self.is_ternary:
            raise ValueError("Norm is already ternary.")
        lift = [0, 1, 1]
        index = np.ix_(lift, [0, 1], lift)
        return ScoringNorm(self.CN[index], self.NC[index], TernaryPartition(tau))

    def __repr__(self):
        tau = f", tau={self.partition.tau}" if self.is_ternary else ""
        return f"ScoringNorm({self.barcode!r}{tau})"

    def __eq__(self, other):
        return (
            isinstance(other, ScoringNorm)
            and self.partition == other.partition
            and np.array_equal(self.CN, other.CN)
            and np.array_equal(self.NC, other.NC)
        )


def all_norms(tau=None, use_CN=True, use_NC=True):
    """Yield every valid norm (binary, or ternary if tau is given).

    An unused rule is fixed to all zeros.
    """
    partition = PossessionPartition() if tau is None else TernaryPartition(tau)
    n = partition.n_classes
    shape = (n, 2, n)

    def rules(enabled, can_lose):
        if not enabled:
            yield np.zeros(shape, dtype=int)
            return
        options = [(0, 1) if not can_lose[idx] else (-1, 0, 1) for idx in np.ndindex(shape)]
        for values in product(*options):
            yield np.array(values).reshape(shape)

    CN_can_lose = np.ones(shape, dtype=bool)
    CN_can_lose[:, :, 0] = False
    NC_can_lose = np.ones(shape, dtype=bool)
    NC_can_lose[0, :, :] = False
    for CN in rules(use_CN, CN_can_lose):
        for NC in rules(use_NC, NC_can_lose):
            yield ScoringNorm(CN, NC, partition)
