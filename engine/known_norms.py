"""Named scoring norms: the Leading Eight and the money norm.

The Leading Eight (Ohtsuki and Iwasa, 2006) are the eight norms with two reputations
that keep cooperation stable. A barcode lists changes rather than new reputations (see
norms.py), so a norm that gives a donor with score 0 the reputation "good" has '+' there
and one that leaves a good donor good has '0'. They only change the donor, so NC is all
zeros. Exactly while scores stay in {0, 1} (no noise) this is the original two-state
model; with noise, scores can pass 1.

The barcodes, their names and the ESS strategy that goes with each are those of
createLeading8() in plotting_funcs.py of https://github.com/bdotloh/origin_money
(MIT License, Copyright (c) 2026 Marcus Frean and Stephen Marsland), which agree with
the norms found independently with this engine's ESS search (tests/test_known_norms.py).
"""

from dataclasses import dataclass

from .norms import ScoringNorm

NO_CHANGE = "00000000"

# The ESS strategy of each norm, as "recipient's class rows | donor's class columns".
HUMBLE = "g,0|g,g"     # also helps a recipient without score if the donor has none (AllC-Disc)
DISC = "0,0|g,g"       # helps a recipient with a score, never one without (Disc-Disc)


@dataclass(frozen=True)
class NamedNorm:
    label: str          # "L1", ..., "L8", "$"
    name: str           # "Stern Judging" and so on; "" where the code gives none
    CN: str
    NC: str
    strategy: str       # the ESS strategy that goes with the norm

    @property
    def barcode(self):
        return f"CN:{self.CN} NC:{self.NC}"

    @property
    def title(self):
        return f"{self.label} {self.name}".strip()

    def norm(self, tau=None):
        """The ScoringNorm; with tau, the binary norm lifted to the ternary classes."""
        norm = ScoringNorm.from_barcode(self.barcode)
        return norm if tau is None else norm.to_ternary(tau)


LEADING_EIGHT = (
    NamedNorm("L1", "", "00+00-+0", NO_CHANGE, HUMBLE),
    NamedNorm("L2", "Consistent Standing", "00+-0-+0", NO_CHANGE, HUMBLE),
    NamedNorm("L3", "Simple Standing", "+0+00-+0", NO_CHANGE, DISC),
    NamedNorm("L4", "", "+0000-+0", NO_CHANGE, DISC),
    NamedNorm("L5", "", "+0+-0-+0", NO_CHANGE, DISC),
    NamedNorm("L6", "Stern Judging", "+00-0-+0", NO_CHANGE, DISC),
    NamedNorm("L7", "Staying", "00000-+0", NO_CHANGE, DISC),
    NamedNorm("L8", "", "000-0-+0", NO_CHANGE, DISC),
)

# When a donor helps a recipient who has a score, the donor gains one and the recipient loses one.
# (The GUI's "Money" preset also moves a score when help is declined.)
MONEY = NamedNorm("$", "Money", "000000++", "000000--", DISC)
