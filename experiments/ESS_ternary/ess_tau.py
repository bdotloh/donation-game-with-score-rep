#!/usr/bin/env python3
"""Which cutoff tau is an ESS under a given scoring norm?

    python experiments/ESS_ternary/ess_tau.py run --norm "CN:000000++ NC:000000--" --eps-up 0.4 --eps-down 0.6
    python experiments/ESS_ternary/ess_tau.py plot experiments/ESS_ternary/results_CN000000++_NC000000--_up0.4_down0.6

Strategies: one decision rule with the cutoff tau free. "I help a receiver with a positive score while
my own score is in 0..tau; above tau I stop helping. I never help a receiver with score 0."
    tau >= 1   TernaryStrategy "0,0,0|g,g,0|g,g,0" with that tau
    tau = 0    "I help you as long as you have a positive score", whatever my own score: "0,0|g,g"
plus Defectors. The norm, the ESS test and the outputs are described in experiments/ess_grid.py.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import ess_grid  # noqa: E402
from engine import BinaryStrategy, TernaryStrategy  # noqa: E402

RULE = "0,0,0|g,g,0|g,g,0"
TAU_ZERO = "0,0|g,g"


def family(a):
    """(label, strategy) for tau = 0..tau_max."""
    out = [("0", BinaryStrategy.from_name(TAU_ZERO))]
    return out + [(str(tau), TernaryStrategy.from_name(RULE, tau)) for tau in range(1, a.tau_max + 1)]


def add_family_args(parser):
    parser.add_argument("--tau-max", type=int, default=50)


if __name__ == "__main__":
    args = ess_grid.build_parser(__doc__, HERE, add_family_args).parse_args()
    if args.command == "run":
        ess_grid.run(args, family, {"rule": RULE, "tau_zero": TAU_ZERO})
    else:
        ess_grid.plot(args, "τ", 10)
