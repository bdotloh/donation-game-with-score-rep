#!/usr/bin/env python3
"""Which phi of phi-Disc is an ESS under a given scoring norm?

    python experiments/ESS_phi/ess_phi.py run --norm "CN:000000++ NC:000000--" --eps-up 0.4 --eps-down 0.6
    python experiments/ESS_phi/ess_phi.py plot experiments/ESS_phi/results_CN000000++_NC000000--_up0.4_down0.6

Strategies: PhiDisc(phi, p0=1) for phi = 0, 0.02, ..., 2. A donor with score j >= 1 helps a receiver
with a positive score with probability j^(-phi); a donor with score 0 always helps such a receiver.
A receiver with score 0 is never helped. phi = 0 helps every receiver with a positive score.
Plus Defectors. The norm, the ESS test and the outputs are described in experiments/ess_grid.py.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import ess_grid  # noqa: E402
from engine import PhiDisc  # noqa: E402


def family(a):
    """(label, strategy) for phi = 0, phi_step, ..., phi_max."""
    n = round(a.phi_max / a.phi_step)
    return [(f"{k * a.phi_step:.2f}", PhiDisc(k * a.phi_step, p0=a.p0)) for k in range(n + 1)]


def add_family_args(parser):
    parser.add_argument("--phi-max", type=float, default=2.0)
    parser.add_argument("--phi-step", type=float, default=0.02)
    parser.add_argument("--p0", type=float, default=1.0, help="help probability of a donor with score 0")


if __name__ == "__main__":
    args = ess_grid.build_parser(__doc__, HERE, add_family_args).parse_args()
    if args.command == "run":
        ess_grid.run(args, family, {"strategy": "PhiDisc(phi, p0)"})
    else:
        ess_grid.plot(args, "φ", 10)
