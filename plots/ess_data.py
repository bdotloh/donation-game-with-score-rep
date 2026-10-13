"""ESS results as rows (norm, ESS strategy, cooperation level) for plotting.

Two formats are read:

* the output of scripts/ess_sweep.py: a directory with ess.csv (or the per-task ess_task*.csv files) and
  config.json, or the csv itself;
* the data files of https://github.com/bdotloh/origin_money (the paper's Julia code): space-separated lines
  "CmeetsN NmeetsC [NmeetsN] Strategy fit w0 w-1 w1 a b c d" after a header of '#' lines.

The cooperation level is the fitness divided by benefit - cost, the most a player can get.
"""

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from engine import ScoringNorm, Strategy, strategy_from_name


@dataclass(frozen=True)
class EssRow:
    norm: ScoringNorm
    strategy: Strategy
    cooperation: float


def read_sweep(path, tau=None):
    """Rows from an ess_sweep.py output directory or csv. tau is read from config.json unless given."""
    path = Path(path)
    if path.is_dir():
        directory = path
        files = [directory / "ess.csv"] if (directory / "ess.csv").exists() else sorted(directory.glob("ess_task*.csv"))
        if not files:
            raise FileNotFoundError(f"No ess.csv or ess_task*.csv in {directory}.")
    else:
        directory, files = path.parent, [path]
    config = {}
    if (directory / "config.json").exists():
        config = json.loads((directory / "config.json").read_text())
    tau = tau if tau is not None else config.get("tau")
    scale = float(config.get("benefit", 2.0)) - float(config.get("cost", 1.0))
    rows = []
    for file in files:
        with file.open(newline="") as handle:
            for record in csv.DictReader(handle):
                barcode = record["norm"]
                ternary = len(barcode.split()[0].split(":")[1]) == 18
                norm = ScoringNorm.from_barcode(barcode, tau=tau if ternary else None)
                strategy = strategy_from_name(record["ess"], tau=tau)
                rows.append(EssRow(norm, strategy, float(record["fitness"]) / scale))
    return rows


def read_paper_data(path):
    """Rows from a data file of the paper's code (binary norms)."""
    benefit = cost = None
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith("#"):
                if (match := re.search(r"BENEFIT\s+([\d.]+)", line)):
                    benefit = float(match.group(1))
                if (match := re.search(r"COST\s+([\d.]+)", line)):
                    cost = float(match.group(1))
                continue
            fields = line.split()
            if not fields:
                continue
            if len(fields) == 12:
                cn, nc, _, strategy, fit = fields[:5]
            elif len(fields) == 11:
                cn, nc, strategy, fit = fields[:4]
            else:
                raise ValueError(f"Expected 11 or 12 columns; got {len(fields)} in {line.strip()!r}.")
            rows.append((cn, nc, strategy, float(fit)))
    if benefit is None or cost is None:
        raise ValueError(f"No '# BENEFIT .., COST ..' line in {path}.")
    scale = benefit - cost
    return [EssRow(ScoringNorm.from_barcode(f"CN:{cn} NC:{nc}"), strategy_from_name(strategy), fit / scale)
            for cn, nc, strategy, fit in rows]
