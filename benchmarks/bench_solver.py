"""Time per solve and per chain step of StableScoreSolver:  python benchmarks/bench_solver.py"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import (  # noqa: E402
    BinaryStrategy, Defectors, Discriminators, Noise, PhiDisc, Population, ScoringNorm, StableScoreSolver,
    TernaryStrategy,
)

CASES = [
    ("2 binary, CN:00000-00", "CN:00000-00 NC:00000000", None, [Discriminators(), Defectors()], Noise(0.2, 0.1, 0.3)),
    ("3 binary, CN:00000-00", "CN:00000-00 NC:00000000", None,
     [Discriminators(), Defectors(), BinaryStrategy.from_name("g,g|g,g")], Noise(0.2, 0.1, 0.3)),
    ("ternary + norm (tau=3)", "CN:00000-0+ NC:00000000", 3,
     [TernaryStrategy.from_name("0,0,0|0,g,0.5|0,g,g", tau=3), Defectors()], Noise(0.1, 0.3, 0.7)),
    ("phi-Disc + Defectors", "CN:00000-0+ NC:00000000", None, [PhiDisc(1), Defectors()], Noise(0.1, 0.2, 0.4)),
    ("money, no noise", "CN:00000-++ NC:00000+--", None, [BinaryStrategy.from_name("0,0|g,g"), Defectors()], Noise(0)),
]

for label, barcode, tau, strategies, noise in CASES:
    norm = ScoringNorm.from_barcode(barcode)
    if tau:
        norm = norm.to_ternary(tau)
    solver = StableScoreSolver(norm, noise, tol=1e-8, max_iter=1000)
    population = Population(strategies)
    solver(population, [0.5, 0.5])
    start = time.perf_counter()
    repeats = 5
    for _ in range(repeats):
        result = solver(population, [0.5, 0.5])
    elapsed = (time.perf_counter() - start) / repeats
    print(f"{label:28s} {elapsed * 1e3:8.1f} ms/solve  {elapsed / result.n_iter * 1e6:6.0f} us/step  "
          f"({result.n_iter} steps, L={result.w.shape[1]})")
