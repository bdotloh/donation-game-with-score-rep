# Donation game with integer scores

`engine/` computes stable score (reputation) distributions and payoffs for a
well-mixed population of n strategies playing the donation game. Notation
follows *Calculating the Stable Reputation Distribution in Infinite Population
Models of Indirect Reciprocity*:

| Notes | Code |
|---|---|
| score j ∈ {0, 1, ...} | `j` |
| strategies i, k; frequencies ρ_k | `population.rho[k]` |
| w_j^(i), **w**₀ | `w[i, j]`, `result.w0` |
| G_j^(i), F_j^(i) (interactions only) | `rates.G[i, j]`, `rates.F[i, j]` |
| g_j^(i), f_j^(i) (with noise) | `rates.g[i, j]`, `rates.f[i, j]` |
| ε, ε↑, ε↓ | `Noise(eps, eps_up, eps_down)` |
| barcode `CN:… NC:…` | `ScoringNorm.from_barcode(...)` |

`g[i, j]` is the rate from score j-1 to j and `f[i, j]` the rate from j to j-1.

## Strategies

All decision points take a help probability in [0, 1] (`'0'` = 0, `'g'` = 1).

- `BinaryStrategy.from_name("0,0|0,g")` reacts to score possession (s = 0 / s ≥ 1).
  Rows are the receiver's class, columns the donor's.
- `TernaryStrategy.from_name("0,0,0|0,g,g|0,0.5,g", tau=2)` reacts to
  s = 0 / 0 < s ≤ τ / s > τ.
- `PhiDisc(phi)` doesn't help when score-less, and helps a receiver with a
  positive score with probability j^(-φ) when the donor has score j.

New kinds of strategy subclass `Strategy` and implement `_compute_donor_help`.

## Norms

`ScoringNorm.from_barcode("CN:00000-00 NC:00000000")` is a binary norm (classes
s = 0 / s ≥ 1). Ternary norms use the classes s = 0 / 0 < s ≤ τ / s > τ, with 18
characters per rule: `ScoringNorm.from_barcode(code, tau=2)`, or lift a binary
norm with `norm.to_ternary(tau)`. `all_norms(tau=None, use_CN=True, use_NC=True)`
enumerates every valid norm.

## Example

```python
from engine import *

norm = ScoringNorm.from_barcode("CN:00000-0+ NC:00000000")
population = Population([PhiDisc(1), Defectors()], rho=[0.9, 0.1])
solver = StableScoreSolver(norm, Noise(eps=0.1, eps_up=0.2, eps_down=0.4))
result = solver(population, w_init=[0.5, 0.5])

result.w            # stable score distributions, one row per strategy
result.rates.g      # g_j^(i)
Payoffs(benefit=2, cost=1)(population, result.w)
```

The chains run until the largest change in a step is at most `tol` (default
1e-5), or `max_iter` steps. The score range grows as needed.

`ESSFinder` finds the ESS among a set of strategies, as `find_ESS` in
`helper_funcs.jl` does.

## Replicator dynamics

```python
solver = StableScoreSolver(norm, Noise(eps=0.1, eps_up=0.2, eps_down=0.4), max_iter=1000)
dynamics = ReplicatorDynamics(solver, Payoffs(benefit=2, cost=1), dt=0.1)
trajectory = dynamics.run([PhiDisc(1), Defectors()], rho=[0.9, 0.1], w_init=[0.5, 0.5], n_steps=500)

trajectory.rho               # (T, n) strategy frequencies over time
trajectory.scores_converged  # False where the score chains hit max_iter
```

Each step restarts the score chains from `w_init`, runs them until they
converge or reach `max_iter`, computes the payoff matrix from the resulting
help probabilities and takes an Euler step of d ρ_i/dt = ρ_i (π_i − π̄). Where
the chains don't converge (scores drifting upwards for ever, or money spreading
out), payoffs come from the distribution after `max_iter` steps, so they
depend on `w_init` and `max_iter`. `warm_start=True` continues the chains from
the previous step instead.

## GUI

```
pip install -r requirements.txt
python -m gui            # then open http://127.0.0.1:8050
```

1. **Strategies**: pick binary, ternary or φ-Disc (or a preset), fill in p(help)
   for each decision point and add it; the 3D plot shows p(help) over receiver
   and donor scores. Choose 2 or 3.
2. **Parameters**: initial shares (fractions or percentages, e.g. 33.33 three
   times), initial score distribution, scoring norm (binary or ternary, one
   dropdown per case, or paste a barcode), noise ε, ε↑, ε↓ (ε↑ + ε↓ = 1), the
   probability q of being able to help, benefit and cost. Settings can be saved
   and loaded as JSON.
3. **Results**: the simplex (a line for 2 strategies, a triangle for 3) with
   arrows coloured and sized by speed, rest points (black stable, white
   unstable, gray saddle) and the trajectory from the initial shares. Click the
   simplex or type shares to see the score distributions, payoffs and helping
   rates there; press Play to step through the trajectory. Figures download as
   PNG or SVG from each plot's toolbar; "Download data" gives the trajectory,
   vector field, rest points and selected point as CSV/JSON.

## Tests

```
pip install numpy pytest
python -m pytest
```

`tests/test_against_julia.py` checks the engine against outputs of
`tests/reference/helper_funcs.jl`, saved in `tests/reference/*.tsv`.
`tests/reference/make_reference.jl` regenerates them, and
`tests/reference/compare_ess.py` checks every ESS row (the test suite checks a sample).
`tests/test_engine.py` checks the closed forms in the notes (Defectors vs
Discriminators, a population of 1-Disc) and the ternary and probabilistic cases.

The older code in `src/` and `experiments/` is unchanged for now.
