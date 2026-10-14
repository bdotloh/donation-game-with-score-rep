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
| ε, ε↑, ε↓ | `Noise(eps, eps_up, eps_down)` (ε↑ + ε↓ ≤ 1; the notes take it to be 1) |
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

`LEADING_EIGHT` (in `engine/known_norms.py`) holds the Leading Eight norms L1 to L8 as
barcodes (they change the donor only), each with the ESS strategy that goes with it, and
`MONEY` the paper's money norm. They are presets in the GUI. A barcode stores changes, so
they are the two-reputation Leading Eight exactly while scores stay at 0 and 1, which is
the case without noise.

## Example

```python
from engine import *

norm = ScoringNorm.from_barcode("CN:00000-0+ NC:00000000")
population = Population([PhiDisc(1), Defectors()], rho=[0.9, 0.1])
solver = StableScoreSolver(norm, Noise(eps=0.1, eps_up=0.4, eps_down=0.6))
result = solver(population, w_init=[0.5, 0.5])

result.w            # stable score distributions, one row per strategy
result.rates.g      # g_j^(i)
Payoffs(benefit=2, cost=1)(population, result.w)
```

The chains run until the largest change in a step is at most `tol` (default
1e-5), or `max_iter` steps. The score range grows as needed. `result.converged`
is False if the chains hit `max_iter`, or if some strategy's up-rate at the top
score is positive and at least its down-rate (`result.drifting`). Its scores are then
still spreading upwards, so a small change per step doesn't mean the distribution has
settled: at those rates there is no stable distribution. One may exist for the class
masses the chain would settle to, but the chain can take very long to get there. A
chain in which nothing can move (both rates 0) is not drifting.

`StableScoreSolver(..., role_factor=q * (1 - q))` and `Payoffs(role_factor=...)`
set the probability of a given ordered (donor, receiver) pair when each
individual can help with probability q (1/4 for q = 1/2, as in `helper_funcs.jl`).

`ESSFinder` finds the ESS among a set of strategies, as `find_ESS` in
`helper_funcs.jl` does.

## Replicator dynamics

```python
solver = StableScoreSolver(norm, Noise(eps=0.1, eps_up=0.4, eps_down=0.6), max_iter=1000)
dynamics = ReplicatorDynamics(solver, Payoffs(benefit=2, cost=1), dt=0.1)
# For runs of a fixed length: StableScoreSolver(..., max_iter=2000, force_max_iter=True, tail_tol=1e-15)
trajectory = dynamics.run([PhiDisc(1), Defectors()], rho=[0.9, 0.1], w_init=[0.5, 0.5], n_steps=500)

trajectory.rho               # (T, n) strategy frequencies over time
trajectory.scores_converged  # False where the score chains hit max_iter
```

Each step restarts the score chains from `w_init`, runs them until they
converge or reach `max_iter`, computes the payoff matrix from the resulting
help probabilities and takes an Euler step of d ρ_i/dt = ρ_i (π_i − π̄).
`ReplicatorField(dynamics, strategies, w_init)` evaluates all of this at any
composition (`evaluate_many` in parallel over `simplex_grid(n, resolution)`), and
`find_rest_points(field.velocity, n)` finds the rest points and classifies them
from the eigenvalues of the Jacobian on the simplex. Where
the chains don't converge (scores drifting upwards for ever, or money spreading
out), payoffs come from the distribution after `max_iter` steps, so they
depend on `w_init` and `max_iter`. `warm_start=True` continues the chains from
the previous step instead.

## GUI

```
pip install -r requirements.txt
python -m gui            # then open http://127.0.0.1:8050
```

1. **Strategies**: pick binary, ternary or φ-Disc, or a preset (Defectors,
   Cooperators, AllD-Disc, Disc-Disc), fill in p(help) for each decision point
   and add it; the 3D plot shows p(help) over receiver and donor scores. Choose
   2 or 3. They are labelled S1, S2, S3 in that order.
2. **Parameters**: initial shares (fractions or percentages, e.g. 33.33 three
   times), initial score distribution, scoring norm (binary or ternary, one
   dropdown per case, a preset, or paste a barcode), noise ε, ε↑, ε↓ (ε↑ + ε↓ = 1),
   the probability q of being able to help, benefit and cost. Under Advanced: the
   step size and number of steps of the replicator equation, the number of score
   iterations, the arrow grid, and whether every score chain runs for all its
   iterations. Defaults: ε = 0.1, ε↑ = ε↓ = 0.5, 2000 score iterations, all of
   them run. With ε↑ = ε↓ many populations have no stable score distribution, so
   values are those after the 2000 iterations. Settings can be saved and loaded
   as JSON.
3. **Results**: the score update norm comes first, as its barcode with how to read it and tables of the
   score changes; then a card shows the strategies as S1, S2, S3 in the colours of the plots, with their
   decision tables (p(help) for each class of receiver and donor; φ-Disc by donor score), their
   initial shares and, if ticked, their p(help) surfaces. Then the simplex (a line for 2 strategies, a triangle for 3) with a
   dense grid of arrows coloured and sized by speed, flow lines through the whole
   simplex, rest points (black stable, white unstable, gray saddle) and the
   trajectory from the initial shares. Where only some points' score chains
   converged, the others' arrows are lighter; a note says when none did. Click
   the simplex or type shares to see the score distributions (overlaid, one per
   strategy, plus the population; the x and y ranges can be typed), the payoff matrix and the helping rates
   there; strategies with share 0 are left out of these. Press Play to step
   through the trajectory. Figures download as PNG or SVG from each plot's
   toolbar; "Download data" gives the trajectory, vector field, rest points and
   selected point as CSV/JSON.

The steps at the top are links, so you can move between the pages freely. From Parameters,
"Current results →" goes back to the results of the last run without running it again (a
note says if the settings have been changed since); Run starts a new one.

**Saved results.** The Results page can save a finished run under a name, load it back, delete it, download
it as a file and upload a file saved earlier (a taken name is refused). A save is one SQLite file in
`saved_results/` (or `$DONATION_GAME_SAVES`), made of plain tables: `points` (one row for every point on the
simplex and every step of the trajectory, with its shares, velocity dρ/dt, payoffs), `payoff_matrix`,
`help_matrix`, `score_distribution` (with `distribution` giving the full length: the far tail, where every
probability is below 1e-12, is not stored and reads as 0), `rest_points`, `strategy`, `settings` and `run`, plus the
views `mean_score` and `point_strategy`. Loading brings back the plots without computing them again and puts the
settings on the Strategies and Parameters pages; a new point clicked on the simplex is still computed. The file
can be read from anywhere:

```python
import sqlite3, pandas as pd
db = sqlite3.connect("saved_results/my_run.sqlite")
pd.read_sql("select * from point_strategy where kind = 'grid' and strategy = 1", db)
pd.read_sql("select * from payoff_matrix where point_id = 1", db)
```

"Download as CSV" gives one CSV per table. Details are at the top of `gui/saves.py`.

`examples/` has settings files (load them on the first page) and phase
portraits for Defectors / AllD-Disc / ternary (τ = 5) and Defectors /
AllD-Disc / φ-Disc (φ = 0.17) under the money norm.

## Scripts

- `scripts/ess_sweep.py` finds the ESS for every norm in a set (binary or
  ternary norms, binary, ternary or φ-Disc strategies). Runs can be resumed and
  split over Slurm array tasks; `scripts/raapoi/` runs it on Rāpoi (see its README).
- `scripts/run_markov_chain.py` runs one- or two-strategy score chains and saves
  every step's rates, class masses and score distributions as CSV, with plots
  (needs matplotlib). `--rich-threshold K` is τ = K − 1.
- `experiments/ESS/find_ESS_parellel.py` is the earlier ESS sweep over the 1296
  CN-only binary norms. It writes the CSV that `experiments/ESS/plot_results.py`
  reads (`plot_results.py` also needs `plotting_funcs`, which isn't in this repo).
- `experiments/verify-zeroness-one-strategy/` groups the 20736 one-strategy
  chains by which of α, β, γ start out positive, and checks the limits claimed
  for each group. Those claims are about scores capped at maxw, so the check runs
  its own capped chain, with rates from the engine.

These used to call the code in `src/`, which is gone; they now use `engine/` and
reproduce the committed results. The old code capped scores at `maxw` and the
engine doesn't, so `--maxw` is no longer accepted. In the committed runs where
the cap mattered, the rates and class masses come out the same and the score
distributions go past `maxw`. A small `maxw` that sends mass back down changed
the rates too.

## Plotting the ESS

```
pip install -r requirements.txt
python scripts/ess_sweep.py run --norms binary-cn --eps 0.01 --maxn 500 --final-maxn 500 --out out/binary-cn
python scripts/ess_sweep.py merge out/binary-cn
python scripts/plot_ess.py out/binary-cn --show-cons          # writes out/binary-cn/ess_figure.png and .pdf
```

One row for every ESS found, sorted by cooperation level (fitness / (benefit - cost)),
lowest at the top. On the left the scoring norm: one cell for each (recipient's class,
whether the donor helped, donor's class), green where the score goes up, blue where it
goes down, grey where it stays. A cell is wide when the ESS strategy takes that action
there and narrow when it doesn't, so the wide cells are the ones that matter. On the right
the cooperation level; the bars of the Leading Eight with their ESS strategy are black
and labelled L1 to L8, and with `--show-cons` conservative norms (scores are only moved
around) are ticked. With up to 30 rows the strategies are written out in the paper's
names (AllD-Disc and so on). Binary and ternary norms both work (a ternary norm has 18
cells); `--threshold`, `--max-rows`, `--title` and `--dpi` change what is drawn.
`--paper-data` reads a data file of the paper's Julia code
(https://github.com/bdotloh/origin_money, `indirect_reciprocity/data-default`). The design
is that figure's (see `NOTICE`).

## Tests

```
pip install -r requirements.txt pytest
python -m pytest                 # engine, GUI logic
python -m pytest -m e2e          # clicks through the GUI in a browser (needs playwright)
```

`tests/test_against_julia.py` checks the engine against outputs of
`tests/reference/helper_funcs.jl`, saved in `tests/reference/*.tsv`.
`tests/reference/make_reference.jl` regenerates them, and
`tests/reference/compare_ess.py` checks every ESS row (the test suite checks a sample).
`tests/test_engine.py` checks the closed forms in the notes (Defectors vs
Discriminators, a population of 1-Disc) and the ternary and probabilistic cases;
`tests/test_simplex.py` checks rest points and their stability against games
with known answers. `tests/test_ess_sweep.py` and `tests/test_run_markov_chain.py`
check the scripts. `benchmarks/bench_solver.py` times the score solver.
