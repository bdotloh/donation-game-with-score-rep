"""The examples of the engine documentation (engine.tex).

Each example is a short piece of code; build_docs.py runs it and puts the code and what it printed into the
document, so the output shown is the engine's own. tests/test_docs.py runs them all.
"""

EXAMPLES = {}


def example(name, code):
    EXAMPLES[name] = code.strip("\n") + "\n"


# ---------------------------------------------------------------- partitions

example("partition", '''
from engine import TernaryPartition

p = TernaryPartition(3)                 # classes: s = 0 | s = 1, 2, 3 | s >= 4
print(p.cutoffs, p.n_classes)
print(p.classify([0, 1, 3, 4, 10]))     # class index of each score
print(p.class_masses([0.2, 0.3, 0.1, 0.1, 0.1, 0.2]))
''')

# ---------------------------------------------------------------- strategies

example("binary_strategy", '''
from engine import BinaryStrategy

disc = BinaryStrategy.from_name("0,0|0,g")   # helps only if both donor and receiver have a score
print(disc.name)
print(disc.P)                                # P[receiver class, donor class]
print(disc.donor_help(4))                    # [receiver class, donor score]
mixed = BinaryStrategy([[0, 0.5], [0.25, 1]])
print(mixed.name)
''')

example("ternary_strategy", '''
from engine import TernaryStrategy

s = TernaryStrategy.from_name("0,0,0|0,g,0.5|0,g,g", tau=2)
print(s.name)
print(s.partition.classify([0, 1, 2, 3, 4]))
print(s.donor_help(5))                       # rows: receiver in class 0, 1, 2
''')

example("phi_disc", '''
from engine import PhiDisc

never = PhiDisc(phi=1)
print(never.name)
print(never.donor_help(5)[1])        # receiver with a score: j^-1 for a donor with score j, 0 at j = 0
sometimes = PhiDisc(phi=1, p0=0.3)   # a donor with score 0 helps with probability 0.3
print(sometimes.name)
print(sometimes.donor_help(5)[1])
print(sometimes.donor_help(5)[0])    # a receiver with score 0 is never helped
''')

example("strategy_helpers", '''
from engine import (Defectors, Discriminators, all_binary_strategies, all_ternary_strategies,
                    strategy_from_name)

print(Defectors().name, Discriminators().name)
print(len(all_binary_strategies()), len(all_ternary_strategies(tau=2)))
print(strategy_from_name("0,g|0.5,g").name)
print(strategy_from_name("0,0,0|0,g,0.5|0,g,g (tau=2)").name)
print(strategy_from_name("phi:0.5,p0=0.2").name)
''')

# ---------------------------------------------------------------- norms

example("norm", '''
from engine import ScoringNorm

norm = ScoringNorm.from_barcode("CN:00000-00 NC:00000000")
print(norm.barcode, norm.is_ternary)
# CN[receiver class, act, donor class]; act 0 = declines, 1 = gives
print(norm.CN[1, 0, 1])      # a donor with a score who declines a receiver with a score: -1
print(norm.CN[1, 1, 1])      # ... who gives: 0
tern = norm.to_ternary(tau=2)
print(tern.barcode, tern.partition)
''')

example("all_norms", '''
from engine import all_norms

norms = list(all_norms(use_NC=False))   # every binary norm that only changes the donor
print(len(norms), norms[0].barcode)
print(norms[-1].barcode)
''')

example("named_norms", '''
from engine import LEADING_EIGHT, MONEY, ScoringNorm

for n in LEADING_EIGHT[:3]:
    print(n.title, "|", n.barcode, "|", n.strategy)
stern = LEADING_EIGHT[5]
print(stern.title, stern.norm().barcode)         # the ScoringNorm
print(stern.norm(tau=2).barcode)                 # lifted to ternary classes
print(MONEY.title, MONEY.barcode, MONEY.strategy)
''')

example("noise", '''
from engine import Noise

noise = Noise(eps=0.1, eps_up=0.4, eps_down=0.6)
print(noise)
try:
    Noise(eps=0.1, eps_up=0.7, eps_down=0.6)
except ValueError as error:
    print("ValueError:", error)
''')

# ---------------------------------------------------------------- population and rates

example("population", '''
from engine import Defectors, PhiDisc, Population
from engine.population import score_array

pop = Population([Defectors(), PhiDisc(1)], rho=[0.7, 0.3])
print(pop, pop.n)
print(score_array([0.5, 0.5], 2))                # one distribution shared by both strategies
print(score_array([[1.0], [0.5, 0.5]], 2))       # one each, padded with zeros
''')

example("rates", '''
import numpy as np
from engine import Noise, PhiDisc, Population, ScoringNorm, TransitionRates

norm = ScoringNorm.from_barcode("CN:00000-0+ NC:00000000")
rates = TransitionRates(norm, Noise(eps=0.1))
pop = Population([PhiDisc(1)])
w = np.array([[0.5, 0.3, 0.2, 0.0]])             # score distribution, last score empty
r = rates(pop, w)
np.set_printoptions(precision=4, suppress=True)
print("G", r.G[0])     # G[j]: rate j-1 -> j from interactions
print("F", r.F[0])     # F[j]: rate j -> j-1
print("g", r.g[0])     # with noise
print("f", r.f[0])
''')

example("chain", '''
import numpy as np
from engine import chain

np.set_printoptions(precision=4, suppress=True)
w = np.array([[0.5, 0.3, 0.2]])
w = chain.extend(w)                              # mass at the top score, so an empty score is added above it
print(w)
g = np.array([[0.0, 0.4, 0.3, 0.2, 0.1]])        # g[j]: rate j-1 -> j (g[0] unused), one more than w has scores
f = np.array([[0.0, 0.1, 0.2, 0.3]])             # f[j]: rate j -> j-1 (f[0] unused)
dw = chain.change(w, g, f)
print(dw, dw.sum())                              # the change sums to 0: probability is conserved
print(chain.stationary([0.0, 0.4, 0.3, 0.2], [0.0, 0.1, 0.2, 0.3]))   # stationary distribution of a fixed chain
''')

example("solver", '''
import numpy as np
from engine import Noise, PhiDisc, Population, ScoringNorm, StableScoreSolver

np.set_printoptions(precision=4, suppress=True)
norm = ScoringNorm.from_barcode("CN:00000-0+ NC:00000000")
solver = StableScoreSolver(norm, Noise(eps=0.1, eps_up=0.3, eps_down=0.7), tol=1e-10, max_iter=5000)
result = solver(Population([PhiDisc(1)]), [0.5, 0.5])
print(result.converged, result.drifting, result.n_iter)
print(result.w[0][:6])
print(result.w.sum(), result.w0)
''')

# ---------------------------------------------------------------- payoffs and dynamics

example("payoffs", '''
import numpy as np
from engine import (Defectors, Noise, Payoffs, PhiDisc, Population, ScoringNorm, StableScoreSolver)

np.set_printoptions(precision=4, suppress=True)
norm = ScoringNorm.from_barcode("CN:00000-0+ NC:00000000")
solver = StableScoreSolver(norm, Noise(eps=0.1, eps_up=0.3, eps_down=0.7), tol=1e-10, max_iter=5000)
pop = Population([PhiDisc(1), Defectors()], [0.5, 0.5])
w = solver(pop, [0.5, 0.5]).w
payoffs = Payoffs(benefit=2, cost=1)
print(payoffs.help_probabilities(pop, w))        # H[i, k]: a Can of i helps a Need of k
print(payoffs(pop, w))                           # payoff of each strategy
''')

example("replicator", '''
import numpy as np
from engine import (Defectors, Discriminators, Noise, Payoffs, ReplicatorDynamics, ScoringNorm, StableScoreSolver)

np.set_printoptions(precision=4, suppress=True)
norm = ScoringNorm.from_barcode("CN:00000-00 NC:00000000")      # lose a score for declining a receiver with a score
solver = StableScoreSolver(norm, Noise(eps=0.01), tol=1e-8, max_iter=2000)
dynamics = ReplicatorDynamics(solver, Payoffs(), dt=20.0)
traj = dynamics.run([Discriminators(), Defectors()], rho=[0.3, 0.7], w_init=[0.5, 0.5], n_steps=5)
print(traj.t)                                    # stopped at t = 60: Defectors have gone extinct
print(traj.rho)
print(traj.payoffs)
print(traj.at_rest, traj.scores_converged)
''')

example("field", '''
import numpy as np
from engine import (Defectors, Discriminators, Noise, Payoffs, ReplicatorDynamics, ReplicatorField, ScoringNorm,
                    StableScoreSolver, find_rest_points, simplex_grid)

np.set_printoptions(precision=4, suppress=True)
norm = ScoringNorm.from_barcode("CN:00000-00 NC:00000000")
solver = StableScoreSolver(norm, Noise(eps=0.01), tol=1e-8, max_iter=2000)
strategies = [Discriminators(), Defectors()]
field = ReplicatorField(ReplicatorDynamics(solver, Payoffs()), strategies, [0.5, 0.5])
point = field.evaluate([0.3, 0.7])
print(point.velocity, point.payoffs)             # d rho / dt and the payoffs at rho = (0.3, 0.7)
print(simplex_grid(2, 4))
for p in find_rest_points(field.velocity, n=2, resolution=8, neutral_tol=1e-6):
    print(p.location, p.kind, p.rho)
''')

example("ess", '''
from engine import (ESSFinder, Noise, Payoffs, ScoringNorm, StableScoreSolver, all_binary_strategies)

norm = ScoringNorm.from_barcode("CN:00000-00 NC:00000000")      # lose a score for declining a receiver with a score
solver = StableScoreSolver(norm, Noise(eps=0.01), tol=1e-5, max_iter=100)
strategies = all_binary_strategies()
finder = ESSFinder(solver, Payoffs(), rho=0.99)
ess = finder(strategies, strategies, w_init=[0.5, 0.5])
print([s.name for s in ess])
''')
