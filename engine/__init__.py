"""Engine for the donation game with integer scores (reputations).

Notation follows "Calculating the Stable Reputation Distribution in Infinite
Population Models of Indirect Reciprocity": scores j in {0, 1, ...},
strategies i, k with frequencies rho, score distributions w[i, j] = w_j^(i),
rates G, F (interactions only) and g, f (with noise eps, eps_up, eps_down).
"""

from .ess import ESSFinder
from .noise import Noise
from .norms import ScoringNorm, all_norms
from .partitions import PossessionPartition, ScorePartition, TernaryPartition
from .payoffs import Payoffs
from .population import Population
from .rates import Rates, TransitionRates
from .replicator import ReplicatorDynamics, Trajectory
from .simplex import PointResult, ReplicatorField, RestPoint, find_rest_points, simplex_grid
from .solver import StableScores, StableScoreSolver
from .strategies import (
    BinaryStrategy,
    ClassStrategy,
    Defectors,
    Discriminators,
    PhiDisc,
    Strategy,
    TernaryStrategy,
)
