"""Vector fields, rest points and their stability on the simplex."""

import numpy as np
import pytest

from engine import (
    BinaryStrategy,
    Defectors,
    Discriminators,
    Noise,
    Payoffs,
    Population,
    ReplicatorDynamics,
    ReplicatorField,
    ScoringNorm,
    StableScoreSolver,
    find_rest_points,
    simplex_grid,
)


class LinearGame:
    """Replicator field of a constant payoff matrix: v = rho (A rho - rho . A rho)."""

    def __init__(self, A):
        self.A = np.asarray(A, dtype=float)

    def __call__(self, rho):
        pi = self.A @ rho
        return rho * (pi - rho @ pi), True


def kinds(points):
    return sorted((p.location, p.kind, tuple(np.round(p.rho, 3))) for p in points)


def find(points, rho):
    matches = [p for p in points if np.allclose(p.rho, rho, atol=1e-4)]
    assert len(matches) == 1, f"no single rest point at {rho}: {kinds(points)}"
    return matches[0]


@pytest.mark.parametrize("n, resolution, m", [(2, 4, 5), (3, 4, 15), (3, 1, 3)])
def test_simplex_grid(n, resolution, m):
    grid = simplex_grid(n, resolution)
    assert grid.shape == (m, n)
    np.testing.assert_allclose(grid.sum(axis=1), 1.0)
    assert len({tuple(p) for p in np.round(grid * resolution).astype(int)}) == m


def test_prisoners_dilemma():
    points = find_rest_points(LinearGame([[3, 0], [5, 1]]), 2, resolution=10)
    assert len(points) == 2
    assert find(points, [1, 0]).kind == "unstable"     # defectors invade cooperators
    assert find(points, [0, 1]).kind == "stable"


def test_hawk_dove():
    points = find_rest_points(LinearGame([[-1, 2], [0, 1]]), 2, resolution=10)
    assert find(points, [0.5, 0.5]).kind == "stable"
    assert find(points, [1, 0]).kind == "unstable" and find(points, [0, 1]).kind == "unstable"


def test_coordination_two_strategies():
    points = find_rest_points(LinearGame([[2, 0], [0, 1]]), 2, resolution=7)
    interior = find(points, [1 / 3, 2 / 3])
    assert interior.kind == "unstable" and interior.location == "interior"
    np.testing.assert_allclose(interior.eigenvalues, [2 / 3], atol=1e-3)
    assert find(points, [1, 0]).kind == "stable" and find(points, [0, 1]).kind == "stable"


@pytest.mark.parametrize("win, loss, kind", [(2, 1, "stable"), (1, 2, "unstable")])
def test_rock_paper_scissors(win, loss, kind):
    A = [[0, -loss, win], [win, 0, -loss], [-loss, win, 0]]
    points = find_rest_points(LinearGame(A), 3, resolution=8)
    centre = find(points, [1 / 3, 1 / 3, 1 / 3])
    assert centre.kind == kind and centre.location == "interior"
    assert np.iscomplexobj(centre.eigenvalues)          # a spiral
    for i in range(3):
        assert find(points, np.eye(3)[i]).kind == "saddle"
    assert len(points) == 4


def test_coordination_three_strategies():
    points = find_rest_points(LinearGame(np.eye(3)), 3, resolution=8)
    assert len(points) == 7
    for i in range(3):
        assert find(points, np.eye(3)[i]).kind == "stable"
    for edge in ([0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]):
        point = find(points, edge)
        assert point.kind == "saddle" and point.location == "edge"
        np.testing.assert_allclose(sorted(np.real(point.eigenvalues)), [-0.5, 0.5], atol=1e-3)
    assert find(points, [1 / 3] * 3).kind == "unstable"


def test_precomputed_grid_is_reused():
    game = LinearGame([[-1, 2], [0, 1]])
    grid = simplex_grid(2, 10)
    velocities = [game(rho)[0] for rho in grid]
    calls = []

    def counting(rho):
        calls.append(rho)
        return game(rho)

    points = find_rest_points(counting, 2, grid=grid, grid_velocities=velocities, grid_converged=[True] * len(grid))
    assert find(points, [0.5, 0.5]).kind == "stable"
    assert not any(np.allclose(c, g) for c in calls for g in grid[1:-1])   # interior grid points not re-evaluated


# ---------------------------------------------------------------- with the engine


def _field(norm="CN:00000-00 NC:00000000", noise=Noise(0.2, 0.1, 0.3), strategies=None, **solver_options):
    solver = StableScoreSolver(ScoringNorm.from_barcode(norm), noise, **{"tol": 1e-8, **solver_options})
    dynamics = ReplicatorDynamics(solver, Payoffs())
    strategies = strategies or [Discriminators(), Defectors()]
    return ReplicatorField(dynamics, strategies, np.array([0.5, 0.5])), dynamics, strategies


def test_evaluate_agrees_with_payoffs():
    field, dynamics, strategies = _field(strategies=[Discriminators(), Defectors(), BinaryStrategy.from_name("g,g|g,g")])
    rho = np.array([0.5, 0.3, 0.2])
    point = field.evaluate(rho)
    population = Population(strategies, rho)
    np.testing.assert_allclose(point.payoffs, Payoffs()(population, point.w), atol=1e-15)
    np.testing.assert_allclose(point.payoff_matrix, dynamics.payoff_matrix(population, point.w), atol=1e-15)
    np.testing.assert_allclose(point.help_matrix, Payoffs().help_probabilities(population, point.w), atol=1e-15)
    np.testing.assert_allclose(point.velocity.sum(), 0.0, atol=1e-15)
    assert point.converged


def test_evaluate_many_parallel_matches_serial():
    field, _, _ = _field()
    points = simplex_grid(2, 5)
    done = []
    serial = field.evaluate_many(points, n_jobs=1)
    parallel = field.evaluate_many(points, n_jobs=2, progress=lambda k, total: done.append((k, total)))
    for a, b in zip(serial, parallel):
        np.testing.assert_array_equal(a.velocity, b.velocity)
        np.testing.assert_array_equal(a.w, b.w)
    assert done[-1] == (len(points), len(points))


def test_converged_flag():
    field, _, _ = _field(norm="CN:00000-0+ NC:00000000", noise=Noise(0.1, 0.2, 0.4), max_iter=50)
    assert not field.evaluate([0.5, 0.5]).converged


def test_engine_rest_points_match_invasion_fitness():
    field, _, _ = _field()
    points = find_rest_points(field.velocity, 2, resolution=10)
    for p in points:
        assert np.abs(field.velocity(p.rho)[0]).max() < 1e-6
    for i in range(2):
        vertex = find(points, np.eye(2)[i])
        pi = field.evaluate(np.eye(2)[i]).payoffs
        invasion = pi[1 - i] - pi[i]
        assert np.sign(np.real(vertex.eigenvalues[0])) == np.sign(invasion)


# ---------------------------------------------------------------- cases from the review


def test_interior_rest_point_near_an_edge_is_found():
    """A stable interior point with one share below one grid cell (resolution 10)."""
    x = np.array([0.628, 0.078, 0.294])
    # A x = c 1 with negative-definite symmetric part on the tangent space: stable.
    B = -np.eye(3)
    A = B - (B @ x)[:, None] + 1.0
    points = find_rest_points(LinearGame(A), 3, resolution=10)
    interior = [p for p in points if p.location == "interior"]
    assert len(interior) == 1
    np.testing.assert_allclose(interior[0].rho, x, atol=1e-5)
    assert interior[0].kind == "stable"


def test_segments_of_rest_points_are_not_reported():
    assert len(find_rest_points(LinearGame(0.3 * np.ones((2, 2))), 2, resolution=10)) == 2   # vertices only
    points = find_rest_points(LinearGame([[1, 1, 0], [1, 1, 0], [0, 0, 2]]), 3, resolution=8)
    assert not [p for p in points if p.location == "edge" and p.rho[2] == 0]   # pi_1 = pi_2 on that edge


def test_zero_eigenvalue_is_neutral():
    points = find_rest_points(LinearGame([[0, 0], [0, 1]]), 2, resolution=10)
    assert find(points, [1, 0]).kind == "neutral"


def test_edge_point_next_to_a_vertex_is_kept():
    eps = 5e-4                                   # unstable rest point at rho_2 = eps / (1 + eps) next to e_1
    points = find_rest_points(LinearGame([[eps, eps], [0, 1]]), 2, resolution=10)
    assert len(points) == 3
    assert find(points, [1, 0]).kind == "stable" and find(points, [0, 1]).kind == "stable"


def test_eigenvalue_near_the_boundary():
    """Coordination game with its rest point at rho_2 = 0.005: eigenvalue rho_1 rho_2 (a + b)."""
    a, b = 0.005, 0.995
    points = find_rest_points(LinearGame([[a, 0], [0, b]]), 2, resolution=10)
    point = [p for p in points if p.location == "interior"][0]
    np.testing.assert_allclose(point.rho, [b, a], atol=1e-6)
    np.testing.assert_allclose(point.eigenvalues, [a * b * (a + b)], rtol=0.05)
