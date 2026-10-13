"""Phase portraits of the replicator dynamics on the strategy simplex.

ReplicatorField evaluates d rho / dt (with the payoffs, help probabilities and
scores behind it) at any rho, one point at a time or many in parallel.
simplex_grid gives a barycentric grid of points, and find_rest_points finds and
classifies the rest points of a velocity field on the simplex for n = 2 or 3.

Stability comes from the Jacobian of the replicator restricted to the simplex,
in the coordinates rho = rho* + sum_k z_k (e_k - e_r), k != r, where r is the
most common strategy at rho*; then dz_k / dt = v_k. Its columns are finite
differences along e_k - e_r, taken so that every point evaluated stays in the
simplex:

- Off the face of rho* (rho*_k = 0), forward steps BOUNDARY_STEP and half of it,
  extrapolated to step 0. As
  v_k = rho_k (pi_k - pi_bar), the difference v_k(rho* + h (e_k - e_r)) / h is
  the invasion fitness pi_k - pi_bar of k at rho_k = h: the step only cancels the
  factor rho_k, so the payoffs' numerical error is not amplified and the step
  can be small (the error is h times the slope of the payoffs). At a vertex e_i
  these are the invasion fitnesses (1 - h)(pi_k - pi_i).
- Within the face (along an edge, or in the interior), central differences
  with FD_STEP, or with half the distance to the boundary when that is closer. Here the
  payoffs' error is divided by the step. With the solver's default tol = 1e-5
  that error is up to ~1e-6 from one point to the next (it jumps when the number
  of chain steps changes; the smooth part of the bias does not matter), so the
  error of a central difference, about 1e-6 / h + h^2 |v'''| / 6, is smallest
  around h ~ 1e-2. Interior rest points are refined by Newton's method with the
  same step, for the same reason (scipy's default steps of ~1e-8 would see only
  this noise).

v_k vanishes on the whole face where rho_k = 0, so the Jacobian is block
triangular: its eigenvalues are those within the face (along the edge, first),
then the invasion fitnesses of the missing strategies (transverse).
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import brentq

from .population import Population

FD_STEP = 1e-2          # finite-difference step within the face of a point
BOUNDARY_STEP = 1e-6    # forward step off the face (invasion fitnesses)
INSIDE_STEP = 1e-3      # boundary grid points are moved this far inside to read their payoff differences
MERGE_TOL = 1e-3        # rest points of the same kind of location closer than this are merged
REST_TOL = 1e-3         # a rest point's payoffs agree to REST_TOL times their largest spread on the grid
ZERO_TOL = 1e-9         # payoff differences below ZERO_TOL times the largest spread count as zero
SINGULAR_TOL = 1e-6     # interior points whose Jacobian has |det| below this times |J|^2 lie on curves
MAX_NEWTON = 30


@dataclass
class PointResult:
    rho: np.ndarray            # (n,)
    velocity: np.ndarray       # (n,) d rho / dt
    payoffs: np.ndarray        # (n,) pi_i
    payoff_matrix: np.ndarray  # (n, n) A, with pi = A @ rho
    help_matrix: np.ndarray    # (n, n) H[i, k] = Pr(Can of strategy i helps Need of strategy k)
    w: np.ndarray              # (n, L) score distributions
    converged: bool            # score chains converged (or transient) rather than hitting max_iter


class ReplicatorField:
    """The velocity field of ReplicatorDynamics, as a function of rho alone.

    At every rho the score chains restart from w_init, as ReplicatorDynamics.run does.
    """

    def __init__(self, dynamics, strategies, w_init):
        self.dynamics = dynamics
        self.strategies = list(strategies)
        self.w_init = w_init

    def evaluate(self, rho):
        population = Population(self.strategies, np.array(rho, dtype=float))
        rho = population.rho
        result = self.dynamics.solver(population, self.w_init)
        A = self.dynamics.payoff_matrix(population, result.w)
        pi = A @ rho
        return PointResult(
            rho=rho,
            velocity=rho * (pi - rho @ pi),
            payoffs=pi,
            payoff_matrix=A,
            help_matrix=self.dynamics.payoffs.help_probabilities(population, result.w),
            w=result.w,
            converged=bool(result.converged or result.transient),
        )

    def velocity(self, rho):
        """(d rho / dt, whether the score chains converged)."""
        result = self.evaluate(rho)
        return result.velocity, result.converged

    def evaluate_many(self, points, n_jobs=None, progress=None):
        """evaluate() at each point, in input order.

        n_jobs worker processes (None: one per CPU; 1: in this process). The
        field is sent to each worker once, so it must be picklable; under the
        'spawn' start method the calling script needs an
        `if __name__ == "__main__":` guard. progress(done, total) is called as
        results arrive.
        """
        points = [np.array(rho, dtype=float) for rho in points]
        total = len(points)
        n_jobs = (os.cpu_count() or 1) if n_jobs is None else n_jobs
        if n_jobs < 1:
            raise ValueError(f"n_jobs must be at least 1; got {n_jobs}.")
        if n_jobs == 1 or total <= 1:
            results = []
            for rho in points:
                results.append(self.evaluate(rho))
                if progress is not None:
                    progress(len(results), total)
            return results

        results = [None] * total
        with ProcessPoolExecutor(min(n_jobs, total), mp_context=_pool_context(), initializer=_set_worker_field,
                                 initargs=(self,)) as pool:
            futures = {pool.submit(_evaluate_in_worker, rho): index for index, rho in enumerate(points)}
            try:
                for done, future in enumerate(as_completed(futures), 1):
                    results[futures[future]] = future.result()
                    if progress is not None:
                        progress(done, total)
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
        return results


_worker_field = None


def _pool_context():
    """Start workers with 'forkserver' where available: forking a process that runs other threads
    (a web server, say) can deadlock the workers."""
    methods = multiprocessing.get_all_start_methods()
    return multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")


def _set_worker_field(field):
    global _worker_field
    _worker_field = field


def _evaluate_in_worker(rho):
    return _worker_field.evaluate(rho)


def simplex_grid(n, resolution):
    """(m, n) points counts / resolution, over all n non-negative integer counts
    summing to resolution. For n = 2 they run evenly from e_1 to e_2."""
    if n < 1 or resolution < 1:
        raise ValueError(f"Need n >= 1 and resolution >= 1; got {n}, {resolution}.")
    return np.array(list(_compositions(resolution, n)), dtype=float) / resolution


def _compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(total, -1, -1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


@dataclass
class RestPoint:
    rho: np.ndarray
    kind: str                  # 'stable' | 'unstable' | 'saddle' | 'neutral'
    eigenvalues: np.ndarray    # of the Jacobian of the replicator restricted to the simplex (n-1 of them)
    location: str              # 'vertex' | 'edge' | 'interior'
    converged: bool


def find_rest_points(velocity_fn, n, resolution=20, grid=None, grid_velocities=None, xtol=1e-6, neutral_tol=1e-7,
                     grid_converged=None):
    """Rest points of the replicator velocity field velocity_fn(rho) -> (velocity, converged), n = 2 or 3.

    grid, grid_velocities (and grid_converged): a simplex_grid (any resolution, any order)
    and the velocities (and converged flags) on it, if already computed; otherwise
    simplex_grid(n, resolution) is evaluated. Vertices are always included. Rest points on the edges are
    sign changes of pi_b - pi_a = v_b / (rho_a rho_b) between grid points (and
    points BOUNDARY_STEP from the vertices), refined by brentq to xtol. Interior
    rest points (n = 3) start from grid cells where pi_1 - pi_3 and pi_2 - pi_3
    both change sign, and are refined by Newton's method; one closer to the
    boundary than about a grid cell can be missed. A refined point is kept if the
    payoffs of the strategies present agree to REST_TOL times their largest
    spread on the grid. Points of segments or curves of rest points (strategies with equal
    payoffs along a whole line) are not reported, except where a curve meets an edge.

    kind: every Re(eigenvalue) < -tol is 'stable', every one > tol 'unstable', some of
    each 'saddle', anything else 'neutral', where tol is neutral_tol or twice the estimated
    error of the eigenvalues, whichever is larger. Pass a neutral_tol of the order of the
    payoffs' numerical error (e.g. ten times the score solver's tolerance) for engine fields. converged is
    whether the score chains converged at the point and at the points used for
    its eigenvalues.
    """
    if n not in (2, 3):
        raise ValueError(f"find_rest_points handles n = 2 or 3; got {n}.")
    field = _CachedField(velocity_fn)
    grid = simplex_grid(n, resolution) if grid is None else np.asarray(grid, dtype=float)
    resolution = _grid_resolution(grid, n)
    if grid_velocities is None:
        velocities = np.array([field.evaluate(rho)[0] for rho in grid])
    else:
        velocities = np.asarray(grid_velocities, dtype=float)
        if velocities.shape != grid.shape:
            raise ValueError(f"grid_velocities must have shape {grid.shape}; got {velocities.shape}.")
        flags = [None] * len(grid) if grid_converged is None else [bool(c) for c in grid_converged]
        for rho, v, c in zip(grid, velocities, flags):
            field.store(rho, v, c)
    on_grid = {tuple(c): (rho, v) for c, rho, v in zip(np.rint(grid * resolution).astype(int), grid, velocities)}
    spread = max(_payoff_spread(rho, v) for rho, v in zip(grid, velocities))
    rest_tol = REST_TOL * spread + 1e-12

    zero_tol = ZERO_TOL * spread + 1e-15
    candidates = [(np.eye(n)[i], "vertex") for i in range(n)]
    for a, b in combinations(range(n), 2):
        location = "interior" if n == 2 else "edge"
        candidates += [(rho, location) for rho in _edge_roots(field, n, a, b, resolution, on_grid, xtol, zero_tol)]
    if n == 3:
        found = []
        for rho in _interior_roots(field, resolution, on_grid, xtol, rest_tol, avoid=found):
            found.append(rho)
        candidates += [(rho, "interior") for rho in found]

    points = []
    for rho, location in candidates:
        # Never merge into a vertex (an edge point can sit very close to one); merge the rest.
        if any((location == "vertex") == (p.location == "vertex") and np.abs(rho - p.rho).max() <= MERGE_TOL
               for p in points):
            continue
        v, converged = field.evaluate(rho)
        if location != "vertex" and _payoff_spread(rho, v) > rest_tol:
            continue
        eigenvalues, error, fd_converged, n_face = field.eigenvalues(rho, v)
        tol = max(neutral_tol, 2 * error)
        if location != "vertex" and np.any(np.abs(eigenvalues[:n_face]) <= tol):
            continue                    # on a curve or segment of rest points within its face
        kind = _classify(eigenvalues, tol)
        points.append(RestPoint(rho, kind, eigenvalues, location, converged and fd_converged))
    return points


class _CachedField:
    """velocity_fn with a cache, and the Jacobian of the replicator on the simplex."""

    def __init__(self, velocity_fn):
        self.velocity_fn = velocity_fn
        self.cache = {}

    @staticmethod
    def _key(rho):
        return tuple(np.round(rho, 14))

    def store(self, rho, v, converged=None):
        self.cache.setdefault(self._key(rho), (np.asarray(v, dtype=float), converged))

    def velocity(self, rho):
        hit = self.cache.get(self._key(rho))
        return self.evaluate(rho)[0] if hit is None else hit[0]

    def evaluate(self, rho):
        """(velocity, converged)."""
        key = self._key(rho)
        hit = self.cache.get(key)
        if hit is None or hit[1] is None:
            v, converged = self.velocity_fn(rho)
            hit = self.cache[key] = (np.asarray(v, dtype=float), bool(converged))
        return hit

    def eigenvalues(self, rho, v0):
        """Eigenvalues of the restricted Jacobian at rho (see the module docstring; the first
        n_face are within the face of rho), an estimate of their error (from repeating the
        differences with half the step), whether the score chains converged at every point used,
        and n_face."""
        n = len(rho)
        r = int(np.argmax(rho))
        face = [k for k in range(n) if k != r and rho[k] > 0]
        off = [k for k in range(n) if k != r and rho[k] == 0]
        J, converged = self._jacobian(rho, v0, r, face, off, 1.0)
        J_half, c = self._jacobian(rho, v0, r, face, off, 0.5)
        converged = converged and c
        # Off the face the forward difference is biased by (step x slope); halving the step and
        # extrapolating removes that bias. Within the face keep the full step (less noise).
        transverse = slice(len(face), None)
        J_best = J.copy()
        J_best[:, transverse] = 2 * J_half[:, transverse] - J[:, transverse]
        eigenvalues = _block_eigenvalues(J_best, len(face))
        error = float(np.abs(np.sort_complex(_block_eigenvalues(J, len(face)).astype(complex))
                             - np.sort_complex(_block_eigenvalues(J_half, len(face)).astype(complex))).max())
        return eigenvalues, error, converged, len(face)

    def _jacobian(self, rho, v0, r, face, off, scale):
        n = len(rho)
        order = face + off
        J = np.empty((n - 1, n - 1))
        converged = True
        for col, k in enumerate(order):
            d = np.zeros(n)
            d[k], d[r] = 1.0, -1.0
            if rho[k] == 0:
                steps = (scale * BOUNDARY_STEP, 0.0)
            else:
                h = scale * min(FD_STEP, rho[k] / 2)   # rho[r] >= 1/n > FD_STEP, so +h stays inside
                steps = (h, -h)
            vs = []
            for h in steps:
                if h == 0.0:
                    vs.append(v0)
                    continue
                v, c = self.evaluate(_on_simplex(rho + h * d))
                vs.append(v)
                converged = converged and c
            J[:, col] = ((vs[0] - vs[1]) / (steps[0] - steps[1]))[order]
        return J, converged


def _on_simplex(rho):
    rho = np.clip(rho, 0.0, None)
    return rho / rho.sum()


def _block_eigenvalues(J, n_face):
    """Eigenvalues of J, block by block if J is block upper triangular."""
    blocks = [J[:n_face, :n_face], J[n_face:, n_face:]] if not J[n_face:, :n_face].any() else [J]
    eigenvalues = []
    for B in blocks:
        if B.size == 0:
            continue
        if not np.triu(B, 1).any() or not np.tril(B, -1).any():
            eigenvalues.append(np.diag(B).copy())
        else:
            eigenvalues.append(np.linalg.eigvals(B))
    return np.concatenate(eigenvalues)


def _classify(eigenvalues, tol):
    re = np.real(eigenvalues)
    if np.any(re > tol) and np.any(re < -tol):
        return "saddle"
    if np.all(re < -tol):
        return "stable"
    if np.all(re > tol):
        return "unstable"
    return "neutral"


def _payoff_spread(rho, v):
    """max - min of pi_k - pi_bar = v_k / rho_k over the strategies present."""
    present = rho > 0
    if present.sum() < 2:
        return 0.0
    relative = v[present] / rho[present]
    return relative.max() - relative.min()


def _grid_resolution(grid, n):
    m = len(grid)
    resolution = m - 1 if n == 2 else int(round((np.sqrt(8 * m + 1) - 3) / 2))
    counts = grid * resolution
    expected = resolution + 1 if n == 2 else (resolution + 1) * (resolution + 2) // 2
    if (
        grid.ndim != 2
        or grid.shape[1] != n
        or resolution < 1
        or m != expected
        or np.abs(counts - np.rint(counts)).max() > 1e-6
        or len({tuple(c) for c in np.rint(counts).astype(int)}) != m
    ):
        raise ValueError("grid must be the points of simplex_grid(n, resolution).")
    return resolution


def _edge_point(n, a, b, t):
    rho = np.zeros(n)
    rho[a], rho[b] = 1.0 - t, t
    return rho


def _edge_roots(field, n, a, b, resolution, on_grid, xtol, zero_tol):
    """Points on the edge from e_a to e_b where pi_a = pi_b. Differences within zero_tol of 0
    count as 0; a run of several zeros (a segment of rest points) is not reported."""

    def payoff_difference(t):
        return field.velocity(_edge_point(n, a, b, t))[b] / (t * (1.0 - t))

    ts = [BOUNDARY_STEP]
    values = [payoff_difference(BOUNDARY_STEP)]
    for j in range(1, resolution):
        counts = [0] * n
        counts[a], counts[b] = resolution - j, j
        rho, v = on_grid[tuple(counts)]
        ts.append(j / resolution)
        values.append(v[b] / (rho[a] * rho[b]))
    ts.append(1.0 - BOUNDARY_STEP)
    values.append(payoff_difference(1.0 - BOUNDARY_STEP))

    signs = [0 if abs(f) <= zero_tol else (1 if f > 0 else -1) for f in values]
    roots = []
    k = 0
    while k < len(ts) - 1:
        if signs[k] != 0 and signs[k + 1] != 0:
            if signs[k] != signs[k + 1]:
                try:
                    roots.append(brentq(payoff_difference, ts[k], ts[k + 1], xtol=xtol))
                except ValueError:      # grid velocities that disagree with velocity_fn
                    pass
            k += 1
            continue
        if signs[k] == 0:
            k += 1
            continue
        # A run of zeros starts at k + 1: keep it only if it is a single point between opposite signs.
        end = k + 1
        while end < len(ts) and signs[end] == 0:
            end += 1
        if end < len(ts) and end == k + 2 and signs[end] == -signs[k]:
            roots.append(ts[k + 1])
        k = end
    return [_edge_point(n, a, b, t) for t in roots]


def _relative_payoffs(rho, v):
    """(pi_1 - pi_3, pi_2 - pi_3) at an interior point."""
    relative = v / rho
    return relative[:2] - relative[2]


def _interior_roots(field, resolution, on_grid, xtol, rest_tol, avoid):
    """Interior rest points for n = 3, from the grid cells where both pi_1 - pi_3 and
    pi_2 - pi_3 change sign. Cells holding a point of `avoid` are skipped."""
    values = {}
    centre = np.full(3, 1 / 3)
    for counts, (rho, v) in on_grid.items():
        if min(counts) > 0:
            values[counts[:2]] = (rho, _relative_payoffs(rho, v))
        else:
            # Payoff differences on the boundary, read just inside it.
            inside = (1 - INSIDE_STEP) * rho + INSIDE_STEP * centre
            values[counts[:2]] = (rho, _relative_payoffs(inside, field.velocity(inside)))
    for a in range(resolution):
        for b in range(resolution - a):
            for cell in (((a, b), (a + 1, b), (a, b + 1)), ((a + 1, b), (a, b + 1), (a + 1, b + 1))):
                if not all(corner in values for corner in cell):
                    continue
                P = np.array([values[corner][0] for corner in cell])
                G = np.array([values[corner][1] for corner in cell])
                if not G.any(axis=0).all():
                    continue            # a payoff difference that vanishes on the whole cell
                if not ((G.min(axis=0) <= 0) & (G.max(axis=0) >= 0)).all():
                    continue
                if any(_in_triangle(P, rho) for rho in avoid):
                    continue
                start = (1 - INSIDE_STEP) * _linear_zero(P, G) + INSIDE_STEP * centre
                rho = _newton(field, start, xtol, rest_tol)
                if rho is not None:
                    yield rho


def _linear_zero(P, G):
    """Zero of the linear interpolation of G over the triangle P, clipped to the triangle."""
    M = (G[1:] - G[0]).T
    try:
        lam = np.linalg.solve(M, -G[0])
        weights = np.clip(np.array([1.0 - lam.sum(), lam[0], lam[1]]), 0.0, None)
    except np.linalg.LinAlgError:
        weights = np.ones(3)
    if not np.isfinite(weights).all() or weights.sum() == 0:
        weights = np.ones(3)
    return weights / weights.sum() @ P


def _in_triangle(P, rho, tol=1e-9):
    lam, *_ = np.linalg.lstsq((P[1:] - P[0]).T, rho - P[0], rcond=None)
    weights = np.array([1.0 - lam.sum(), lam[0], lam[1]])
    return weights.min() >= -tol and np.abs(weights @ P - rho).max() <= tol


def _newton(field, rho, xtol, rest_tol):
    """Damped Newton's method for pi_1 = pi_2 = pi_3 inside the simplex, in the
    coordinates (rho_1, rho_2). None if it does not get the payoffs to agree."""

    def residual(rho):
        return _relative_payoffs(rho, field.velocity(rho))

    def jacobian(rho, F):
        J = np.empty((2, 2))
        for j in range(2):
            d = np.zeros(3)
            d[j], d[2] = 1.0, -1.0
            h = FD_STEP if rho[2] > FD_STEP else -min(FD_STEP, rho[j] / 2)
            J[:, j] = (residual(rho + h * d) - F) / h
        return J

    F = residual(rho)
    for _ in range(MAX_NEWTON):
        if np.abs(F).max() <= 1e-15:
            break
        J = jacobian(rho, F)
        try:
            step = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            break
        step = np.array([step[0], step[1], -step.sum()])
        scale = 1.0
        while scale > 1e-6:
            trial = rho + scale * step
            if trial.min() > 0:
                F_trial = residual(trial)
                if np.abs(F_trial).max() < np.abs(F).max():
                    break
            scale /= 2
        else:
            break
        rho, F = trial, F_trial
        if np.abs(scale * step).max() <= xtol:
            break
    if np.abs(F).max() > rest_tol:
        return None
    J = jacobian(rho, F)
    if abs(np.linalg.det(J)) <= SINGULAR_TOL * (J ** 2).sum():
        return None                     # on a curve of rest points, not an isolated one
    return rho
