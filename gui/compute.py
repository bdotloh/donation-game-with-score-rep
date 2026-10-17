"""Running an experiment: the vector field and rest points, and evolutions from chosen points, in background threads.

Results stay in this process (the GUI is a local, single-user app); the browser only
holds the run id. Score distributions are kept as numpy arrays here.
"""

import copy
import threading
import uuid

import numpy as np

from engine import Population
from engine.simplex import ReplicatorField, find_rest_points, simplex_grid

from . import figures, spec


class Run:
    def __init__(self, experiment):
        self.id = uuid.uuid4().hex
        self.experiment = experiment
        built = spec.build(experiment)
        self.names = built["names"]
        self.labels = [f"S{i + 1}" for i in range(len(self.names))]
        self.strategies = built["strategies"]
        self.w_init = built["w_init"]
        self.dynamics = built["dynamics"]
        self.field = ReplicatorField(self.dynamics, self.strategies, self.w_init)
        self.n = len(self.strategies)
        self.state = "running"
        self.progress = 0.0
        self.message = "Starting…"
        self.error = ""
        self.grid = self.grid_results = self.rest_points = self.figure = None
        self.trajectory = None                  # set by an evolution (or by a save made before evolutions were separate)
        self.trajectory_H = self.trajectory_A = None
        self.evolve_state = "idle"              # idle, running, done or error
        self.evolve_progress = 0.0
        self.evolve_error = ""
        self._points = {}

    # ------------------------------------------------------------ computation

    def compute(self):
        try:
            self.message = "Computing arrows…"
            resolution = self.experiment["resolution"]
            self.grid = simplex_grid(self.n, resolution if self.n == 3 else 2 * resolution)
            self.grid_results = self.field.evaluate_many(self.grid, progress=self._progress(0.0, 0.9))
            self.message = "Finding rest points…"
            velocities = [r.velocity for r in self.grid_results]
            self.rest_points = find_rest_points(self.field.velocity, self.n, grid=self.grid,
                                                grid_velocities=velocities,
                                                grid_converged=[r.converged for r in self.grid_results],
                                                neutral_tol=10 * self.experiment["tol"])
            self.progress = 0.95
            self.message = "Drawing the phase portrait…"
            self.draw()
            self.state = "done"
            self.progress = 1.0
        except Exception as error:  # shown to the user
            self.state = "error"
            self.error = f"{type(error).__name__}: {error}"

    def draw(self):
        self.figure = figures.simplex_figure(
            self.names, self.grid, [r.velocity for r in self.grid_results],
            [r.converged for r in self.grid_results], self.rest_point_dicts(),
            trajectory=None if self.trajectory is None else self.trajectory.rho,
            current=None if self.trajectory is None else self.trajectory.rho[0], labels=self.labels,
            barcode=self.barcode(), codes=[spec.strategy_code(s) for s in self.experiment["strategies"]])

    def barcode(self):
        """The scoring norm as text, e.g. CN:00000-00 NC:00000000 (with τ for a ternary norm)."""
        norm = self.experiment["norm"]
        text = f"CN:{norm['CN']} NC:{norm['NC']}"
        return text + (f"  τ = {norm['tau']}" if norm["type"] == "ternary" else "")

    def _progress(self, start, end, total=None):
        def report(done, total_=total):
            self.progress = start + (end - start) * done / max(1, total_)
        return report

    def start_evolve(self, rho, n_steps, dt):
        """Evolve the shares from rho in a thread (False if one is already running). The trajectory replaces
        the previous one when it is done."""
        if self.evolve_state == "running":
            return False
        self.evolve_state, self.evolve_progress, self.evolve_error = "running", 0.0, ""
        threading.Thread(target=self.evolve, args=(np.asarray(rho, dtype=float), n_steps, dt), daemon=True).start()
        return True

    def evolve(self, rho, n_steps, dt):
        try:
            dynamics = copy.copy(self.dynamics)
            dynamics.dt = dt

            def report(done, total):
                self.evolve_progress = done / max(1, total)

            trajectory = dynamics.run(self.strategies, rho, self.w_init, n_steps, record_scores=True, progress=report)
            matrices = self._trajectory_matrices(trajectory)
            self.trajectory, (self.trajectory_H, self.trajectory_A) = trajectory, matrices
            self.draw()
            self.evolve_state = "done"
        except Exception as error:  # shown to the user
            self.evolve_state = "error"
            self.evolve_error = f"{type(error).__name__}: {error}"

    def _trajectory_matrices(self, trajectory):
        payoffs = self.dynamics.payoffs
        helping, payoff_matrices = [], []
        for rho, w in zip(trajectory.rho, trajectory.w):
            population = Population(self.strategies, rho)
            helping.append(payoffs.help_probabilities(population, w))
            payoff_matrices.append(self.dynamics.payoff_matrix(population, w))
        return helping, payoff_matrices

    # ------------------------------------------------------------ queries

    def frame(self, k):
        """State at step k of the trajectory, in the same form as point()."""
        t = self.trajectory
        return {
            "rho": t.rho[k],
            "velocity": self._velocity(t.rho[k], t.payoffs[k]),
            "payoffs": t.payoffs[k],
            "payoff_matrix": self.trajectory_A[k],
            "help_matrix": self.trajectory_H[k],
            "w": t.w[k],
            "converged": bool(t.scores_converged[k]),
            "time": float(t.t[k]),
        }

    def point(self, rho):
        """State at any composition rho (cached)."""
        rho = np.clip(np.asarray(rho, dtype=float), 0, None)
        rho = rho / rho.sum()
        key = tuple(np.round(rho, 6))
        if key not in self._points:
            r = self.field.evaluate(rho)
            self._points[key] = {
                "rho": r.rho, "velocity": r.velocity, "payoffs": r.payoffs, "payoff_matrix": r.payoff_matrix,
                "help_matrix": r.help_matrix, "w": r.w, "converged": bool(r.converged), "time": None,
            }
        return self._points[key]

    @staticmethod
    def _velocity(rho, payoffs):
        return rho * (payoffs - rho @ payoffs)

    def rest_point_dicts(self):
        return [
            {"rho": np.asarray(p.rho).tolist(), "kind": p.kind, "location": p.location,
             "eigenvalues": [complex(e) if np.iscomplexobj(e) and np.imag(e) != 0 else float(np.real(e))
                             for e in np.atleast_1d(p.eigenvalues)],
             "converged": bool(p.converged)}
            for p in self.rest_points
        ]


class Runs:
    """Runs by id; each one computes in its own thread."""

    def __init__(self):
        self._runs = {}
        self._lock = threading.Lock()

    def start(self, experiment):
        run = Run(experiment)
        with self._lock:
            self._runs[run.id] = run
        threading.Thread(target=run.compute, daemon=True).start()
        return run.id

    def add(self, run):
        """Register a run that is already finished (one loaded from a file)."""
        with self._lock:
            self._runs[run.id] = run
        return run.id

    def get(self, run_id):
        with self._lock:
            return self._runs.get(run_id)


RUNS = Runs()
