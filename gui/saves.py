"""Saving finished runs under a name and loading them again, without computing anything.

A save is one SQLite file in the saves folder (saved_results/ in the directory the GUI is started in, or
$DONATION_GAME_SAVES), laid out as plain tables so it can be read from anywhere:

    run                the name, the date, the number of strategies and the settings as JSON (one row)
    settings           every setting as key / value
    strategy           the strategies: number (1, 2, 3), label (S1, ...), name, description
    initial_scores     the initial score distribution
    points             one row per point of the arrow grid on the simplex: kind ('grid'), the shares share_1..,
                       the velocity dRho/dt velocity_1.., the payoffs payoff_1.., and whether the score chains
                       converged. (Saves made before evolutions were separate from running also hold the
                       trajectory, kind 'trajectory'; it is loaded when it is there, and no longer written.)
    payoff_matrix      (point_id, strategy, against, value)       A[i, k], pi = A rho
    help_matrix        (point_id, donor, receiver, value)         H[i, k]
    distribution       (point_id, strategy, n_scores, tail_start) how long the score distribution was, and how
                       many of its scores are stored
    score_distribution (point_id, strategy, score, probability)   scores 0 .. tail_start - 1
    rest_points        shares, kind, location and the eigenvalues of the Jacobian

and the views mean_score and point_strategy (one row per point and strategy, with share, payoff, velocity, mean
score and the share at score 0). The scores of a distribution from tail_start on, where every probability is
below 1e-12, are not stored and read back as 0; with n_scores they restore the full vector. Loading rebuilds the
run from its settings and puts the saved numbers back, so clicking a new point on the simplex still computes it.
A file is only ever read with fixed queries, so an uploaded one can't run anything.

From Python:  pandas.read_sql("select * from point_strategy where kind = 'grid'", sqlite3.connect(file)).
"""

import csv
import io
import json
import os
import re
import sqlite3
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

from engine.replicator import Trajectory
from engine.simplex import PointResult, RestPoint

from . import spec
from .compute import Run

FORMAT = 1
APPLICATION_ID = 0x44475352            # "DGSR"
MAX_NAME = 60
MAX_FILE_BYTES = 300 * 1024 * 1024
TAIL = 1e-12                            # probabilities below this at the end of a distribution are not stored
MAX_STRATEGIES = 3
EXPORTED = ["run", "settings", "strategy", "initial_scores", "points", "payoff_matrix", "help_matrix", "distribution",
            "score_distribution", "rest_points", "mean_score", "point_strategy"]


class SaveError(ValueError):
    """A problem with a save that is worth showing to the user as it is."""


def _columns(prefix, kind="REAL"):
    return ", ".join(f"{prefix}_{i} {kind}" for i in range(1, MAX_STRATEGIES + 1))


def _pick(column):
    return f"CASE s.strategy WHEN 1 THEN p.{column}_1 WHEN 2 THEN p.{column}_2 ELSE p.{column}_3 END"


SCHEMA = f"""
CREATE TABLE run (name TEXT NOT NULL, saved TEXT NOT NULL, format INTEGER NOT NULL, n_strategies INTEGER NOT NULL,
                  tail_cutoff REAL NOT NULL, experiment TEXT NOT NULL);
CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE strategy (strategy INTEGER PRIMARY KEY, label TEXT NOT NULL, name TEXT NOT NULL,
                       description TEXT NOT NULL);
CREATE TABLE initial_scores (score INTEGER PRIMARY KEY, mass REAL NOT NULL);
CREATE TABLE points (point_id INTEGER PRIMARY KEY, kind TEXT NOT NULL CHECK (kind IN ('grid', 'trajectory')),
                     step INTEGER, t REAL, {_columns("share")}, {_columns("velocity")}, {_columns("payoff")},
                     converged INTEGER NOT NULL);
CREATE TABLE payoff_matrix (point_id INTEGER NOT NULL, strategy INTEGER NOT NULL, against INTEGER NOT NULL,
                            value REAL NOT NULL, PRIMARY KEY (point_id, strategy, against)) WITHOUT ROWID;
CREATE TABLE help_matrix (point_id INTEGER NOT NULL, donor INTEGER NOT NULL, receiver INTEGER NOT NULL,
                          value REAL NOT NULL, PRIMARY KEY (point_id, donor, receiver)) WITHOUT ROWID;
CREATE TABLE distribution (point_id INTEGER NOT NULL, strategy INTEGER NOT NULL, n_scores INTEGER NOT NULL,
                           tail_start INTEGER NOT NULL, PRIMARY KEY (point_id, strategy)) WITHOUT ROWID;
CREATE TABLE score_distribution (point_id INTEGER NOT NULL, strategy INTEGER NOT NULL, score INTEGER NOT NULL,
                                 probability REAL NOT NULL, PRIMARY KEY (point_id, strategy, score)) WITHOUT ROWID;
CREATE TABLE rest_points (rest_id INTEGER PRIMARY KEY, {_columns("share")}, kind TEXT NOT NULL,
                          location TEXT NOT NULL, converged INTEGER NOT NULL,
                          eigenvalue_1_re REAL, eigenvalue_1_im REAL, eigenvalue_2_re REAL, eigenvalue_2_im REAL);
CREATE VIEW mean_score AS
    SELECT point_id, strategy, SUM(score * probability) AS mean_score FROM score_distribution
    GROUP BY point_id, strategy;
CREATE VIEW point_strategy AS
    SELECT p.point_id, p.kind, p.step, p.t, s.strategy, s.label, s.name,
           {_pick("share")} AS share, {_pick("payoff")} AS payoff, {_pick("velocity")} AS velocity,
           m.mean_score,
           (SELECT d.probability FROM score_distribution d
             WHERE d.point_id = p.point_id AND d.strategy = s.strategy AND d.score = 0) AS share_at_score_0
    FROM points p CROSS JOIN strategy s
    LEFT JOIN mean_score m ON m.point_id = p.point_id AND m.strategy = s.strategy;
"""


def default_folder():
    return Path(os.environ.get("DONATION_GAME_SAVES", "saved_results"))


def file_name(name):
    """The file a name is saved in; SaveError if the name can't be used."""
    name = (name or "").strip()
    if not name:
        raise SaveError("Give the save a name.")
    if len(name) > MAX_NAME:
        raise SaveError(f"The name can be at most {MAX_NAME} characters.")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not slug:
        raise SaveError("The name needs at least one letter or digit.")
    return f"{slug}.sqlite"


def _flat(prefix, value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flat(f"{prefix}.{key}" if prefix else key, item)
    elif not isinstance(value, (list, tuple)):
        yield prefix, "" if value is None else str(value)


def _share_columns(rho):
    return [float(rho[i]) if i < len(rho) else None for i in range(MAX_STRATEGIES)]


def _distribution_rows(point_id, w):
    """Rows for the distribution table and the score_distribution table of one point (w: strategies x scores)."""
    info, scores = [], []
    for i, row in enumerate(np.asarray(w, dtype=float)):
        above = np.flatnonzero(row >= TAIL)
        stored = int(above.max()) + 1 if len(above) else 1
        info.append((point_id, i + 1, len(row), stored))
        scores.extend((point_id, i + 1, j, float(row[j])) for j in range(stored))
    return info, scores


def save_run(run, name, folder=None):
    """Write a finished run to <folder>/<name>.sqlite and return the path. SaveError for an unfinished run or a
    bad or taken name."""
    if run is None or run.state != "done":
        raise SaveError("There is no finished result to save.")
    folder = Path(folder or default_folder())
    path = folder / file_name(name)
    if path.exists():
        raise SaveError(f"There is already a save called “{name.strip()}”. Pick another name.")
    folder.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".sqlite.partial")
    temp.unlink(missing_ok=True)
    try:
        _write(run, name.strip(), temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def _write(run, name, path):
    experiment, n = run.experiment, run.n
    with sqlite3.connect(path) as db:
        db.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        db.execute(f"PRAGMA user_version = {FORMAT}")
        db.executescript(SCHEMA)
        db.execute("INSERT INTO run VALUES (?, ?, ?, ?, ?, ?)",
                   (name, time.strftime("%Y-%m-%d %H:%M:%S"), FORMAT, n, TAIL, json.dumps(experiment)))
        db.executemany("INSERT INTO settings VALUES (?, ?)", list(_flat("", {
            k: v for k, v in experiment.items() if k not in ("strategies", "shares", "scores", "version")})))
        db.executemany("INSERT INTO strategy VALUES (?, ?, ?, ?)", [
            (i + 1, run.labels[i], cfg["name"], spec.describe_strategy(cfg))
            for i, cfg in enumerate(experiment["strategies"])])
        db.executemany("INSERT INTO initial_scores VALUES (?, ?)",
                       [(int(e["score"]), float(e["mass"])) for e in experiment["scores"]])

        point_id = 0

        def add_point(kind, step, t, rho, velocity, payoffs, converged, payoff_matrix, help_matrix, w):
            nonlocal point_id
            point_id += 1
            db.execute(f"INSERT INTO points VALUES ({', '.join('?' * 14)})",
                       (point_id, kind, step, t, *_share_columns(rho), *_share_columns(velocity),
                        *_share_columns(payoffs), int(converged)))
            db.executemany("INSERT INTO payoff_matrix VALUES (?, ?, ?, ?)",
                           [(point_id, i + 1, k + 1, float(payoff_matrix[i][k])) for i in range(n) for k in range(n)])
            db.executemany("INSERT INTO help_matrix VALUES (?, ?, ?, ?)",
                           [(point_id, i + 1, k + 1, float(help_matrix[i][k])) for i in range(n) for k in range(n)])
            info, scores = _distribution_rows(point_id, w)
            db.executemany("INSERT INTO distribution VALUES (?, ?, ?, ?)", info)
            db.executemany("INSERT INTO score_distribution VALUES (?, ?, ?, ?)", scores)

        for r in run.grid_results:
            add_point("grid", None, None, r.rho, r.velocity, r.payoffs, r.converged, r.payoff_matrix, r.help_matrix,
                      r.w)
        for rest_id, p in enumerate(run.rest_points, start=1):
            eig = [(float(np.real(e)), float(np.imag(e))) for e in np.atleast_1d(p.eigenvalues)][:2]
            eig += [(None, None)] * (2 - len(eig))
            db.execute("INSERT INTO rest_points VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (rest_id, *_share_columns(p.rho), p.kind, p.location, int(p.converged),
                        eig[0][0], eig[0][1], eig[1][0], eig[1][1]))
        db.execute("INSERT INTO settings VALUES ('grid_points', ?)", (str(len(run.grid_results)),))
    return path


def _connect(path):
    """A read-only connection to a save, checked to be one of ours."""
    path = Path(path)
    try:
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        db.execute("PRAGMA query_only = ON")
        if db.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
            raise SaveError("That file isn't a saved result of this program.")
        if db.execute("PRAGMA user_version").fetchone()[0] != FORMAT:
            raise SaveError("This save was made by a different version of the program.")
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        missing = set(EXPORTED[:9]) - tables
        if missing:
            raise SaveError(f"That save is incomplete (no table {', '.join(sorted(missing))}).")
        return db
    except SaveError:
        raise
    except Exception as error:  # noqa: BLE001  (a damaged or foreign file)
        raise SaveError(f"That file can't be read as a saved result ({type(error).__name__}).") from error


def read_meta(path):
    db = _connect(path)
    try:
        name, saved, n = db.execute("SELECT name, saved, n_strategies FROM run").fetchone()
        strategies = [r[0] for r in db.execute("SELECT name FROM strategy ORDER BY strategy")]
    finally:
        db.close()
    return {"name": name, "saved": saved, "n_strategies": n, "strategies": strategies}


def load_run(path):
    """A finished Run from a save; SaveError if the file can't be used."""
    db = _connect(path)
    try:
        experiment_json, n = db.execute("SELECT experiment, n_strategies FROM run").fetchone()
        columns = [row[1] for row in db.execute("PRAGMA table_info(run)")]
        at_rest = db.execute("SELECT at_rest FROM run").fetchone()[0] if "at_rest" in columns else 0   # older saves
        experiment = json.loads(experiment_json)
        error = spec.check(experiment)
        if error:
            raise SaveError(error)
        run = Run(experiment)
        if run.n != n:
            raise SaveError("That save is damaged (the number of strategies doesn't match).")
        lengths = {(p, s): (length, stored) for p, s, length, stored in
                   db.execute("SELECT point_id, strategy, n_scores, tail_start FROM distribution")}
        w = {}
        for p, s, j, probability in db.execute("SELECT point_id, strategy, score, probability FROM score_distribution"):
            if (p, s) not in w:
                w[(p, s)] = np.zeros(lengths[(p, s)][0])
            w[(p, s)][j] = probability

        def matrices(table, a, b):
            out = {}
            for p, i, k, value in db.execute(f"SELECT point_id, {a}, {b}, value FROM {table}"):
                out.setdefault(p, np.zeros((n, n)))[i - 1, k - 1] = value
            return out

        payoff_matrix, help_matrix = matrices("payoff_matrix", "strategy", "against"), matrices("help_matrix", "donor", "receiver")

        def vector(row, offset):
            return np.array(row[offset:offset + n], dtype=float)

        def distributions(p):
            return np.array([w[(p, i + 1)] for i in range(n)])

        grid, grid_results, traj_rows = [], [], []
        for row in db.execute("SELECT * FROM points ORDER BY point_id"):
            p, kind, step, t = row[:4]
            rho, velocity, payoffs, converged = vector(row, 4), vector(row, 7), vector(row, 10), bool(row[13])
            if kind == "grid":
                grid.append(rho)
                grid_results.append(PointResult(rho=rho, velocity=velocity, payoffs=payoffs,
                                                payoff_matrix=payoff_matrix[p], help_matrix=help_matrix[p],
                                                w=distributions(p), converged=converged))
            else:
                traj_rows.append((step, t, rho, payoffs, converged, p))
        traj_rows.sort(key=lambda r: r[0])
        ws = [distributions(r[5]) for r in traj_rows]
        run.grid = np.array(grid)
        run.grid_results = grid_results
        run.rest_points = []
        for row in db.execute("SELECT * FROM rest_points ORDER BY rest_id"):
            eig = [complex(re, im) if im else re for re, im in ((row[7], row[8]), (row[9], row[10]))
                   if re is not None][: max(0, n - 1)]
            run.rest_points.append(RestPoint(rho=vector(row, 1), kind=row[4], location=row[5],
                                             eigenvalues=np.array(eig), converged=bool(row[6])))
        if traj_rows:                                    # only in saves made before evolutions were separate
            run.trajectory = Trajectory(
                t=np.array([r[1] for r in traj_rows]), rho=np.array([r[2] for r in traj_rows]),
                payoffs=np.array([r[3] for r in traj_rows]),
                w0=np.array([[x[0] for x in wk] for wk in ws]),
                mean_score=np.array([[np.arange(len(x)) @ x for x in wk] for wk in ws]),
                scores_converged=np.array([r[4] for r in traj_rows], dtype=bool), strategies=run.strategies,
                at_rest=bool(at_rest), w=ws)
            run.trajectory_H = [help_matrix[r[5]] for r in traj_rows]
            run.trajectory_A = [payoff_matrix[r[5]] for r in traj_rows]
    except SaveError:
        raise
    except Exception as error:  # noqa: BLE001  (a damaged file)
        raise SaveError(f"That save is damaged ({type(error).__name__}: {error}).") from error
    finally:
        db.close()
    run.draw()
    run.progress, run.message, run.state = 1.0, "Loaded", "done"
    return run


def list_saves(folder=None):
    """[{"file", "name", "saved", "label"}] of the readable saves, newest first."""
    folder = Path(folder or default_folder())
    found = []
    for path in folder.glob("*.sqlite") if folder.is_dir() else []:
        try:
            meta = read_meta(path)
        except SaveError:
            continue
        found.append({"file": path.name, "name": meta["name"], "saved": meta["saved"],
                      "label": f"{meta['name']} · {', '.join(meta['strategies'])} · {meta['saved']}"})
    return sorted(found, key=lambda s: s["saved"], reverse=True)


def _path(file, folder):
    path = Path(folder or default_folder()) / Path(file).name          # never outside the folder
    if path.suffix != ".sqlite" or not path.is_file():
        raise SaveError("That save is not there any more.")
    return path


def delete_save(file, folder=None):
    _path(file, folder).unlink()


def rename_save(file, new_name, folder=None):
    """Give a save a new name (the file is renamed to match); returns the new file name. SaveError if the name
    can't be used or belongs to another save."""
    folder = Path(folder or default_folder())
    path = _path(file, folder)
    new_name = (new_name or "").strip()
    target = folder / file_name(new_name)
    if target != path and target.exists():
        raise SaveError(f"There is already a save called “{new_name}”. Pick another name.")
    old_name = read_meta(path)["name"]
    _set_name(path, new_name)
    if target != path:
        try:
            path.replace(target)
        except OSError as error:
            _set_name(path, old_name)
            raise SaveError(f"The file couldn't be renamed ({error.strerror or type(error).__name__}).") from error
    return target.name


def _set_name(path, name):
    db = sqlite3.connect(path)
    try:
        with db:
            db.execute("UPDATE run SET name = ?", (name,))
    finally:
        db.close()


def read_file(file, folder=None):
    """The bytes of a save, to download."""
    return _path(file, folder).read_bytes()


def export_csv(file, folder=None):
    """A zip with one CSV per table (and the two views) of a save."""
    db = _connect(_path(file, folder))
    buffer = io.BytesIO()
    try:
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
            for table in EXPORTED:
                cursor = db.execute(f"SELECT * FROM {table}")
                text = io.StringIO()
                writer = csv.writer(text)
                writer.writerow([c[0] for c in cursor.description])
                writer.writerows(cursor)
                z.writestr(f"{table}.csv", text.getvalue())
    finally:
        db.close()
    return buffer.getvalue()


def import_file(content, folder=None):
    """Add an uploaded save to the folder under the name it was saved with; returns the file name.
    SaveError if it isn't a save, can't be loaded or its name is taken."""
    if len(content) > MAX_FILE_BYTES:
        raise SaveError(f"That file is too big (the limit is {MAX_FILE_BYTES // 2 ** 20} MB).")
    folder = Path(folder or default_folder())
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".sqlite.partial", delete=False) as handle:
        handle.write(content)
        temp = Path(handle.name)
    try:
        meta = read_meta(temp)
        target = folder / file_name(meta["name"])
        if target.exists():
            raise SaveError(f"There is already a save called “{meta['name']}”. Delete it or rename one of them first.")
        load_run(temp)                                   # the whole file has to be usable
        temp.replace(target)
        return target.name
    finally:
        temp.unlink(missing_ok=True)
