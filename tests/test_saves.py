"""Saving a run under a name, loading it again, and the layout of the file."""

import sqlite3

import numpy as np
import pytest

from gui import saves, spec
from gui.compute import Run
from gui.pages.parameters import experiment_from_inputs
from gui.saves import SaveError


def small_run(n=3):
    strategies = [spec.preset_strategy("AllD-Disc"), spec.preset_strategy("Defectors")]
    shares = ["0.5", "0.5"]
    if n == 3:
        strategies.append({"type": "phi", "name": "phi", "P": None, "tau": None, "phi": 1.0})
        shares = ["0.4", "0.3", "0.3"]
    experiment, error = experiment_from_inputs(
        strategies, shares, [{"score": 0, "mass": 0.5}, {"score": 1, "mass": 0.5}], "binary", None,
        list("00000-0+"), list("00000000"), "0.1", "0.4", "0.6", "0.5", "2", "1", "0.5", "6", "60", "1e-5", "3")
    assert error == ""
    run = Run(experiment)
    run.compute()
    assert run.state == "done"
    return run


@pytest.fixture(scope="module", params=[2, 3])
def run(request):
    return small_run(request.param)


def test_round_trip_restores_the_numbers(run, tmp_path):
    path = saves.save_run(run, "my first run", tmp_path)
    assert path.name == "my_first_run.sqlite"
    loaded = saves.load_run(path)
    assert loaded.state == "done" and loaded.experiment == run.experiment and loaded.labels == run.labels
    np.testing.assert_array_equal(loaded.grid, run.grid)
    for a, b in zip(loaded.grid_results, run.grid_results):
        for field in ("rho", "velocity", "payoffs", "payoff_matrix", "help_matrix"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        assert a.w.shape == b.w.shape and np.abs(a.w - b.w).max() < 1e-12     # the far tail is stored as zero
        assert a.converged == b.converged
    assert [(p.kind, p.location, p.converged) for p in loaded.rest_points] == \
           [(p.kind, p.location, p.converged) for p in run.rest_points]
    for a, b in zip(loaded.rest_points, run.rest_points):
        np.testing.assert_array_equal(a.rho, b.rho)
        np.testing.assert_allclose(a.eigenvalues, b.eigenvalues)
    t, u = loaded.trajectory, run.trajectory
    for field in ("t", "rho", "payoffs", "scores_converged"):
        np.testing.assert_array_equal(getattr(t, field), getattr(u, field))
    np.testing.assert_allclose(t.w0, u.w0, atol=1e-12)
    np.testing.assert_allclose(t.mean_score, u.mean_score, atol=1e-5)
    assert t.at_rest == u.at_rest and len(t.w) == len(u.w)
    for k in (0, len(t.t) - 1):
        for key in ("rho", "payoffs", "payoff_matrix", "help_matrix"):
            np.testing.assert_array_equal(np.asarray(loaded.frame(k)[key]), np.asarray(run.frame(k)[key]))
        assert np.abs(loaded.frame(k)["w"] - run.frame(k)["w"]).max() < 1e-12
    assert len(loaded.figure.data) == len(run.figure.data)


def test_a_loaded_run_still_computes_new_points(run, tmp_path):
    loaded = saves.load_run(saves.save_run(run, "again", tmp_path))
    rho = np.full(run.n, 1 / run.n)
    np.testing.assert_allclose(loaded.point(rho)["payoffs"], run.point(rho)["payoffs"], atol=1e-12)


def test_the_file_is_plain_tables_that_can_be_queried(run, tmp_path):
    path = saves.save_run(run, "queries", tmp_path)
    db = sqlite3.connect(path)
    n = run.n
    tables = {r[0] for r in db.execute("select name from sqlite_master where type = 'table'")}
    assert {"run", "settings", "strategy", "initial_scores", "points", "payoff_matrix", "help_matrix", "distribution",
            "score_distribution", "rest_points"} <= tables
    grid, steps = len(run.grid_results), len(run.trajectory.t)
    assert db.execute("select count(*) from points where kind = 'grid'").fetchone()[0] == grid
    assert db.execute("select count(*) from points where kind = 'trajectory'").fetchone()[0] == steps
    assert db.execute("select count(*) from payoff_matrix").fetchone()[0] == (grid + steps) * n * n
    assert db.execute("select count(*) from help_matrix").fetchone()[0] == (grid + steps) * n * n
    assert db.execute("select count(*) from distribution").fetchone()[0] == (grid + steps) * n
    # the shares of every point add to 1, and the vertices are in the index
    sums = [r[0] for r in db.execute("select share_1 + share_2 + coalesce(share_3, 0) from points")]
    np.testing.assert_allclose(sums, 1.0, atol=1e-9)
    vertices = db.execute("select count(*) from points where kind = 'grid' and (share_1 = 1 or share_2 = 1)").fetchone()
    assert vertices[0] >= 2
    # the views
    first = db.execute("select point_id, strategy, share, payoff, mean_score, share_at_score_0 from point_strategy "
                       "where kind = 'trajectory' and step = 0 order by strategy").fetchall()
    assert len(first) == n and abs(sum(r[2] for r in first) - 1) < 1e-9
    frame = run.frame(0)
    np.testing.assert_allclose([r[3] for r in first], frame["payoffs"])
    np.testing.assert_allclose([r[4] for r in first], [np.arange(len(x)) @ x for x in frame["w"]], atol=1e-5)
    np.testing.assert_allclose([r[5] for r in first], [x[0] for x in frame["w"]], atol=1e-12)
    # a score distribution stops where the tail starts, and the full length is recorded
    n_scores, tail_start, stored = db.execute(
        "select d.n_scores, d.tail_start, (select count(*) from score_distribution s "
        "where s.point_id = d.point_id and s.strategy = d.strategy) from distribution d order by n_scores desc limit 1"
    ).fetchone()
    assert stored == tail_start <= n_scores
    assert db.execute("select min(probability) from score_distribution where score = 0").fetchone()[0] >= 0
    assert dict(db.execute("select key, value from settings"))["noise.eps"] == "0.1"


def test_the_tail_below_the_cutoff_is_not_stored(tmp_path):
    info, rows = saves._distribution_rows(7, [[0.5, 0.3, 0.2, 1e-13, 5e-13, 0.0], [1.0, 0, 0, 0, 0, 0]])
    assert info == [(7, 1, 6, 3), (7, 2, 6, 1)]
    assert [r[2] for r in rows if r[1] == 1] == [0, 1, 2]


def test_names(run, tmp_path):
    saves.save_run(run, "  A/B: test!  ", tmp_path)
    assert (tmp_path / "A_B_test.sqlite").exists()
    with pytest.raises(SaveError, match="already a save"):
        saves.save_run(run, "A/B: test!", tmp_path)
    for bad in ("", "   ", "...", "x" * 61, None):
        with pytest.raises(SaveError):
            saves.file_name(bad)


def test_listing_and_deleting(run, tmp_path):
    assert saves.list_saves(tmp_path / "nothing yet") == []
    saves.save_run(run, "one", tmp_path)
    saves.save_run(run, "two", tmp_path)
    (tmp_path / "junk.sqlite").write_bytes(b"not a save")
    (tmp_path / "notes.txt").write_text("hello")
    listed = saves.list_saves(tmp_path)
    assert sorted(s["name"] for s in listed) == ["one", "two"]
    assert all("AllD-Disc" in s["label"] for s in listed)
    saves.delete_save("one.sqlite", tmp_path)
    assert [s["name"] for s in saves.list_saves(tmp_path)] == ["two"]
    with pytest.raises(SaveError):
        saves.delete_save("one.sqlite", tmp_path)
    with pytest.raises(SaveError):
        saves.delete_save("../notes.txt", tmp_path)                      # stays inside the folder
    assert (tmp_path / "notes.txt").exists()


def test_only_finished_runs_can_be_saved(tmp_path):
    unfinished = Run(small_run(2).experiment)
    with pytest.raises(SaveError, match="finished"):
        saves.save_run(unfinished, "early", tmp_path)
    with pytest.raises(SaveError, match="finished"):
        saves.save_run(None, "none", tmp_path)
    assert list(tmp_path.iterdir()) == []                                 # nothing half-written is left


def test_download_and_upload_between_folders(run, tmp_path):
    here, there = tmp_path / "here", tmp_path / "there"
    saves.save_run(run, "travelling", here)
    content = saves.read_file("travelling.sqlite", here)
    assert saves.import_file(content, there) == "travelling.sqlite"
    assert saves.list_saves(there)[0]["name"] == "travelling"
    np.testing.assert_array_equal(saves.load_run(there / "travelling.sqlite").grid, run.grid)
    with pytest.raises(SaveError, match="already a save"):
        saves.import_file(content, there)                                 # a taken name is refused
    assert sorted(p.name for p in there.iterdir()) == ["travelling.sqlite"]   # no leftovers


def test_csv_export(run, tmp_path):
    import csv
    import io
    import zipfile

    saves.save_run(run, "export", tmp_path)
    z = zipfile.ZipFile(io.BytesIO(saves.export_csv("export.sqlite", tmp_path)))
    assert sorted(z.namelist()) == sorted(f"{t}.csv" for t in saves.EXPORTED)
    rows = list(csv.DictReader(io.StringIO(z.read("points.csv").decode())))
    assert len(rows) == len(run.grid_results) + len(run.trajectory.t) and "share_1" in rows[0]
    assert len(list(csv.DictReader(io.StringIO(z.read("point_strategy.csv").decode())))) == len(rows) * run.n


def test_files_that_are_not_saves_are_refused(run, tmp_path):
    (tmp_path / "bad.sqlite").write_bytes(b"not a save")
    with pytest.raises(SaveError, match="can't be read"):
        saves.load_run(tmp_path / "bad.sqlite")
    other = sqlite3.connect(tmp_path / "foreign.sqlite")                  # a database of someone else's
    other.execute("create table x (a)")
    other.commit()
    other.close()
    with pytest.raises(SaveError, match="isn't a saved result"):
        saves.load_run(tmp_path / "foreign.sqlite")
    path = saves.save_run(run, "old", tmp_path)
    db = sqlite3.connect(path)
    db.execute("pragma user_version = 99")
    db.commit()
    db.close()
    with pytest.raises(SaveError, match="different version"):
        saves.load_run(path)
    db = sqlite3.connect(saves.save_run(run, "damaged", tmp_path))
    db.execute("drop view point_strategy")
    db.execute("drop table help_matrix")
    db.commit()
    db.close()
    with pytest.raises(SaveError, match="incomplete"):
        saves.load_run(tmp_path / "damaged.sqlite")
    with pytest.raises(SaveError):
        saves.import_file(b"x" * 100, tmp_path / "up")
    assert not any(p.name.endswith(".partial") for p in (tmp_path / "up").iterdir())
