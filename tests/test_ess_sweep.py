"""scripts/ess_sweep.py: norm sets, splitting into tasks, resuming, and agreement with helper_funcs.jl."""

import csv
import importlib.util
import json
import random
from pathlib import Path

import pytest

from engine import ScoringNorm, TernaryStrategy, all_binary_strategies, all_norms, all_ternary_strategies, strategy_from_name

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ess_sweep", ROOT / "scripts" / "ess_sweep.py")
ess_sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ess_sweep)


def test_norm_sets():
    assert len(ess_sweep.NormSet("binary-cn")) == 1296
    assert len(ess_sweep.NormSet("binary")) == 1296 ** 2
    assert len(ess_sweep.NormSet("ternary-cn", tau=5)) == 2 ** 6 * 3 ** 12
    binary_cn = ess_sweep.NormSet("binary-cn")
    codes = {binary_cn.barcode(i) for i in range(len(binary_cn))}
    assert codes == {n.barcode for n in all_norms(use_NC=False)}
    ternary = ess_sweep.NormSet("ternary", tau=2)
    for i in random.Random(0).sample(range(len(ternary)), 50):
        norm = ternary.norm(i)                           # every index decodes to a valid ternary norm
        assert norm.is_ternary and norm.partition.tau == 2
    lifted = ess_sweep.NormSet("lifted-cn", tau=3)
    assert lifted.norm(5) == ScoringNorm.from_barcode(binary_cn.barcode(5)).to_ternary(3)


@pytest.mark.parametrize("sample", [None, 37])
def test_tasks_cover_every_norm_once(sample):
    tasks = [ess_sweep.task_indices(100, k, 7, sample, seed=3) for k in range(7)]
    flat = [i for t in tasks for i in t]
    assert len(flat) == len(set(flat)) == (sample or 100)


def test_slurm_environment(monkeypatch):
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "12")
    monkeypatch.setenv("SLURM_ARRAY_TASK_MIN", "10")
    monkeypatch.setenv("SLURM_ARRAY_TASK_MAX", "19")
    assert ess_sweep.slurm_task() == (2, 10)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID")
    assert ess_sweep.slurm_task() == (0, 1)


def test_strategy_names_round_trip():
    for s in all_binary_strategies() + all_ternary_strategies(4)[::37]:
        assert strategy_from_name(s.name).name == s.name
    assert strategy_from_name("phi:0.17").phi == 0.17
    assert isinstance(strategy_from_name("0,0,0|0,g,0|0,g,0", tau=5), TernaryStrategy)


def _julia_rows():
    rows = [r for r in csv.DictReader(open(ROOT / "tests/reference/ess_sweep.tsv"), delimiter="\t")
            if r["noise"] == "0.0" and r["barcode"].endswith("NmeetsC:00000000")]
    return random.Random(1).sample(rows, 25)


def test_run_resume_merge_and_julia_agreement(tmp_path):
    rows = _julia_rows()
    norm_file = tmp_path / "norms.txt"
    norm_file.write_text("# 25 CN norms\n" + "\n".join(r["barcode"] for r in rows) + "\n")
    out = tmp_path / "out"
    base = ["run", "--norms", f"file:{norm_file}", "--out", str(out), "--workers", "1"]
    for task in range(2):                                # two array tasks, the first one interrupted
        ess_sweep.main(base + ["--task", str(task), "--n-tasks", "2"] + (["--limit", "5"] if task == 0 else []))
    ess_sweep.main(base + ["--task", "0", "--n-tasks", "2"])          # resumes task 0
    done = (out / "done_task0000.txt").read_text().splitlines() + (out / "done_task0001.txt").read_text().splitlines()
    assert len(done) == len({line.split("\t")[0] for line in done}) == 25
    ess_sweep.main(["merge", str(out)])
    found = {}
    for r in csv.DictReader(open(out / "ess.csv")):
        found.setdefault(r["norm"], set()).add(r["ess"])
    for r in rows:
        barcode = ScoringNorm.from_barcode(r["barcode"]).barcode
        assert found.get(barcode, set()) == (set(r["ess"].split(";")) if r["ess"] else set()), barcode
    assert json.loads((out / "config.json").read_text())["norms"] == f"file:{norm_file}"
    with pytest.raises(SystemExit):                      # other settings, same directory
        ess_sweep.main(base + ["--eps", "0.01"])
