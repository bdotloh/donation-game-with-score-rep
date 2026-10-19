"""docs/verification/compare_published.py (Check 22 of the verification report)."""

import sys
from pathlib import Path

import pytest

from engine import ScoringNorm, strategy_from_name
from plots.ess_data import EssRow

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs" / "verification"))
from compare_published import compare  # noqa: E402


def test_comparison_with_a_published_table():
    def row(barcode, strategy, cooperation):
        return EssRow(ScoringNorm.from_barcode(barcode), strategy_from_name(strategy), cooperation)

    a, b, c = "CN:00000-00 NC:00000000", "CN:+0000-00 NC:00000000", "CN:0+000-00 NC:00000000"
    engine = [row(a, "0,0|0,g", 0.5), row(b, "0,0|0,g", 0.7), row(c, "0,0|0,g", 0.2)]
    paper = [row(a, "0,0|0,g", 0.5004), row(b, "0,0|0,g", 0.7), row(a, "g,0|g,g", 0.9)]
    result = compare(engine, paper, threshold=0.25)
    assert (result["published"], result["engine"], result["common"]) == (3, 3, 2)
    assert result["missing"] == [(ScoringNorm.from_barcode(a).barcode, "g,0|g,g")]
    assert result["extra"] == [(ScoringNorm.from_barcode(c).barcode, "0,0|0,g")] and result["extra_above_threshold"] == []
    assert result["max_difference"] == pytest.approx(4e-4) and result["published_norms"] == 2
