"""The examples of docs/engine/engine.tex all run, and print something."""

import sys
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs" / "engine"
sys.path.insert(0, str(DOCS))

from examples import EXAMPLES  # noqa: E402
from run_examples import run   # noqa: E402


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_example_runs(name):
    output, _ = run(name)
    assert output.strip()


def test_every_example_is_in_the_document():
    text = "".join(path.read_text() for path in (DOCS / "parts").glob("*.tex"))
    for name in EXAMPLES:
        assert "\\ex{" + name + "}" in text, f"example {name} is not used in the document"
