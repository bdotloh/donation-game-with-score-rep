"""Run the documentation's examples and return what each printed."""

import contextlib
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from examples import EXAMPLES  # noqa: E402


def run(name):
    """(stdout of the example's code, seconds); the code runs in a fresh namespace."""
    buffer = io.StringIO()
    start = time.time()
    with contextlib.redirect_stdout(buffer):
        exec(compile(EXAMPLES[name], f"<example {name}>", "exec"), {"__name__": "__example__"})
    return buffer.getvalue(), time.time() - start


if __name__ == "__main__":
    for name in EXAMPLES:
        out, seconds = run(name)
        print(f"=== {name} ({seconds:.1f}s)\n{out}")
