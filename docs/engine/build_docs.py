"""Build the engine documentation: run the examples, write generated/ex_<name>.tex, compile engine.tex.

    python docs/engine/build_docs.py          # examples + PDF (needs pdflatex)
    python docs/engine/build_docs.py --no-pdf # only the example files
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from examples import EXAMPLES  # noqa: E402
from run_examples import run   # noqa: E402


def write_example(name):
    out, seconds = run(name)
    code = EXAMPLES[name]
    for text in (code, out):
        assert "\\end{lstlisting}" not in text
    body = ("\\begin{lstlisting}[style=code]\n" + code + "\\end{lstlisting}\n"
            "\\begin{lstlisting}[style=output]\n" + out.rstrip("\n") + "\n\\end{lstlisting}\n")
    (HERE / "generated" / f"ex_{name}.tex").write_text(body)
    return seconds


def main(pdf=True):
    (HERE / "generated").mkdir(exist_ok=True)
    for name in EXAMPLES:
        print(f"example {name}: {write_example(name):.1f}s")
    if pdf:
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "engine.tex"], cwd=HERE,
                           check=True, stdout=subprocess.DEVNULL)
        print("wrote", HERE / "engine.pdf")


if __name__ == "__main__":
    main(pdf="--no-pdf" not in sys.argv)
