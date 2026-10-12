"""Compare the engine's ESS with every row of ess_sweep.tsv (find_ESS in helper_funcs.jl).

    python tests/reference/compare_ess.py

tests/test_against_julia.py checks a random sample of the rows; this checks all of them.
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_against_julia import ESS_ROWS, find_ess  # noqa: E402


def check(row):
    expected = row["ess"].split(";") if row["ess"] else []
    found = find_ess(row["barcode"], float(row["noise"]))
    return row["barcode"], row["noise"], expected, found


if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(check, ESS_ROWS, chunksize=8))
    mismatches = [r for r in results if r[2] != r[3]]
    for barcode, noise, expected, found in mismatches:
        print(f"{barcode} noise={noise}: julia {expected} engine {found}")
    n_ess = sum(len(r[2]) for r in results)
    print(f"{len(results) - len(mismatches)}/{len(results)} rows agree ({n_ess} ESS in the reference).")
    sys.exit(1 if mismatches else 0)
