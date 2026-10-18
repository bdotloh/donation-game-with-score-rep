"""Compare an ess_sweep.py run with a data file of the paper's Julia code (Check 22).

    python scripts/ess_sweep.py run --norms binary-cn --eps 0.01 --maxn 500 --final-maxn 500 \\
        --fitness-threshold 0.01 --out out/binary-cn      # the paper's settings
    python scripts/ess_sweep.py merge out/binary-cn
    python docs/verification/compare_published.py out/binary-cn \\
        ~/bdotloh/origin_money/indirect_reciprocity/data-default/tokens_CmeetsN_Noise0.01_B2

An ESS is identified by its norm and its strategy. Prints how many published ESS the engine finds, which it
misses, which it finds that the file does not have (with the file's own fitness threshold, 0.01, those with a
cooperation level below 0.01 cannot be in the file), and how far the cooperation levels of the common rows are apart.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plots.ess_data import read_paper_data, read_sweep  # noqa: E402


def key(row):
    return row.norm.barcode, row.strategy.name


def compare(engine_rows, paper_rows, threshold=0.01):
    engine = {key(r): r.cooperation for r in engine_rows}
    paper = {key(r): r.cooperation for r in paper_rows}
    common = sorted(set(engine) & set(paper))
    missing = sorted(set(paper) - set(engine))
    extra = sorted(set(engine) - set(paper))
    extra_above = [k for k in extra if engine[k] >= threshold]
    differences = [abs(engine[k] - paper[k]) for k in common]
    return {
        "published": len(paper), "engine": len(engine), "common": len(common), "missing": missing,
        "extra": extra, "extra_above_threshold": extra_above,
        "max_difference": max(differences, default=0.0),
        "mean_difference": sum(differences) / len(differences) if differences else 0.0,
        "published_norms": len({k[0] for k in paper}), "missing_norms": len({k[0] for k in missing}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep")
    parser.add_argument("paper")
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()
    result = compare(read_sweep(args.sweep), read_paper_data(args.paper), args.threshold)
    print(f"published {result['published']} ESS ({result['published_norms']} norms); engine found "
          f"{result['common']} of them, missed {len(result['missing'])} (in {result['missing_norms']} norms).")
    print(f"engine rows not in the file: {len(result['extra'])}, of which with cooperation >= {args.threshold}: "
          f"{len(result['extra_above_threshold'])}")
    print(f"cooperation level of the common rows: mean difference {result['mean_difference']:.2e}, "
          f"max {result['max_difference']:.2e}")
    for norm, strategy in result["missing"][:10]:
        print("  missed:", norm, strategy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
