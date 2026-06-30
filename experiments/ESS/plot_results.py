"""Plot ESS results saved by find_ESS_parellel.py."""

import argparse
import sys
from pathlib import Path

import pandas as pd


ORIGINAL_CODE_DIR = (
    Path(__file__).resolve().parents[1] / "origin_money" / "indirect_reciprocity"
)
sys.path.insert(0, str(ORIGINAL_CODE_DIR))

from plotting_funcs import checkConservation, plotFigure  # noqa: E402


def read_results(csv_file: Path, threshold: float) -> pd.DataFrame:
    results = pd.read_csv(csv_file)

    required_columns = {
        "bb_name",
        "ess_name",
        "fitness",
        "w0",
        "w1",
        "alpha",
        "beta",
        "gamma",
    }
    missing_columns = required_columns - set(results.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing columns in {csv_file}: {missing}")

    big_brothers = results["bb_name"].str.extract(
        r"^CmeetsN:(\S+)\s+NmeetsC:(\S+)$"
    )
    if big_brothers.isna().any().any():
        bad_name = results.loc[big_brothers.isna().any(axis=1), "bb_name"].iloc[0]
        raise ValueError(f"Could not read Big Brother name: {bad_name}")

    # Convert the Python CSV names to the names expected by plotting_funcs.py.
    plot_data = pd.DataFrame(
        {
            "CmeetsN": big_brothers[0],
            "NmeetsC": big_brothers[1],
            "Strategy": results["ess_name"],
            "fit": results["fitness"],
            "w0": results["w0"],
            "w-1": 0.0,
            "w1": results["w1"],
            "a": results["alpha"],
            "b": results["beta"],
            "c": results["gamma"],
            "d": 0.0,
        }
    )

    plot_data = plot_data[plot_data["fit"] >= threshold].reset_index(drop=True)
    plot_data["Conservative"] = plot_data.apply(
        lambda row: checkConservation(row["CmeetsN"], row["NmeetsC"]), axis=1
    )
    return plot_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the CSV results produced by find_ESS_parellel.py."
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="tokens_CmeetsN_Noise0_B2.csv",
        type=Path,
        help="results CSV to plot",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="figure directory; defaults to Figs beside the CSV",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.0,
        help="ignore ESS results with fitness below this value",
    )
    parser.add_argument("--royal-road", "-r", action="store_true")
    parser.add_argument("--show-bb-labels", action="store_true")
    parser.add_argument("--show-cons", action="store_true")
    args = parser.parse_args()

    csv_file = args.csv_file.resolve()
    output_dir = args.output_dir or csv_file.parent / "Figs"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_data = read_results(csv_file, args.threshold)
    if plot_data.empty:
        print("No results remain after applying the fitness threshold.")
        return

    print(f"Plotting {len(plot_data)} ESS results from {csv_file}")
    print(f"Writing figures to {output_dir}")
    plotFigure(
        plot_data,
        csv_file.stem,
        str(output_dir),
        doSort=True,
        royal_road=args.royal_road,
        show_bb_labels=args.show_bb_labels,
        show_cons=args.show_cons,
    )


if __name__ == "__main__":
    main()
