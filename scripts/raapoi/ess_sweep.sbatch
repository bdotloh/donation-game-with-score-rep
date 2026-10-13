#!/bin/bash
# ESS sweep as a Slurm job array on Rāpoi. Each array task does its share of the norms with
# --cpus-per-task worker processes, writes into $OUT, and skips norms it already finished if rerun.
#   mkdir -p logs && sbatch scripts/raapoi/ess_sweep.sbatch
# When all tasks have finished:
#   python scripts/ess_sweep.py merge $OUT
#
#SBATCH --job-name=ess-sweep
#SBATCH --partition=parallel
#SBATCH --array=0-19
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/ess-%A_%a.out
#SBATCH --error=logs/ess-%A_%a.err

# ---- edit these -------------------------------------------------------------------------
REPO=$HOME/donation-game-with-score-rep
VENV=$HOME/venvs/donation-game
PYTHON_MODULES="GCCcore/10.3.0 Python/3.9.5"
OUT=/nfs/scratch/$USER/ess/binary-cn-noise0
ARGS="--norms binary-cn --strategies binary"
# Ternary examples:
#   ARGS="--norms lifted-cn --tau 3 --strategies ternary"
#   ARGS="--norms ternary-cn --tau 5 --strategies ternary --sample 100000 --seed 1"
#   ARGS="--norms file:$REPO/my_norms.txt --tau 5 --strategies file:$REPO/my_strategies.txt"
# Model options (defaults as find_ESS in helper_funcs.jl): --eps 0.1 --rho 0.99 --maxn 100 ...
# -----------------------------------------------------------------------------------------

module purge
module load config
module load $PYTHON_MODULES
source "$VENV/bin/activate"
# One thread per worker process; the parallelism comes from the processes.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
python scripts/ess_sweep.py run $ARGS --out "$OUT"
