#!/bin/bash
# One-off setup on Rāpoi's login node: a Python virtual environment with numpy and scipy.
#   bash scripts/raapoi/setup_env.sh
# Check the available Pythons with `module spider Python`, and what each needs loaded first
# with `module spider Python/<version>`; then set PYTHON_MODULES (also in ess_sweep.sbatch).
set -euo pipefail
PYTHON_MODULES=${PYTHON_MODULES:-"GCCcore/10.3.0 Python/3.9.5"}
VENV=${VENV:-$HOME/venvs/donation-game}

module purge
module load config
module load $PYTHON_MODULES
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r "$(dirname "$0")/../../requirements-engine.txt"
python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
echo "Virtual environment ready: $VENV"
