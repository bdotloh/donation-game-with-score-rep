# ESS sweeps on Rāpoi

`scripts/ess_sweep.py` finds, for every scoring norm in a set, the ESS among a set of
candidate strategies, as `find_ESS` in `helper_funcs.jl` does (it reproduces the Julia
results for all 1296 binary CN norms at noise 0 and 0.01). These files run it as a
Slurm job array on Rāpoi.

## Once

```bash
git clone https://github.com/bdotloh/donation-game-with-score-rep.git
cd donation-game-with-score-rep
module spider Python                     # pick a Python; the engine works with 3.9 and later
bash scripts/raapoi/setup_env.sh         # makes ~/venvs/donation-game with numpy and scipy
```

If you pick another Python, set `PYTHON_MODULES` (e.g. `PYTHON_MODULES="GCCcore/... Python/3.11.5"`,
from `module spider Python/3.11.5`) when running `setup_env.sh` and in `ess_sweep.sbatch`.

## A quick test (quicktest partition)

```bash
module load GCCcore/10.3.0 Python/3.9.5 && source ~/venvs/donation-game/bin/activate
srun --partition=quicktest --cpus-per-task=4 --mem=8G --time=0:30:00 \
    python scripts/ess_sweep.py run --norms lifted-cn --tau 3 --strategies ternary \
    --out /nfs/scratch/$USER/ess/test --limit 20
```

It prints the time per norm, so you can size the real run.

## The sweep

Edit the block at the top of `ess_sweep.sbatch` (`OUT`, `ARGS`, the array size, CPUs and
time), then

```bash
mkdir -p logs
sbatch scripts/raapoi/ess_sweep.sbatch
squeue -u $USER                          # watch it
python scripts/ess_sweep.py merge /nfs/scratch/$USER/ess/binary-cn-noise0
```

`merge` writes `ess.csv` and says how many norms are still missing. If a task fails or hits
its time limit, resubmit just that task (e.g. `sbatch --array=7 scripts/raapoi/ess_sweep.sbatch`):
it skips the norms it has already done. Keep each `OUT` for one set of settings; the script
refuses to mix runs with different settings in one directory.

## Choosing norms and strategies

| `--norms` | norms |
|---|---|
| `binary-cn`, `binary-nc`, `binary` | every binary norm with only CN, only NC, or both (1296, 1296, 1679616) |
| `lifted-cn`, `lifted` | the same binary norms, as ternary norms (`--tau`) |
| `ternary-cn`, `ternary-nc`, `ternary` | every ternary norm (`--tau`; 34012224 CN-only ones, so use `--sample N`) |
| `file:PATH` | one barcode per line, `CN:... NC:...` (8 or 18 characters each) |

`--strategies` (and `--invaders`, which default to the same): `binary` (16), `ternary`
(512, `--tau`) or `file:PATH` with one name per line (`0,0|0,g`, `0,0,0|0,g,0|0,g,0`, `phi:0.17`).
`python scripts/ess_sweep.py count --norms ternary-cn --tau 5` shows how many norms a set has.

Model options default to `find_ESS`'s: no noise (`--eps 0.1` is `NOISE_VAL = 0.1`; `--eps-up`
and `--eps-down` default to 0.5), `--rho 0.99`, `--maxn 100`, `--w-init 0.5,0.5`, benefit 2,
cost 1, `--fitness-threshold 1e-5`.

## Sizing

Measured here: about 0.2 CPU-seconds per binary norm with the 16 binary strategies, and
about 20 CPU-seconds per norm with the 512 ternary strategies (more for norms with many
ESS). So 1296 lifted norms with ternary strategies take about 7 CPU-hours (minutes with a
few hundred CPUs); a sample of 100000 ternary norms about 550 CPU-hours, e.g. about an
hour on 20 tasks x 32 CPUs. Time a few norms first (`--limit`), since the cost depends on the
norms. Ask for 2 GB per CPU (`--mem-per-cpu=2G`).

## Output

`ess.csv` has one row per ESS: `norm`, `ess`, `fitness` (4 x payoff, as `helper_funcs.jl`
prints it), `w0`, `w1`, `mean_score`, the rates `alpha` (g₁), `beta` (g₂), `gamma` (f₁), for
ternary runs also `beta_top` and `gamma_top` (up- and down-rates in the top class), and
`converged`. `config.json` records the settings.
