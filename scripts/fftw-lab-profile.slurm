#!/usr/bin/env bash
#SBATCH --job-name=fftw-lab-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --output=tmp/logs/lab-profile-%j.out
#SBATCH --error=tmp/logs/lab-profile-%j.err
# Diagnostic ONLY; launch after final scoring is finished. Never rank these times.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${FINAL_JOB:?set FINAL_JOB to the completed final comparison job}"
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache" NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores
PY="$HOME/venv/bin/python"
git rev-parse HEAD
grep Cpus_allowed_list /proc/self/status
scratch=$(mktemp -d "/tmp/fftw-profile-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
FFTW_PREFIX="$PWD/tmp/opt/fftw-3.3.11" "$PY" -u scripts/fftw_profile.py \
  > "tmp/logs/lab-profile-a2-$SLURM_JOB_ID.txt" 2>&1
export FINAL_JOB
"$PY" -u - <<'PY'
import json,math,os,subprocess,sys
from pathlib import Path
rows=json.loads(Path(f'tmp/logs/final-compare-{os.environ["FINAL_JOB"]}.json').read_text())
names={r['program'] for r in rows}-{'a2','padded4','interleaved4'}
winner=min(names,key=lambda n:sum(math.log(r['median_s']) for r in rows if r['program']==n))
envs=json.loads(Path('tmp/fftw-lab/final/environments.json').read_text())
env=dict(os.environ,**envs.get(winner,{})); env['FFTW_PREFIX']=str(Path('tmp/opt/fftw-3.3.11-float').resolve())
power='double' if winner=='float-dp' or winner.endswith('-double') else 'float'
cmd=[sys.executable,'-u','scripts/fftw_profile.py','--precision','float','--power',power]
if winner in envs:cmd+=['--wisdom']
print(f'Diagnostic candidate={winner}; selection from release medians, NEVER from instrumented timings',flush=True)
with Path(f'tmp/logs/lab-profile-winner-{os.environ["SLURM_JOB_ID"]}.txt').open('w') as out:
    subprocess.run(cmd,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
PY
