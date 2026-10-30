#!/usr/bin/env bash
#SBATCH --job-name=fftw-lab-complete
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --output=tmp/logs/complete-%j.out
#SBATCH --error=tmp/logs/complete-%j.err
# Short closeout only: no rebuild, replanning, repeat score matrix or profiling.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${FINAL_JOB:?set FINAL_JOB to the completed release comparison job}"
export FINAL_JOB
git rev-parse HEAD
git diff --exit-code d8234a5 -- src include Makefile
grep Cpus_allowed_list /proc/self/status
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache" NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores
unset MELSPEC_FFTWF_WISDOM MELSPEC_FFTWF_WISDOM_DIRECTORY MELSPEC_FFTW_WISDOM MELSPEC_FFTW_WISDOM_DIRECTORY
PY="$HOME/venv/bin/python"
O=tmp/fftw-lab/final
sha256sum "$O"/a2 "$O"/float-* "$O"/combo-* "$O"/measure/*.wisdom "$O"/patient/*.wisdom
scratch=$(mktemp -d "/tmp/fftw-complete-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
"$PY" -u - <<'PY'
import json,os,subprocess,sys
from pathlib import Path
o=Path('tmp/fftw-lab/final').resolve()
rows=json.loads(Path(f'tmp/logs/final-compare-{os.environ["FINAL_JOB"]}.json').read_text())
assert len(rows)==36 and all(r['passed'] and len(r['times'])==5 for r in rows)
envs=json.loads((o/'environments.json').read_text())
for name,power in (('patient-float','float'),('patient-double','double')):
    env=dict(os.environ,**envs[name])
    run=subprocess.run([sys.executable,'-u','validate.py',str(o/f'combo-{power}'),'--runs','1'],
                       env=env,capture_output=True,text=True)
    text=run.stdout+'\n'+run.stderr
    Path(f'tmp/logs/complete-official-{name}-{os.environ["SLURM_JOB_ID"]}.txt').write_text(text)
    print(f'OFFICIAL candidate={name}\n{text}',flush=True)
    if run.returncode or 'Result: 4/4 PASSED' not in text:raise SystemExit(1)
PY
"$PY" scripts/fftw_lab_memory.py --programs "a2=$O/a2" "float-fp=$O/float-float" "float-dp=$O/float-double" \
  "measure-float=$O/combo-float" "measure-double=$O/combo-double" \
  "patient-float=$O/combo-float" "patient-double=$O/combo-double" \
  "padded4=$O/layout-0" "interleaved4=$O/layout-1" --env-json "$O/environments.json" \
  --out "tmp/logs/complete-memory-$SLURM_JOB_ID.json"
