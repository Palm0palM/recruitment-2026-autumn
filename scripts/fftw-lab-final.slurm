#!/usr/bin/env bash
#SBATCH --job-name=fftw-lab-final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=tmp/logs/final-%j.out
#SBATCH --error=tmp/logs/final-%j.err
# Parent only. All builds, offline planning and scored invocations are serial stages.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache" NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores
unset MELSPEC_FFTW_WISDOM MELSPEC_FFTW_WISDOM_DIRECTORY MELSPEC_FFTWF_WISDOM MELSPEC_FFTWF_WISDOM_DIRECTORY
PY="$HOME/venv/bin/python"
D="$PWD/tmp/opt/fftw-3.3.11"
F="$PWD/tmp/opt/fftw-3.3.11-float"
O=tmp/fftw-lab/final
E=versions/V2-fftw/experiments
mkdir -p "$O/measure" "$O/patient"
echo "job=$SLURM_JOB_ID node=$(hostname)"
git rev-parse HEAD
git status --short
grep Cpus_allowed_list /proc/self/status
g++ --version
make -B -j4 FFTW_PREFIX="$D" FFT_PRECISION=double EXEC="$O/a2"
test "$(sha256sum "$O/a2" | cut -d' ' -f1)" = f1ab166c877e3921ed1a1d330a69ca542e243cd69eaef9184443662e7eb88e64
for power in float double; do
  make -B -j4 FFTW_PREFIX="$F" FFT_PRECISION=float FFT_POWER="$power" EXEC="$O/float-$power"
  make -B -j4 FFTW_PREFIX="$F" FFT_PRECISION=float FFT_POWER="$power" FFTW_WISDOM=1 EXEC="$O/combo-$power"
done
# Generator uses the exact production helper; it is NOT linked into the app.
make FFT_PRECISION=float FFTW_PREFIX="$F" wisdom-generator
LIB_ID=$(sha256sum "$F/lib/libfftw3f.a" | cut -d' ' -f1)
for planner in measure patient; do
  for n in 1 2 8 256 2048 8192 16384 32768 65536; do
    tmp/bin/fftwf-wisdom-generate --size "$n" --planner "$planner" --library-id "$LIB_ID" \
      --output "$O/$planner/r2c-$n.wisdom"
  done
done
for layout in 0 1; do
  g++ -std=c++17 -O3 -march=native -pthread -Iinclude -I"$D/include" \
    -DMELSPEC_FFTW_BATCH=4 -DMELSPEC_BATCH_LAYOUT="$layout" src/main.cpp "$E/batch/replacement_layout.cpp" \
    "$D/lib/libfftw3.a" -lm -o "$O/layout-$layout"
done
sha256sum "$D/lib/libfftw3.a" "$F/lib/libfftw3f.a" "$O"/a2 "$O"/float-* "$O"/combo-* "$O"/layout-*
scratch=$(mktemp -d "/tmp/fftw-final-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
"$PY" scripts/fftw_check.py --reference "$O/a2" "$O/float-float" "$O/float-double" "$O/layout-0" "$O/layout-1"
for planner in measure patient; do
  MELSPEC_FFTWF_LIBRARY_ID="$LIB_ID" MELSPEC_FFTWF_WISDOM_DIRECTORY="$PWD/$O/$planner" \
    "$PY" scripts/fftw_check.py --reference "$O/a2" "$O/combo-float" "$O/combo-double"
  "$PY" scripts/fftw_wisdom_check.py --reference "$O/a2" --program "$O/combo-float" \
    --wisdom-dir "$O/$planner" --library-id "$LIB_ID" --precision float \
    --out "tmp/logs/final-envelope-$planner-$SLURM_JOB_ID.json"
  MELSPEC_FFTWF_LIBRARY_ID="$LIB_ID" MELSPEC_FFTWF_WISDOM_DIRECTORY="$PWD/$O/$planner" \
    "$PY" "$E/precision/stress_precision.py" --reference "$PWD/$O/a2" \
      --programs "float-fp=$PWD/$O/float-float" "float-dp=$PWD/$O/float-double" \
      "combo-fp=$PWD/$O/combo-float" "combo-dp=$PWD/$O/combo-double" \
      --out "$O/stress-$planner" > "tmp/logs/final-stress-$planner-$SLURM_JOB_ID.txt"
  cp "$O/stress-$planner/summary.json" "tmp/logs/final-stress-$planner-$SLURM_JOB_ID.json"
done
O="$PWD/$O" LIB_ID="$LIB_ID" "$PY" - <<'PY'
import json, os
from pathlib import Path
o=Path(os.environ['O'])
env={}
for planner in ('measure','patient'):
    for power in ('float','double'):
        env[f'{planner}-{power}']={'MELSPEC_FFTWF_LIBRARY_ID':os.environ['LIB_ID'],
          'MELSPEC_FFTWF_WISDOM_DIRECTORY':str(o/planner)}
(o/'environments.json').write_text(json.dumps(env,indent=2)+'\n')
PY
programs=("a2=$O/a2" "float-fp=$O/float-float" "float-dp=$O/float-double"
  "measure-float=$O/combo-float" "measure-double=$O/combo-double"
  "patient-float=$O/combo-float" "patient-double=$O/combo-double"
  "padded4=$O/layout-0" "interleaved4=$O/layout-1")
"$PY" -u scripts/fftw_bench.py --runs 5 --programs "${programs[@]}" \
  --env-json "$O/environments.json" --require-wisdom measure-float measure-double patient-float patient-double \
  --out "tmp/logs/final-compare-$SLURM_JOB_ID.json"
# Official validator of every integrated float precision/planner combination,
# not just the one with the best hand-picked minimum. Original tolerance unchanged.
O="$PWD/$O" JOB="$SLURM_JOB_ID" "$PY" -u - <<'PY'
import json,os,subprocess,sys
from pathlib import Path
o=Path(os.environ['O']); envs=json.loads((o/'environments.json').read_text())
targets={'float-fp':o/'float-float','float-dp':o/'float-double'}
targets.update({n:o/('combo-float' if n.endswith('-float') else 'combo-double') for n in envs})
for name,program in targets.items():
    env=dict(os.environ,**envs.get(name,{}))
    run=subprocess.run([sys.executable,'-u','validate.py',str(program),'--runs','3'],env=env,capture_output=True,text=True)
    text=run.stdout+'\n'+run.stderr
    Path(f'tmp/logs/final-official-{name}-{os.environ["JOB"]}.txt').write_text(text)
    print(f'OFFICIAL candidate={name}\n{text}',flush=True)
    if run.returncode or 'Result: 4/4 PASSED' not in text: raise SystemExit(1)
PY
"$PY" scripts/fftw_lab_memory.py --programs "${programs[@]}" --env-json "$O/environments.json" \
  --out "tmp/logs/final-memory-$SLURM_JOB_ID.json"
cp "$O/environments.json" "tmp/logs/final-environments-$SLURM_JOB_ID.json"
