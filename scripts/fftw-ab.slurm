#!/usr/bin/env bash
#SBATCH --job-name=fftw-ab-final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=tmp/logs/ab-%j.out
#SBATCH --error=tmp/logs/ab-%j.err
# Prepared A/B: frozen float+PATIENT vs old double A2 vs SIMD Batch8 candidate.
# Run ONLY when the node is not used by another Agent's benchmark job.
# The SIMD candidate is supplied by its owner; this worktree does not build or
# modify another experiment. Interleaved rotation and full-output checks only.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${SIMD_BATCH8_EXE:?set SIMD_BATCH8_EXE to the SIMD Batch8 executable (absolute path)}"
if [ ! -x "$SIMD_BATCH8_EXE" ]; then
  echo "ABORT: SIMD_BATCH8_EXE is not executable: $SIMD_BATCH8_EXE" >&2
  exit 2
fi
echo "job=$SLURM_JOB_ID node=$(hostname)"
grep Cpus_allowed_list /proc/self/status
git rev-parse HEAD
# Refuse to benchmark while another Agent's job is running on this node.
busy=$(squeue -h -w "$SLURMD_NODENAME" -t RUNNING -o "%i %u" \
  | awk -v me="$SLURM_JOB_ID" '$1 != me && $1 != me".batch" && $1 != me".extern" {print}' | head -n 3)
if [ -n "$busy" ]; then
  echo "ABORT: node is busy with another job; resubmit when idle:" >&2
  echo "$busy" >&2
  exit 3
fi
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache" NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores
unset MELSPEC_FFTWF_WISDOM MELSPEC_FFTWF_WISDOM_DIRECTORY MELSPEC_FFTWF_LIBRARY_ID
PY="$HOME/venv/bin/python"
D="$PWD/tmp/opt/fftw-3.3.11"
F="$PWD/tmp/opt/fftw-3.3.11-float"
W="$PWD/versions/V2-fftw/experiments/raw/cluster/final-wisdom/patient"
O=tmp/fftw-lab/ab
mkdir -p "$O"
LIB_ID=$(sha256sum "$F/lib/libfftw3f.a" | cut -d' ' -f1)
test "$LIB_ID" = 15c7fb154e1a65ed84858cf1776ce8d8a18ec46ddc55f556208f12e098fd761a
make -B -j4 FFTW_PREFIX="$D" FFT_PRECISION=double EXEC="$O/a2"
make -B -j4 FFTW_PREFIX="$F" FFT_PRECISION=float FFT_POWER=float FFTW_WISDOM=1 EXEC="$O/final"
LIB_ID="$LIB_ID" W="$W" O="$PWD/$O" "$PY" - <<'PY'
import json,os
from pathlib import Path
Path(os.environ['O'],'environments.json').write_text(json.dumps({"final": {
    "MELSPEC_FFTWF_LIBRARY_ID": os.environ["LIB_ID"],
    "MELSPEC_FFTWF_WISDOM_DIRECTORY": os.environ["W"]}}, indent=2) + "\n")
PY
sha256sum "$O/a2" "$O/final" "$SIMD_BATCH8_EXE"
scratch=$(mktemp -d "/tmp/fftw-ab-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
"$PY" -u scripts/fftw_bench.py --runs 5 \
  --programs "a2=$O/a2" "final=$O/final" "simd-batch8=$SIMD_BATCH8_EXE" \
  --env-json "$O/environments.json" --require-wisdom final \
  --out "tmp/logs/ab-$SLURM_JOB_ID.json" | tee "tmp/logs/ab-table-$SLURM_JOB_ID.txt"
"$PY" - "tmp/logs/ab-$SLURM_JOB_ID.json" <<'PY'
import json,math,sys
rows=json.load(open(sys.argv[1]))
names=sorted({r["program"] for r in rows})
print("geomean(same-job median ratios vs a2):")
for n in names:
    logs=[math.log(next(r for r in rows if r["case"]==c and r["program"]=="a2")["median_s"] /
                   next(r for r in rows if r["case"]==c and r["program"]==n)["median_s"])
          for c in ("default","small_mels","large_fft","narrow_band")]
    print(f"  {n}: {math.exp(sum(logs)/len(logs)):.4f}x")
PY
