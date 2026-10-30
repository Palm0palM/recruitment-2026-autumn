#!/usr/bin/env bash
#SBATCH --job-name=fftw-prof-final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:15:00
#SBATCH --output=tmp/logs/profile-final-%j.out
#SBATCH --error=tmp/logs/profile-final-%j.err
# One lightweight, targeted profile of the frozen float+PATIENT candidate.
# Reuses the committed PATIENT wisdom; never regenerates plans or rereads PCM
# for tuning. NOT a score: separate instrumented build, diagnostic times only.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "job=$SLURM_JOB_ID node=$(hostname)"
grep Cpus_allowed_list /proc/self/status
git rev-parse HEAD
# Best-effort guard: do not share this node with another Agent's running job.
busy=$(squeue -h -w "$SLURMD_NODENAME" -t RUNNING -o "%i %u" \
  | awk -v me="$SLURM_JOB_ID" '$1 != me && $1 != me".batch" && $1 != me".extern" {print}' | head -n 3)
if [ -n "$busy" ]; then
  echo "ABORT: other job(s) already running on $SLURMD_NODENAME:" >&2
  echo "$busy" >&2
  exit 3
fi
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache" NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores
unset MELSPEC_FFTWF_WISDOM MELSPEC_FFTWF_WISDOM_DIRECTORY MELSPEC_FFTWF_LIBRARY_ID
PY="$HOME/venv/bin/python"
F="$PWD/tmp/opt/fftw-3.3.11-float"
W="$PWD/versions/V2-fftw/experiments/raw/cluster/final-wisdom/patient"
LIB_ID=$(sha256sum "$F/lib/libfftw3f.a" | cut -d' ' -f1)
test "$LIB_ID" = 15c7fb154e1a65ed84858cf1776ce8d8a18ec46ddc55f556208f12e098fd761a
test -f "$W/r2c-8192.wisdom" && test -f "$W/r2c-16384.wisdom"
mkdir -p tmp/fftw-lab/final
make -B -j4 FFTW_PREFIX="$F" FFT_PRECISION=float FFT_POWER=float FFTW_WISDOM=1 \
  EXEC=tmp/fftw-lab/final/float-patient
export FFTW_PREFIX="$F"
export MELSPEC_FFTWF_LIBRARY_ID="$LIB_ID"
export MELSPEC_FFTWF_WISDOM_DIRECTORY="$W"
scratch=$(mktemp -d "/tmp/fftw-profile-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
"$PY" -u scripts/fftw_profile.py --precision float --power float --wisdom --repeat 2 \
  --out-json "tmp/logs/profile-final-$SLURM_JOB_ID.json" \
  > "tmp/logs/profile-final-$SLURM_JOB_ID.txt" 2>&1
# Deployment-level proof: exact candidate, all official cases, loaded vs fallback.
"$PY" scripts/fftw_verify_deployment.py --program tmp/fftw-lab/final/float-patient \
  --wisdom-dir "$W" --library-id "$LIB_ID" \
  --out "tmp/logs/deploy-final-$SLURM_JOB_ID.json" \
  > "tmp/logs/deploy-final-$SLURM_JOB_ID.txt" 2>&1
sha256sum tmp/fftw-lab/final/float-patient "$F/lib/libfftw3f.a" "$W"/*.wisdom \
  > "tmp/logs/freeze-final-$SLURM_JOB_ID.txt"
echo "profile + deployment verification complete"
