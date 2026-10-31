#!/bin/bash
#SBATCH -J final-64-verify
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -w hepnode2
#SBATCH --time=00:15:00
#SBATCH --output=tmp/logs/threads64-verify-%j.out
#SBATCH --error=tmp/logs/threads64-verify-%j.err
set -euo pipefail
if [ "${V5_BOUND:-0}" != 1 ]; then
    exec srun --cpu-bind=cores --distribution=block:block --export=ALL,V5_BOUND=1 bash "$0"
fi
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache" NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export MELSPEC_DATA=/cache/recuritment/melspec/data
export CUDA_HOME=/home/zsx/melspec-v3-cuda-agent/toolkit/cuda-13.4
export PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_CACHE_PATH="$PWD/tmp/cuda-cache" XDG_CACHE_HOME="$PWD/tmp/xdg-cache"
scratch=$(mktemp -d "/tmp/melspec-v5-final-64verify-$SLURM_JOB_ID-XXXXXX")
ln -s "$scratch" "tmp/scratch/threads64-verify-$SLURM_JOB_ID"
export TMPDIR="$scratch"
trap 'rmdir "$scratch"' EXIT
PY="$HOME/venv/bin/python"
echo "job=$SLURM_JOB_ID host=$(hostname)"
grep Cpus_allowed_list /proc/self/status
sha256sum src/* include/* Makefile
make CUDA=1 HIP=0 CPU_THREADS=64 L40_THREADS64=1 CUDA_ARCH=89 -B -j16 \
    EXEC=tmp/bin/final64 > tmp/logs/threads64-verify-build-$SLURM_JOB_ID.txt 2>&1
"$PY" -u scripts/test_cpu_threads.py tmp/bin/final64 > tmp/logs/threads64-contract-$SLURM_JOB_ID.json
cat > tmp/bin/cpu64verify <<SH
#!/bin/bash
export MELSPEC_BACKEND=cpu MELSPEC_CPU_THREADS=64
exec "$PWD/tmp/bin/final64" "\$@"
SH
cat > tmp/bin/cudaverify <<SH
#!/bin/bash
export MELSPEC_BACKEND=cuda
exec "$PWD/tmp/bin/final64" "\$@"
SH
chmod +x tmp/bin/cpu64verify tmp/bin/cudaverify
"$PY" -u scripts/final_bench.py cpu=tmp/bin/cpu16 cpu64=tmp/bin/cpu64verify \
    cuda=tmp/bin/cudaverify auto16=tmp/bin/auto16 auto64=tmp/bin/final64 \
    --suite official --runs 5 --out tmp/logs/threads64-verify-ab-$SLURM_JOB_ID.json
MELSPEC_BACKEND=auto "$PY" -u validate.py tmp/bin/final64 --runs 1 \
    | tee tmp/logs/threads64-official-$SLURM_JOB_ID.txt
grep -q 'Result: 4/4 PASSED' tmp/logs/threads64-official-$SLURM_JOB_ID.txt
sha256sum tmp/bin/final64
