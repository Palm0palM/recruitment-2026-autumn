#!/bin/bash
#SBATCH -J final-threads64
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -w hepnode2
#SBATCH --time=00:25:00
#SBATCH --output=tmp/logs/threads64-%j.out
#SBATCH --error=tmp/logs/threads64-%j.err
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
mkdir -p tmp/bin tmp/logs tmp/scratch
scratch=$(mktemp -d "/tmp/melspec-v5-final-threads64-$SLURM_JOB_ID-XXXXXX")
ln -s "$scratch" "tmp/scratch/threads64-$SLURM_JOB_ID"
export TMPDIR="$scratch"
trap 'rmdir "$scratch"' EXIT
PY="$HOME/venv/bin/python"
echo "job=$SLURM_JOB_ID host=$(hostname)"
grep Cpus_allowed_list /proc/self/status
lscpu
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
sha256sum src/* include/* Makefile
make CUDA=1 HIP=0 CPU_THREADS=64 CUDA_ARCH=89 -B -j16 EXEC=tmp/bin/hybrid64 > tmp/logs/threads64-build-$SLURM_JOB_ID.txt 2>&1
for n in 16 32 64; do
    cat > tmp/bin/cpu$n <<SH
#!/bin/bash
export MELSPEC_BACKEND=cpu MELSPEC_CPU_THREADS=$n MELSPEC_DISPATCH_LOG=1
exec taskset -c 0-$((n-1)) "$PWD/tmp/bin/hybrid64" "\$@"
SH
done
cat > tmp/bin/cuda64 <<SH
#!/bin/bash
export MELSPEC_BACKEND=cuda
exec "$PWD/tmp/bin/hybrid64" "\$@"
SH
cat > tmp/bin/auto16 <<SH
#!/bin/bash
export MELSPEC_CPU_THREADS=16
exec taskset -c 0-15 "$PWD/tmp/bin/hybrid64" "\$@"
SH
chmod +x tmp/bin/cpu{16,32,64} tmp/bin/cuda64 tmp/bin/auto16
"$PY" -u scripts/final_bench.py cpu=tmp/bin/cpu16 cpu32=tmp/bin/cpu32 cpu64=tmp/bin/cpu64 \
    cuda=tmp/bin/cuda64 auto16=tmp/bin/auto16 --suite official --runs 3 \
    --out tmp/logs/threads64-ab-$SLURM_JOB_ID.json
sha256sum tmp/bin/hybrid64
