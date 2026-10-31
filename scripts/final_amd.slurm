#!/bin/bash
#SBATCH -J final-amd
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH -w hepnode0
#SBATCH --time=00:40:00
#SBATCH --output=tmp/logs/final-amd-%j.out
#SBATCH --error=tmp/logs/final-amd-%j.err
set -euo pipefail
if [ "${V5_BOUND:-0}" != 1 ]; then
    exec srun --cpu-bind=cores --distribution=block:block --export=ALL,V5_BOUND=1 bash "$0"
fi
cd "${SLURM_SUBMIT_DIR:?}"
# Toolchain only: read-only, no writes into the old AMD experiment directory.
source /home/zsx/melspec-v3-amd/tmp/amd-gpu/env.sh
export HIP_VISIBLE_DEVICES=0
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache" NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
mkdir -p tmp/bin tmp/logs tmp/rocfft-cache tmp/scratch
export ROCFFT_RTC_CACHE_PATH="$PWD/tmp/rocfft-cache/warm.db"
scratch=$(mktemp -d "/tmp/melspec-v5-final-amd-$SLURM_JOB_ID-XXXXXX")
ln -s "$scratch" "tmp/scratch/node-$SLURM_JOB_ID"
export TMPDIR="$PWD/tmp/scratch/node-$SLURM_JOB_ID"
trap 'rmdir "$scratch"' EXIT
PY="$HOME/venv/bin/python"
echo "job=$SLURM_JOB_ID host=$(hostname) mode=${MODE:-train} commit=$(cat tmp/source-commit.txt)"
grep Cpus_allowed_list /proc/self/status
lscpu
cat /sys/kernel/mm/transparent_hugepage/enabled
sha256sum src/* include/* Makefile
rocminfo | grep -E 'Marketing Name|Name:.*gfx' || true
flags=(-std=c++17 -O3 -march=native -pthread -Iinclude -DMELSPEC_FP32_BATCH=16)
if [ "${MODE:-train}" = train ]; then
    make CUDA=0 HIP=0 -B -j8 EXEC=tmp/bin/amd-cpu > tmp/logs/f3-cpu-build-$SLURM_JOB_ID.txt 2>&1
    make CUDA=0 HIP=1 -B -j8 EXEC=tmp/bin/final-hip > tmp/logs/f3-build-$SLURM_JOB_ID.txt 2>&1
    make CUDA=0 HIP=1 HIP_ONLY=1 -B -j8 EXEC=tmp/bin/dedicated-hip > tmp/logs/f3-dedicated-build-$SLURM_JOB_ID.txt 2>&1
    g++ "${flags[@]}" src/main.cpp tmp/control/cpu-frozen.cpp -o tmp/bin/amd-frozen-cpu
    "$HIPCC" -std=c++17 -O3 --offload-arch=gfx1201 --rocm-path="$ROCM_PATH" \
        --rocm-device-lib-path="$HIP_DEVICE_LIB_PATH" -march=native -pthread \
        -Itmp/control -Iinclude -I"$ROCM_PATH/include" -I"$ROCM_PATH/include/hipfft" \
        -DMELSPEC_HIP_OVERLAP_INIT=1 -c tmp/control/hip-frozen.cu -o tmp/control/hip-frozen.o
    g++ "${flags[@]}" -Itmp/control -DMELSPEC_HIP=1 src/main.cpp tmp/control/amd-math.cpp \
        tmp/control/hip-frozen.o -L"$ROCM_PATH/lib/x86_64-linux-gnu" \
        -Wl,-rpath,"$ROCM_PATH/lib/x86_64-linux-gnu" -lamdhip64 -lhipfft -lrocfft \
        -o tmp/bin/frozen-hip
fi
for mode in cpu hip; do
cat > tmp/bin/amd-force-$mode <<SH
#!/bin/bash
export MELSPEC_BACKEND=$mode
exec "$PWD/tmp/bin/final-hip" "\$@"
SH
chmod +x tmp/bin/amd-force-$mode
done
sha256sum tmp/bin/amd-cpu tmp/bin/final-hip tmp/bin/dedicated-hip tmp/bin/frozen-hip
if [ "${MODE:-train}" = train ]; then
    "$PY" -u scripts/cpu_pipeline_check.py tmp/bin/amd-frozen-cpu tmp/bin/amd-force-hip \
        > tmp/logs/f3-hip-68-$SLURM_JOB_ID.txt
    "$PY" -u scripts/final_bench.py cpu=tmp/bin/amd-force-cpu hip=tmp/bin/amd-force-hip \
        --suite matrix --runs 5 --out tmp/logs/f3-train-$SLURM_JOB_ID.json
    "$PY" -u scripts/final_bench.py cpu=tmp/bin/amd-force-cpu hip=tmp/bin/amd-force-hip \
        --suite extended --runs 5 --out tmp/logs/f3-extended-$SLURM_JOB_ID.json
    "$PY" -u scripts/final_bench.py cpu=tmp/bin/amd-force-cpu hip=tmp/bin/amd-force-hip \
        dedicated=tmp/bin/dedicated-hip frozen=tmp/bin/frozen-hip oldcpu=tmp/bin/amd-frozen-cpu \
        --suite official --runs 6 --out tmp/logs/f3-ab-$SLURM_JOB_ID.json
else
    make CUDA=0 HIP=1 -B -j8 EXEC=tmp/bin/final-hip > tmp/logs/f4-build-$SLURM_JOB_ID.txt 2>&1
    "$PY" -u scripts/final_bench.py cpu=tmp/bin/amd-force-cpu hip=tmp/bin/amd-force-hip \
        auto=tmp/bin/final-hip dedicated=tmp/bin/dedicated-hip \
        --suite holdout --runs 6 --out tmp/logs/f4-amd-holdout-$SLURM_JOB_ID.json
    for mode in auto cpu hip; do
        MELSPEC_BACKEND="$mode" "$PY" -u validate.py tmp/bin/final-hip --runs 1 \
            | tee tmp/logs/f4-amd-official-$mode-$SLURM_JOB_ID.txt
        grep -q 'Result: 4/4 PASSED' tmp/logs/f4-amd-official-$mode-$SLURM_JOB_ID.txt
    done
    "$PY" -u validate.py tmp/bin/dedicated-hip --runs 1 \
        | tee tmp/logs/f4-amd-dedicated-official-$SLURM_JOB_ID.txt
    grep -q 'Result: 4/4 PASSED' tmp/logs/f4-amd-dedicated-official-$SLURM_JOB_ID.txt
    for dedicated in 0 1; do
        make CUDA=0 HIP=1 HIP_ONLY="$dedicated" HIP_PROFILE=1 -B -j8 \
            EXEC=tmp/bin/hip-profile-$dedicated > tmp/logs/f4-hip-profile-build-$dedicated-$SLURM_JOB_ID.txt 2>&1
        "$PY" -u scripts/final_gpu_profile.py tmp/bin/hip-profile-$dedicated --backend hip \
            > tmp/logs/f4-hip-profile-$dedicated-$SLURM_JOB_ID.txt
    done
fi
sha256sum "$ROCFFT_RTC_CACHE_PATH" || true
