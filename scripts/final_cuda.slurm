#!/bin/bash
#SBATCH -J final-cuda
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=00:35:00
#SBATCH --output=tmp/logs/final-cuda-%j.out
#SBATCH --error=tmp/logs/final-cuda-%j.err
set -euo pipefail
if [ "${V5_BOUND:-0}" != 1 ]; then
    exec srun --cpu-bind=cores --distribution=block:block \
        --export=ALL,V5_BOUND=1 bash "$0"
fi
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPYCACHEPREFIX="$PWD/tmp/pycache" NUMBA_CACHE_DIR="$PWD/tmp/numba-cache"
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export CUDA_HOME=/home/zsx/melspec-v3-cuda-agent/toolkit/cuda-13.4
export PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
PY="$HOME/venv/bin/python"
mkdir -p tmp/bin tmp/logs tmp/scratch
scratch=$(mktemp -d "/tmp/melspec-v5-final-cuda-$SLURM_JOB_ID-XXXXXX")
ln -s "$scratch" "tmp/scratch/node-$SLURM_JOB_ID"
export TMPDIR="$PWD/tmp/scratch/node-$SLURM_JOB_ID"
trap 'rmdir "$scratch"' EXIT
echo "job=$SLURM_JOB_ID host=$(hostname) mode=${MODE:-train} commit=$(cat tmp/source-commit.txt)"
grep Cpus_allowed_list /proc/self/status
lscpu
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
cat /sys/kernel/mm/transparent_hugepage/enabled
g++ --version | head -1
sha256sum src/* include/* Makefile src/main.cpp validate.py
flags=(-std=c++17 -O3 -march=native -pthread -Iinclude -DMELSPEC_FP32_BATCH=16)
if [ "${MODE:-train}" = train ]; then
    make CUDA=0 -B -j8 EXEC=tmp/bin/final-cpu > tmp/logs/f1-build-$SLURM_JOB_ID.txt 2>&1
    make CUDA=1 -B -j8 CUDA_ARCH=89 EXEC=tmp/bin/final-cuda > tmp/logs/f2-build-$SLURM_JOB_ID.txt 2>&1
    g++ "${flags[@]}" src/main.cpp tmp/control/cpu-frozen.cpp -o tmp/bin/frozen-cpu
    "$CUDA_HOME/bin/nvcc" -std=c++17 -O3 -arch=sm_89 -Xcompiler -O3,-march=native,-pthread \
        -Iinclude -DMELSPEC_CUDA_PRECISION=32 -DMELSPEC_CUDA_BATCH=512 \
        -DMELSPEC_CUDA_FUSED=1 -DMELSPEC_CUDA_OVERLAP_INIT=1 \
        -DMELSPEC_CUDA_ASYNC_OUTPUT=1 -DMELSPEC_CUDA_RESIZE_THREAD=1 \
        -DMELSPEC_CUDA_STAGING_ROWS=16 -c tmp/control/cuda-frozen.cu -o tmp/control/cuda-frozen.o
    g++ "${flags[@]}" -DMELSPEC_CUDA=1 -I"$CUDA_HOME/targets/x86_64-linux/include" \
        src/main.cpp tmp/control/cuda-math.cpp tmp/control/cuda-frozen.o \
        -L"$CUDA_HOME/lib64" -Wl,-rpath,"$CUDA_HOME/lib64" -lcufft -lcudart -o tmp/bin/frozen-cuda
    "$PY" -u scripts/cpu_pipeline_check.py tmp/bin/frozen-cpu tmp/bin/final-cpu \
        > tmp/logs/f1-b16-check-$SLURM_JOB_ID.txt
    g++ -std=c++17 -O1 -g -march=native -pthread -Iinclude -DMELSPEC_FP32_BATCH=16 \
        -DMELSPEC_PACK_REUSE=1 -DMELSPEC_OUTPUT_HUGEPAGE=1 \
        -fsanitize=address,undefined -fno-omit-frame-pointer \
        scripts/cpu_pipeline_interface_check.cpp src/melspec.cpp src/melspec_cpu.cpp \
        -o tmp/bin/f1-sanitize
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ASAN_OPTIONS=detect_leaks=1 \
        tmp/bin/f1-sanitize > tmp/logs/f1-sanitize-$SLURM_JOB_ID.txt 2>&1
fi
cat > tmp/bin/force-cpu <<SH
#!/bin/bash
export MELSPEC_BACKEND=cpu
exec "$PWD/tmp/bin/final-cuda" "\$@"
SH
cat > tmp/bin/force-cuda <<SH
#!/bin/bash
export MELSPEC_BACKEND=cuda
exec "$PWD/tmp/bin/final-cuda" "\$@"
SH
chmod +x tmp/bin/force-cpu tmp/bin/force-cuda
sha256sum tmp/bin/final-cpu tmp/bin/final-cuda tmp/bin/frozen-cpu tmp/bin/frozen-cuda
if [ "${MODE:-train}" = train ]; then
    "$PY" -u scripts/final_bench.py cpu=tmp/bin/force-cpu cuda=tmp/bin/force-cuda \
        --suite matrix --runs 5 --out tmp/logs/f2-train-$SLURM_JOB_ID.json
    "$PY" -u scripts/final_bench.py cpu=tmp/bin/force-cpu oldcpu=tmp/bin/frozen-cpu \
        cuda=tmp/bin/force-cuda frozen=tmp/bin/frozen-cuda cpuonly=tmp/bin/final-cpu \
        --suite official --runs 6 --out tmp/logs/f2-ab-$SLURM_JOB_ID.json
elif [ "$MODE" = extended ]; then
    "$PY" -u scripts/final_bench.py cpu=tmp/bin/force-cpu cuda=tmp/bin/force-cuda \
        --suite extended --runs 5 --out tmp/logs/f2-extended-$SLURM_JOB_ID.json
else
    make CUDA=1 -B -j8 CUDA_ARCH=89 EXEC=tmp/bin/final-cuda > tmp/logs/f2-auto-build-$SLURM_JOB_ID.txt 2>&1
    "$PY" -u scripts/final_bench.py cpu=tmp/bin/force-cpu cuda=tmp/bin/force-cuda \
        auto=tmp/bin/final-cuda --suite holdout --runs 6 --out tmp/logs/f2-holdout-$SLURM_JOB_ID.json
    for mode in auto cpu cuda; do
        MELSPEC_BACKEND="$mode" "$PY" -u validate.py tmp/bin/final-cuda --runs 3 \
            | tee tmp/logs/f2-official-$mode-$SLURM_JOB_ID.txt
        grep -q 'Result: 4/4 PASSED' tmp/logs/f2-official-$mode-$SLURM_JOB_ID.txt
    done
    "$PY" -u scripts/cpu_pipeline_profile.py --tag final --runs 1 \
        --cppflag=-DMELSPEC_PACK_REUSE=1 --cppflag=-DMELSPEC_OUTPUT_HUGEPAGE=1 \
        > tmp/logs/f2-cpu-profile-$SLURM_JOB_ID.txt
    cp tmp/pipeline-profile/final/results.json tmp/logs/f2-cpu-profile-$SLURM_JOB_ID.json
fi
