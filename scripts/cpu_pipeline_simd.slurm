#!/bin/bash
#SBATCH -J cpu-simd-lab
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=tmp/logs/pipeline-%j.out
#SBATCH --error=tmp/logs/pipeline-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source tmp/session-env.sh
export OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
echo "job=$SLURM_JOB_ID node=$(hostname) mode=${MODE:-profile}"
grep Cpus_allowed_list /proc/self/status
lscpu
g++ --version
sha256sum src/main.cpp src/melspec.cpp validate.py tmp/control/melspec.cpp
scratch=$(mktemp -d "/tmp/cpu-simd-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
export TMPDIR="$scratch"
mkdir -p tmp/bin
flags=(-std=c++17 -O3 -march=native -pthread -Iinclude -DMELSPEC_FP32_BATCH=16)
if [ "${MODE:-profile}" = profile ]; then
    "$PY" -u scripts/cpu_pipeline_profile.py --source tmp/control/melspec.cpp --runs 3
    cp tmp/pipeline-profile/control/results.json "tmp/logs/profile-$SLURM_JOB_ID.json"
    g++ "${flags[@]}" src/main.cpp tmp/control/melspec.cpp -o tmp/bin/frozen-b16
    "$PY" -u scripts/bench.py tmp/bin/frozen-b16 --runs 3 --no-librosa
elif [ "$MODE" = reuse ]; then
    g++ "${flags[@]}" src/main.cpp tmp/control/melspec.cpp -o tmp/bin/frozen-b16
    g++ "${flags[@]}" -DMELSPEC_PACK_REUSE=1 src/main.cpp src/melspec.cpp -o tmp/bin/reuse-b16
    sha256sum tmp/bin/frozen-b16 tmp/bin/reuse-b16
    "$PY" -u scripts/cpu_pipeline_check.py tmp/bin/frozen-b16 tmp/bin/reuse-b16 --exact
    "$PY" -u scripts/smoke.py tmp/bin/reuse-b16
    "$PY" -u scripts/multi_bench.py frozen=tmp/bin/frozen-b16 reuse=tmp/bin/reuse-b16 --runs 8 \
        --out "tmp/logs/reuse-$SLURM_JOB_ID.json"
    "$PY" -u scripts/cpu_pipeline_profile.py --source src/melspec.cpp --tag reuse --runs 2 \
        --cppflag=-DMELSPEC_PACK_REUSE=1
    cp tmp/pipeline-profile/reuse/results.json "tmp/logs/reuse-profile-$SLURM_JOB_ID.json"
elif [ "$MODE" = reuse-confirm ]; then
    g++ "${flags[@]}" src/main.cpp tmp/control/melspec.cpp -o tmp/bin/frozen-b16
    g++ "${flags[@]}" -DMELSPEC_PACK_REUSE=1 src/main.cpp src/melspec.cpp -o tmp/bin/reuse-b16
    g++ "${flags[@]}" -DMELSPEC_PACK_REUSE=1 scripts/cpu_pipeline_interface_check.cpp src/melspec.cpp -o tmp/bin/interface-reuse
    tmp/bin/interface-reuse
    g++ -std=c++17 -O1 -g -march=native -pthread -Iinclude -DMELSPEC_FP32_BATCH=16 \
        -DMELSPEC_PACK_REUSE=1 -fsanitize=address,undefined -fno-omit-frame-pointer \
        scripts/cpu_pipeline_interface_check.cpp src/melspec.cpp -o tmp/bin/sanitize-reuse
    ASAN_OPTIONS=detect_leaks=1 tmp/bin/sanitize-reuse
    "$PY" -u scripts/multi_bench.py frozen=tmp/bin/frozen-b16 reuse=tmp/bin/reuse-b16 --runs 8 \
        --out "tmp/logs/reuse-confirm-$SLURM_JOB_ID.json"
    "$PY" -u validate.py tmp/bin/reuse-b16 --runs 3 | tee "tmp/logs/official-reuse-$SLURM_JOB_ID.txt"
    grep -q 'Result: 4/4 PASSED' "tmp/logs/official-reuse-$SLURM_JOB_ID.txt"
elif [ "$MODE" = hugepage ] || [ "$MODE" = hugepage-confirm ]; then
    cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag
    cat /sys/kernel/mm/transparent_hugepage/hpage_pmd_size /proc/sys/kernel/numa_balancing
    perf stat -e cycles true 2>&1 || true
    g++ "${flags[@]}" src/main.cpp tmp/control/melspec.cpp -o tmp/bin/frozen-b16
    g++ "${flags[@]}" -DMELSPEC_OUTPUT_HUGEPAGE=1 src/main.cpp src/melspec.cpp -o tmp/bin/hugepage-b16
    sha256sum tmp/bin/frozen-b16 tmp/bin/hugepage-b16
    "$PY" -u scripts/cpu_pipeline_check.py tmp/bin/frozen-b16 tmp/bin/hugepage-b16 --exact
    g++ -std=c++17 -O1 -g -march=native -pthread -Iinclude -DMELSPEC_FP32_BATCH=16 \
        -DMELSPEC_OUTPUT_HUGEPAGE=1 -fsanitize=address,undefined -fno-omit-frame-pointer \
        scripts/cpu_pipeline_interface_check.cpp src/melspec.cpp -o tmp/bin/sanitize-hugepage
    ASAN_OPTIONS=detect_leaks=1 tmp/bin/sanitize-hugepage
    "$PY" -u scripts/multi_bench.py frozen=tmp/bin/frozen-b16 hugepage=tmp/bin/hugepage-b16 --runs 8 \
        --out "tmp/logs/$MODE-$SLURM_JOB_ID.json"
    if [ "$MODE" = hugepage ]; then
        "$PY" -u scripts/cpu_pipeline_profile.py --source src/melspec.cpp --tag hugepage --runs 2 \
            --cppflag=-DMELSPEC_OUTPUT_HUGEPAGE=1
        cp tmp/pipeline-profile/hugepage/results.json "tmp/logs/hugepage-profile-$SLURM_JOB_ID.json"
    else
        "$PY" -u validate.py tmp/bin/hugepage-b16 --runs 3 | tee "tmp/logs/official-hugepage-$SLURM_JOB_ID.txt"
        grep -q 'Result: 4/4 PASSED' "tmp/logs/official-hugepage-$SLURM_JOB_ID.txt"
    fi
elif [ "$MODE" = final ]; then
    cat /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/defrag
    g++ "${flags[@]}" src/main.cpp tmp/control/melspec.cpp -o tmp/bin/frozen-b16
    for variant in reuse hugepage combined; do
        extra=()
        if [ "$variant" != hugepage ]; then extra+=(-DMELSPEC_PACK_REUSE=1); fi
        if [ "$variant" != reuse ]; then extra+=(-DMELSPEC_OUTPUT_HUGEPAGE=1); fi
        g++ "${flags[@]}" "${extra[@]}" src/main.cpp src/melspec.cpp -o "tmp/bin/$variant-b16"
        sha256sum "tmp/bin/$variant-b16"
        "$PY" -u scripts/cpu_pipeline_check.py tmp/bin/frozen-b16 "tmp/bin/$variant-b16" --exact
        g++ -std=c++17 -O1 -g -march=native -pthread -Iinclude -DMELSPEC_FP32_BATCH=16 \
            "${extra[@]}" -fsanitize=address,undefined -fno-omit-frame-pointer \
            scripts/cpu_pipeline_interface_check.cpp src/melspec.cpp -o "tmp/bin/sanitize-$variant"
        ASAN_OPTIONS=detect_leaks=1 "tmp/bin/sanitize-$variant"
        "$PY" -u scripts/multi_bench.py frozen=tmp/bin/frozen-b16 "$variant=tmp/bin/$variant-b16" --runs 8 \
            --out "tmp/logs/final-$variant-$SLURM_JOB_ID.json"
    done
    "$PY" -u scripts/cpu_pipeline_synthetic_bench.py tmp/bin/frozen-b16 tmp/bin/reuse-b16 \
        --out "tmp/logs/synthetic-reuse-$SLURM_JOB_ID.json"
    "$PY" -u scripts/multi_bench.py reuse=tmp/bin/reuse-b16 combined=tmp/bin/combined-b16 --runs 8 \
        --out "tmp/logs/incremental-$SLURM_JOB_ID.json"
    "$PY" -u scripts/cpu_pipeline_profile.py --source src/melspec.cpp --tag combined --runs 2 \
        --cppflag=-DMELSPEC_PACK_REUSE=1 --cppflag=-DMELSPEC_OUTPUT_HUGEPAGE=1
    cp tmp/pipeline-profile/combined/results.json "tmp/logs/combined-profile-$SLURM_JOB_ID.json"
    for variant in hugepage combined; do
        "$PY" -u validate.py "tmp/bin/$variant-b16" --runs 1 | tee "tmp/logs/official-$variant-$SLURM_JOB_ID.txt"
        grep -q 'Result: 4/4 PASSED' "tmp/logs/official-$variant-$SLURM_JOB_ID.txt"
    done
else
    echo 'Set MODE=profile until a separately reviewed experiment is available.' >&2
    exit 2
fi
