#!/usr/bin/env bash
#SBATCH --job-name=melspec-mpi-probe
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:05:00
#SBATCH --output=tmp/logs/mpi-probe-%j.out
#SBATCH --error=tmp/logs/mpi-probe-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p tmp/logs tmp/bin
O=/cache/toolchain/openmpi-4.1.6
U=/cache/toolchain/ucx-1.18.1
export PATH="$O/bin:$PATH"
export LD_LIBRARY_PATH="$O/lib:$U/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPAL_PREFIX="$O"
export OPAL_LIBDIR="$O/lib"
export OMPI_MCA_rmaps_base_oversubscribe=1

echo "job=$SLURM_JOB_ID node=$(hostname) tasks=$SLURM_NTASKS"
srun --version | head -2
srun --mpi=list 2>&1 || true
"$O/bin/ompi_info" --version
"$O/bin/mpicxx" --showme 2>&1 || true
printf 'wrapper-prefix: '; grep '^prefix=' "$O/lib/pkgconfig/ompi.pc" || true
printf 'compiler: '; g++ --version | head -1

echo '== compile MPI probe inside allocation =='
g++ -std=c++17 -O2 -I"$O/include" scripts/mpi_probe.cpp \
  -L"$O/lib" -Wl,-rpath,"$O/lib" -lmpi -o tmp/bin/mpi_probe || exit 1
ldd tmp/bin/mpi_probe | grep -E 'not found|mpi|open-rte|open-pal|pmix' || true

run_case() {
  local name="$1"
  shift
  echo "== $name =="
  "$@"
  local rc=$?
  echo "exit[$name]=$rc"
  return 0
}
run_case singleton ./tmp/bin/mpi_probe
run_case srun-default srun -N1 -n2 --cpus-per-task=1 --hint=nomultithread ./tmp/bin/mpi_probe
run_case srun-pmix srun -N1 -n2 --cpus-per-task=1 --hint=nomultithread --mpi=pmix ./tmp/bin/mpi_probe
run_case mpirun-default "$O/bin/mpirun" -np 2 --bind-to none ./tmp/bin/mpi_probe
run_case mpirun-tcp "$O/bin/mpirun" -np 2 --bind-to none \
  --mca pml ob1 --mca btl self,tcp ./tmp/bin/mpi_probe
exit 0
