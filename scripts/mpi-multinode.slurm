#!/usr/bin/env bash
#SBATCH --job-name=mpi-two-node-probe
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --nodelist=hepnode[1-3]
#SBATCH --time=00:05:00
#SBATCH --output=tmp/logs/mpi-two-node-%j.out
#SBATCH --error=tmp/logs/mpi-two-node-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
O=/cache/toolchain/openmpi-4.1.6
U=/cache/toolchain/ucx-1.18.1
export PATH="$O/bin:$PATH"
export LD_LIBRARY_PATH="$O/lib:$U/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPAL_PREFIX="$O" OPAL_LIBDIR="$O/lib"
export OMPI_MCA_rmaps_base_oversubscribe=1
echo "job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST"
scontrol show hostname "$SLURM_JOB_NODELIST"

g++ -std=c++17 -O2 -I"$O/include" scripts/mpi_probe.cpp \
  -L"$O/lib" -Wl,-rpath,"$O/lib" -lmpi -o tmp/bin/mpi_probe || exit 1
run_case() {
  local name="$1"; shift
  echo "== $name =="
  "$@"
  echo "exit[$name]=$?"
  return 0
}
run_case srun-two-node srun -N2 -n2 --ntasks-per-node=1 -c16 --hint=nomultithread --mpi=pmix ./tmp/bin/mpi_probe
run_case mpirun-two-node "$O/bin/mpirun" -np 2 --map-by ppr:1:node:pe=16 --bind-to none ./tmp/bin/mpi_probe
run_case mpirun-two-node-tcp "$O/bin/mpirun" -np 2 --map-by ppr:1:node:pe=16 --bind-to none \
  --mca pml ob1 --mca btl self,tcp ./tmp/bin/mpi_probe
exit 0
