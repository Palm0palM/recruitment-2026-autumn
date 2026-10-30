#!/usr/bin/env bash
# Research-only launcher used to run the UNMODIFIED validate.py against the
# independent MPI driver. This wrapper is not claimed to be a final submission.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
O="${OPENMPI_HOME:-/cache/toolchain/openmpi-4.1.6}"
RANKS="${MELSPEC_MPI_RANKS:-2}"
CPUS="${MELSPEC_MPI_CPUS:-16}"
export MELSPEC_MPI_IO="${MELSPEC_MPI_IO:-scatter}"
exec "$O/bin/mpirun" -np "$RANKS" --map-by "ppr:1:node:pe=$CPUS" \
  --bind-to none "$ROOT/tmp/bin/mpi-melspec" "$@"
