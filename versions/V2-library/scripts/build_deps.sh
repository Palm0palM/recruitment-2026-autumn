#!/usr/bin/env bash
# Unprivileged dependency bootstrap for the C++ library experiment.
# Run on a compute node, never on the cluster login shell. Not used by the
# Python winner; retained so the C++ comparison is reproducible and auditable.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEPS="$ROOT/tmp/deps"; OPT="$ROOT/tmp/opt"
mkdir -p "$DEPS/downloads" "$DEPS/src" "$OPT"
FFTW_VER=3.3.11; FFTW_SHA=5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1
EIGEN_VER=3.4.0
OPENBLAS_VER=0.3.34
cd "$DEPS/downloads"
[ -f "fftw-$FFTW_VER.tar.gz" ] || curl -fsSLO "https://www.fftw.org/fftw-$FFTW_VER.tar.gz"
printf '%s  %s\n' "$FFTW_SHA" "fftw-$FFTW_VER.tar.gz" | sha256sum -c -
[ -f "eigen-$EIGEN_VER.tar.gz" ] || curl -fsSL "https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VER/eigen-$EIGEN_VER.tar.gz" -o "eigen-$EIGEN_VER.tar.gz"
[ -f "openblas-$OPENBLAS_VER.tar.gz" ] || curl -fsSL "https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v$OPENBLAS_VER.tar.gz" -o "openblas-$OPENBLAS_VER.tar.gz"
cd "$DEPS/src"
[ -d "fftw-$FFTW_VER" ] || tar -xzf "../downloads/fftw-$FFTW_VER.tar.gz"
[ -d "eigen-$EIGEN_VER" ] || tar -xzf "../downloads/eigen-$EIGEN_VER.tar.gz"
[ -d "openblas-$OPENBLAS_VER" ] || tar -xzf "../downloads/openblas-$OPENBLAS_VER.tar.gz"
if [ ! -f "$OPT/fftw-float/lib/libfftw3f.a" ]; then
  mkdir -p "$DEPS/build-fftw-float"; cd "$DEPS/build-fftw-float"
  CC=gcc CFLAGS="-O3 -march=native" "$DEPS/src/fftw-$FFTW_VER/configure" \
    --prefix="$OPT/fftw-float" --enable-float --enable-static --disable-shared \
    --disable-fortran --disable-threads --disable-openmp --disable-mpi \
    --enable-sse2 --enable-avx --enable-avx2
  make -j"${JOBS:-8}"; make install
fi
if [ ! -f "$OPT/openblas/lib/libopenblas.a" ]; then
  cd "$DEPS/src/openblas-$OPENBLAS_VER"
  make -j"${JOBS:-8}" NO_LAPACK=1 NO_SHARED=1 NOFORTRAN=1 USE_OPENMP=1 \
    TARGET=SKYLAKEX PREFIX="$OPT/openblas" install
fi
mkdir -p "$OPT/eigen/include"
ln -sfn "$DEPS/src/eigen-$EIGEN_VER" "$OPT/eigen/include/eigen3"
echo "Deps ready: FFTW=$OPT/fftw-float EIGEN=$OPT/eigen/include/eigen3 BLAS=$OPT/openblas"
