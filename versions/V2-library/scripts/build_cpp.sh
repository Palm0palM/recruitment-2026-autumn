#!/usr/bin/env bash
# Build the standalone library-first C++ reference against system-style libs.
# Override EIGEN_PREFIX/FFTW_PREFIX/OPENBLAS_PREFIX for an unprivileged build.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CXX="${CXX:-g++}"
OUT="${OUT:-$ROOT/tmp/bin/cpp-library}"
CXXFLAGS="${CXXFLAGS:--O3 -march=native -pthread -DEIGEN_DONT_PARALLELIZE}"
if [ -n "${EIGEN_PREFIX:-}" ]; then EIGEN_CFLAGS="-I$EIGEN_PREFIX/include/eigen3"; else EIGEN_CFLAGS="$(pkg-config --cflags eigen3)"; fi
if [ -n "${OPENBLAS_PREFIX:-}" ]; then
  BLAS_CFLAGS="-I$OPENBLAS_PREFIX/include -I$OPENBLAS_PREFIX/include/openblas"
  BLAS_LIBS="$OPENBLAS_PREFIX/lib/libopenblas.a"
else
  BLAS_CFLAGS="$(pkg-config --cflags openblas)"
  BLAS_LIBS="$(pkg-config --libs openblas)"
fi
if [ -n "${FFTW_PREFIX:-}" ]; then
  FFTW_CFLAGS="-I$FFTW_PREFIX/include"; FFTW_LIBS="$FFTW_PREFIX/lib/libfftw3f.a"
else
  FFTW_CFLAGS="$(pkg-config --cflags fftw3f 2>/dev/null || pkg-config --cflags fftw3)"
  FFTW_LIBS="$(pkg-config --libs fftw3f 2>/dev/null || echo -lfftw3f)"
fi
mkdir -p "$(dirname "$OUT")"
set -x
$CXX -std=c++17 $CXXFLAGS -I"$ROOT/include" $EIGEN_CFLAGS $BLAS_CFLAGS $FFTW_CFLAGS \
  "$ROOT/versions/V2-library/impl/cpp/library_melspec.cpp" \
  -o "$OUT" $FFTW_LIBS $BLAS_LIBS -lm
