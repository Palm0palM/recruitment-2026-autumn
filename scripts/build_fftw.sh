#!/usr/bin/env bash
# Unprivileged, static, double-only FFTW build. Run ON the target CPU, inside
# a Slurm allocation on the cluster; never start this build on its login shell.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${FFTW_PREFIX:-$ROOT/tmp/opt/fftw-3.3.11}"
WORK="$ROOT/tmp/deps"
ARCHIVE="$WORK/downloads/fftw-3.3.11.tar.gz"
SHA=5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1
mkdir -p "$WORK/downloads"
if [ ! -f "$ARCHIVE" ]; then
  curl --fail --location --retry 2 https://www.fftw.org/fftw-3.3.11.tar.gz -o "$ARCHIVE"
fi
printf '%s  %s\n' "$SHA" "$ARCHIVE" | sha256sum -c -
if [ -f "$PREFIX/lib/libfftw3.a" ] && [ -f "$PREFIX/include/fftw3.h" ]; then
  echo "Reuse FFTW: $PREFIX"
  exit 0
fi
mkdir -p "$WORK/src"
if [ ! -d "$WORK/src/fftw-3.3.11" ]; then tar -xzf "$ARCHIVE" -C "$WORK/src"; fi
mkdir -p "$WORK/build-fftw"
cd "$WORK/build-fftw"
# All execution threading disabled. SIMD variants are runtime-dispatched;
# AVX-512 is built only if exposed by this machine (WSL is AVX2-only).
SIMD=(--enable-sse2 --enable-avx --enable-avx2)
if grep -qw avx512f /proc/cpuinfo; then SIMD+=(--enable-avx512); fi
CC="${CC:-gcc}" CFLAGS="-O3 -march=native" \
  "$WORK/src/fftw-3.3.11/configure" --prefix="$PREFIX" \
  --enable-static --disable-shared --disable-fortran --disable-threads \
  --disable-openmp --disable-mpi "${SIMD[@]}"
make -j "${JOBS:-4}"
make install
echo "FFTW installed in $PREFIX (GPL-2.0-or-later; COPYING in source archive)"
