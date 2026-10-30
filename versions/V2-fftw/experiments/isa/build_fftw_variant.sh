#!/usr/bin/env bash
# REVIEW DRAFT: not executed by the ISA subagent.
# Build an isolated FFTW 3.3.11 variant under tmp/fftw-lab/isa only.
# Run only after the main Agent schedules CPU time and reviews all paths/flags.
set -euo pipefail

# Archived script lives under versions/, but dependency builds must stay tmp/.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
LAB="$(realpath -m "${FFTW_ISA_LAB:-$ROOT/tmp/fftw-lab/isa}")"
case "$LAB" in
  "$ROOT/tmp/fftw-lab/isa"|"$ROOT/tmp/fftw-lab/isa/"*) ;;
  *) printf 'ISA build directory must remain under this worktree tmp/fftw-lab/isa\n' >&2; exit 2 ;;
esac
VARIANT="${VARIANT:?set VARIANT=avx2 or avx512}"
PRECISION="${PRECISION:-double}"
JOBS="${JOBS:-1}"
CC="${CC:-gcc}"
case "$VARIANT" in
  avx2|avx512) ;;
  *) printf 'invalid VARIANT=%s\n' "$VARIANT" >&2; exit 2 ;;
esac
case "$PRECISION" in
  double|float) ;;
  *) printf 'invalid PRECISION=%s\n' "$PRECISION" >&2; exit 2 ;;
esac

if ! grep -qw avx2 /proc/cpuinfo; then
  printf 'target does not expose avx2\n' >&2; exit 2
fi
if [[ "$VARIANT" == avx512 ]] && ! grep -qw avx512f /proc/cpuinfo; then
  printf 'AVX-512 variant must be built on an avx512f target\n' >&2; exit 2
fi

WORK="$LAB/work"
ARCHIVE="$WORK/downloads/fftw-3.3.11.tar.gz"
SHA=5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1
SRC="$WORK/src/fftw-3.3.11"
BUILD="$WORK/build-$VARIANT-$PRECISION"
PREFIX="$LAB/prefix/$VARIANT-$PRECISION"
mkdir -p "$WORK/downloads" "$WORK/src" "$LAB/prefix"
if [[ ! -f "$ARCHIVE" ]]; then
  curl --fail --location --retry 2 \
    https://www.fftw.org/fftw-3.3.11.tar.gz -o "$ARCHIVE"
fi
printf '%s  %s\n' "$SHA" "$ARCHIVE" | sha256sum -c -
if [[ ! -d "$SRC" ]]; then
  tar -xzf "$ARCHIVE" -C "$WORK/src"
fi

# Never silently reuse a stale library with unknown compiler flags/wisdom.
if [[ -e "$BUILD" || -e "$PREFIX" ]]; then
  printf 'refusing to reuse existing paths; choose a fresh reviewed LAB copy:\n%s\n%s\n' \
    "$BUILD" "$PREFIX" >&2
  exit 2
fi
mkdir -p "$BUILD" "$PREFIX"

SIMD=(--enable-avx --enable-avx2)
if [[ "$PRECISION" == float ]]; then SIMD+=(--enable-sse); else SIMD+=(--enable-sse2); fi
# Identical scalar/compiler baseline for BOTH libraries. FFTW adds the
# required -mavx512f to its enabled AVX-512 codelet compilation units only.
# Audit the build log before attributing a result to enabled codelets.
CFLAGS='-O3 -march=x86-64-v3'
if [[ "$VARIANT" == avx512 ]]; then
  SIMD+=(--enable-avx512)
fi
PRECISION_FLAG=()
if [[ "$PRECISION" == float ]]; then
  PRECISION_FLAG=(--enable-float)
fi

{
  printf 'variant=%s\nprecision=%s\nCC=%s\nCFLAGS=%s\n' \
    "$VARIANT" "$PRECISION" "$CC" "$CFLAGS"
  printf 'SIMD='; printf '%q ' "${SIMD[@]}"; printf '\n'
  printf 'archive_sha256=%s\n' "$SHA"
  "$CC" --version
  uname -a
  grep -m1 'Model name' /proc/cpuinfo || true
} | tee "$LAB/config-$VARIANT-$PRECISION.txt"

cd "$BUILD"
CC="$CC" CFLAGS="$CFLAGS" "$SRC/configure" \
  --prefix="$PREFIX" \
  --enable-static --disable-shared --disable-fortran \
  --disable-threads --disable-openmp --disable-mpi \
  "${PRECISION_FLAG[@]}" "${SIMD[@]}"
make -j "$JOBS"
make install
printf 'installed isolated variant: %s\n' "$PREFIX"
