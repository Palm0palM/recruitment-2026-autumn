#!/usr/bin/env bash
# REVIEW DRAFT: read-only inspection of supplied artifacts; reports go under LAB.
# Text matches are triage, NOT a proof of ISA absence. Review full disassembly.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
LAB="$ROOT/tmp/fftw-lab/isa"
if [[ $# -eq 0 ]]; then
  printf 'usage: bash %s ARTIFACT [ARTIFACT ...]\n' "$0" >&2
  exit 2
fi
OUT="$LAB/audit"
mkdir -p "$OUT"
: > "$OUT/command-versions.txt"
for tool in file sha256sum objdump readelf nm; do
  command -v "$tool" >> "$OUT/command-versions.txt"
done
objdump --version | head -n 1 >> "$OUT/command-versions.txt"

index=0
for artifact in "$@"; do
  index=$((index + 1))
  if [[ ! -f "$artifact" ]]; then
    printf 'not a regular file: %s\n' "$artifact" >&2
    exit 2
  fi
  stem=$(printf '%03d-%s' "$index" "$(basename "$artifact")" | tr '/ ' '__')
  printf '%s\n' "$artifact" > "$OUT/$stem.path.txt"
  file "$artifact" > "$OUT/$stem.file.txt"
  sha256sum "$artifact" > "$OUT/$stem.sha256.txt"
  readelf -d -n -W "$artifact" > "$OUT/$stem.dynamic-notes.txt" 2>&1 || true
  readelf -Ws -W "$artifact" > "$OUT/$stem.symbols.txt" 2>&1 || true
  nm -u "$artifact" > "$OUT/$stem.undefined.txt" 2>&1 || true
  objdump -d -w "$artifact" > "$OUT/$stem.disassembly.txt"

  # AT&T objdump triage. zmm/opmask/embedded codelet names can reveal AVX-512,
  # but xmm/ymm EVEX forms and other ISA extensions require manual review.
  grep -E -i 'avx512|%zmm[0-9]+|\{%k[0-7]\}|\{k[0-7]\}' \
    "$OUT/$stem.disassembly.txt" \
    > "$OUT/$stem.suspect-avx512.txt" || true
  grep -E -i 'fftw|NEEDED' "$OUT/$stem.dynamic-notes.txt" \
    > "$OUT/$stem.fftw-needed.txt" || true
  printf 'audited %s -> %s (review suspects and full disassembly)\n' \
    "$artifact" "$OUT/$stem"
done
