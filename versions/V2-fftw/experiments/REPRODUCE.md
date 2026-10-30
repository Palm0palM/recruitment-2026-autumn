# Reproducing the native-subagent round

## 1. Models, permissions, and provenance

OpenCode 2.2.1/V2 configuration is in `.opencode/agents/`; primary is `openai/gpt-6.1-sol`,
all five worker definitions explicitly select `ppio/moonshotai/kimi-k3.1-pro-highspeed#high`.
One read-only probe plus four specialist child sessions were used. Nine native subagent calls reused
these five sessions; workers authored prototypes/tests but did not execute shell or CPU workloads.
Resolved definitions and actual provider/model/variant plus nonzero reasoning usage are independently
recorded in `agent-verification.json` and `session-verification.json`. No manual extra sessions/worktrees.

The initial `worker-source-sha256.json` files identify the initial worker snapshots at commit `f9d2080`,
not later amended NOTES. Follow-up prototypes are versioned in subsequent commits; do not mistake an
initial NOTES digest for a current-file digest. Production ports and decisions belong to the parent.

## 2. Dependencies (no sudo, no shared environment changes)

Double FFTW 3.3.11: use `scripts/build_fftw.sh` and the existing top-level `REPRODUCE.md`.
Float FFTW uses the SAME pinned source archive SHA-256
`5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1`, a separate build directory/prefix:

```sh
# On cluster: compilation itself belongs inside a Slurm allocation.
mkdir -p tmp/deps/build-fftw-float
cd tmp/deps/build-fftw-float
CC=gcc CFLAGS='-O3 -march=native' ../src/fftw-3.3.11/configure \
  --prefix="$SLURM_SUBMIT_DIR/tmp/opt/fftw-3.3.11-float" \
  --enable-float --enable-static --disable-shared --disable-fortran \
  --disable-threads --disable-openmp --disable-mpi \
  --enable-sse --enable-avx --enable-avx2 --enable-avx512
make -j8
make install
```

For a local build substitute this worktree's absolute prefix; never reuse someone else's build products.
Use `source tmp/session-env.sh` then `$PY` for local Python. Do not install/upgrade the shared venv.
Native compilation is target-dependent, not a portable-binary promise. Both FFTW precisions are
GPL-2.0-or-later; prior source/build-distribution and original-code license compatibility caveats still apply.
No FFTW archives/libraries, executables, PCM/expected arrays, or perf.data are committed.

## 3. Release builds and explicit numerical contract

```sh
make -B FFT_PRECISION=double FFT_MODE=r2c FFTW_PREFIX="$DOUBLE_PREFIX" EXEC=tmp/bin/a2
make -B FFT_PRECISION=float FFT_POWER=float FFTW_PREFIX="$FLOAT_PREFIX" EXEC=tmp/bin/float-fp
make -B FFT_PRECISION=float FFT_POWER=double FFTW_PREFIX="$FLOAT_PREFIX" EXEC=tmp/bin/float-dp
make -B FFT_PRECISION=float FFT_POWER=float FFTW_WISDOM=1 \
  FFTW_PREFIX="$FLOAT_PREFIX" EXEC=tmp/bin/float-wisdom
```

Default remains all-double A2; float is an explicit approximate arithmetic option. Float still uses double
Hann construction/input multiplication, double Mel weights/accumulation and double API output. It does not
promise double accuracy for arbitrary-magnitude public-API vectors; tested task-domain inputs are PCM16.
All task correctness comparisons retain `rtol=1e-3, atol=1e-4`. Neither max absolute error alone nor a previous
double PASS proves that a float configuration passes. Keep A1 C2C double as a separately buildable control.
Use `-B` whenever changing FFTW libraries; object paths separate mode/planner/precision/power/wisdom.

## 4. Offline wisdom, online cold CLI

```sh
make FFT_PRECISION=float FFTW_PREFIX="$FLOAT_PREFIX" wisdom-generator
LIB_ID=$(sha256sum "$FLOAT_PREFIX/lib/libfftw3f.a" | cut -d' ' -f1)
mkdir -p tmp/target-wisdom
tmp/bin/fftwf-wisdom-generate --size 8192 --planner measure \
  --library-id "$LIB_ID" --output tmp/target-wisdom/r2c-8192.wisdom
# Repeat for required FFT sizes; use a DIFFERENT directory for --planner patient.
unset MELSPEC_FFTWF_WISDOM
MELSPEC_FFTWF_LIBRARY_ID="$LIB_ID" \
MELSPEC_FFTWF_WISDOM_DIRECTORY="$PWD/tmp/target-wisdom" \
  tmp/bin/float-wisdom input.pcm tmp/output.bin --n_fft 8192
```

The generator never reads PCM. Runtime only imports a matching envelope and requests
`FFTW_WISDOM_ONLY | FFTW_ESTIMATE`; that accepts the offline MEASURE/PATIENT plan without measuring online.
Check `status=loaded`; a matching file is not assumed merely because an environment variable exists.
Absent/corrupt/incompatible files or missing applicable plan yield explicit safe ESTIMATE fallback.
Single-file `MELSPEC_FFTWF_WISDOM` and directory mode are mutually exclusive. Do not confuse the
float `MELSPEC_FFTWF_*` namespace with double experiment `MELSPEC_FFTW_*` variables.

Binding covers exact FFTW version/compiler/library ID, precision, x86 CPUID/XCR0 (APIC ID excluded),
size, fixed single-transform out-of-place R2C layout and alignment class. Library ID must truthfully identify
the actual linked archive; it is not automatic runtime attestation. FNV protects accidental corruption,
not malicious input. Non-x86 CPU identity is intentionally coarse; regenerate/revalidate on another target.
Single-transform wisdom must NOT be presumed applicable to batch/interleaved layouts or another precision.

Reported wisdom process timings include process startup, file import and planning, compute, read/write;
they EXCLUDE separate offline generation. If deployment must generate wisdom inside every timed process,
these gains do not apply: retain float ESTIMATE, or include and amortize preparation cost explicitly.

## 5. Tests and serialized cluster reproduction

Local parent-only gates: `scripts/fftw_check.py`, `precision/stress_precision.py`,
`batch/check_boundaries.py`, `scripts/fftw_wisdom_check.py`. Temporary arrays belong in TMPDIR.
The float direct-FFT numerical contract is approximate; the existing strict direct-DFT helper test remains
a double-only control. Six-layout batch boundaries and default direct-DFT checks are recorded under raw/.

Submit ONLY after the preceding benchmark job ends, in this experiment's isolated remote directory:

```sh
sbatch --wait -w hepnode2 scripts/fftw-lab-cluster.slurm  # initial dependencies/independent candidates
sbatch --wait -w hepnode2 scripts/fftw-lab-isa.slurm      # controlled ISA + original batch
sbatch --wait -w hepnode2 scripts/fftw-lab-final.slurm    # exact integrated variants + revised layouts
# After final scoring has completed, diagnostic only:
FINAL_JOB=<completed-job-id> sbatch --wait -w hepnode2 scripts/fftw-lab-profile.slurm
```

Every script requests one task/16 CPUs/nomultithread and no --mem. No login-node computation or concurrent
benchmark jobs. ISA retry preserves failed prefixes and uses a new `tmp/fftw-lab/isa/v3/` root. Test flags,
actual plans, compiler commands, hashes, raw repeats, official output and rejected attempts are retained.
The final harness gives each candidate its own direct-exec environment (no timing-distorting wrapper),
and refuses to score a wisdom-labelled candidate unless it reports successful import without fallback.
The separate profile job selects the float finalist from uninstrumented release medians, then builds a
distinct instrumented executable and generated read-only-main wrapper. Profile numbers never rank candidates;
worker-stage percentages are summed worker wall counters, not percentages of whole-process wall time.

Memory is a separate invocation: sampled post-exec `/proc/<pid>/status` VmHWM, not inherited wait4 RSS.
Sampling every 5 ms may miss a very short terminal peak. Whole-process peak can be dominated by the
protected main's final double-to-float write buffer even when FFT scratch differs significantly.
