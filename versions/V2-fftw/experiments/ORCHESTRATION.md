# Native Kimi Subagent experiments

## Baseline and model verification

- Parent: GPT-6.1 Sol, `openai/gpt-6.1-sol`.
- OpenCode CLI/service: **2.2.1**, V2 Markdown frontmatter and ordered `permissions` rules.
- Explicit worker model: **`ppio/moonshotai/kimi-k3.1-pro-highspeed#high`** (not the non-highspeed model).
- Agent-level request overlays are not used: V2 documentation says they are not currently sent.
- Read-only probe session: `ses_e749e9920ffdjN7upUwUV7hLVO`; no model override in invocation,
  therefore this tests the configured agent selection, not a tool-supplied substitute.
- Service session metadata independently confirms provider `ppio`, model
  `moonshotai/kimi-k3.1-pro-highspeed`, variant `high`, **741 reasoning tokens**.
  Probe returned correct A2 commit, four official timings, planning-cost explanation, and 16×12=192.
  This is metadata evidence of enabled reasoning, not a claim based on model self-identification.
- Formal A2 ESTIMATE code baseline remains `b4552f3962751b5a02d2aa56798907c5a1c29afc`;
  branch baseline before this orchestration is `61f58a6`. No new worktrees/sessions are created manually;
  native subagents own automatically-created child sessions and share this worktree.

## Isolation and scheduling

| Agent | Only writable experiment directory | Background child session |
|---|---|---|
| fftw-wisdom | tmp/fftw-lab/wisdom/ | ses_e749dabfcffdQjt9ttwQWC9jYB |
| fftw-batch | tmp/fftw-lab/batch/ | ses_e749d5f8dffebYfSQpPKO9wtiT |
| fftw-precision | tmp/fftw-lab/precision/ | ses_e749ccc45ffdmpoPyXiy2S45M5 |
| fftw-isa | tmp/fftw-lab/isa/ | ses_e749aea9efferIvVtrAJ2Tx5xj |

Each definition explicitly pins the Kimi high variant. Broad edit-deny precedes its sole private-directory
exception. Shell/interpreters/external directories require approval; direct Git, SSH, sbatch/srun, Code Mode
and recursive subagent delegation are denied. Other command forms require approval, not automatic execution.
Initial assignments prohibit shell/compilation/testing:
parallelism is research and prototype authoring only, preventing CPU benchmark interference.
These are permission controls, NOT a filesystem/container sandbox; never broadly approve an arbitrary
interpreter or script. Inspection of resolved agent definitions confirms agent rules follow global rules.

Parent alone builds/tests prototypes, reviews numerical correctness, schedules local measurements,
submits cluster jobs, modifies production files, archives results and commits. Protected main/validator/data
and shared Python environment remain unchanged. Effective research sources and prototype code are
archived here only after review; raw binaries, PCM, caches and perf.data stay under tmp/.

## Acceptance gates

1. Review private source and exact compile/link/dependency requirements. No integration merely on a worker claim.
2. Parent compiles each candidate separately. Run random/full-scale/silence/impulse/Nyquist/boundary tests;
   float variants must satisfy original `rtol=1e-3, atol=1e-4`, never relaxed tolerance.
3. Serialize local performance runs and retain same-environment A2 ESTIMATE; record all repeats, not only minima.
4. Only passing candidates reach same-node Slurm comparison (1 task, 16 CPU, nomultithread, no mem).
5. Integrate individually, run complete official validator, then test combinations. Independent gains do not multiply.

Configuration and live read-only probe verified; three prototype-authoring jobs launched,
then ISA research launched without CPU work. Resolved-rule evaluations are in `agent-verification.json`;
they are configuration checks, not a claim of OS sandboxing or destructive runtime penetration tests.
Parent rebuilt the A2 baseline in its own orchestrator directory, SHA-256 unchanged:
`97594fa7f89161aae8e41bb086dee0736ec2839cf83f59556947188c0f57720c`.
All four workers returned prototypes/research without executing shell or CPU work. Parent-only local
compilation and tests are recorded in REVIEW.md and raw/. All four experiments reached cluster evaluation;
opt-in precision and safe float wisdom are integrated, while batch and AVX2-only changes were rejected.
Job 151 completed all 180 release comparisons; job 209 completed its last two official validators (both 4/4)
and all memory checks in 3:35, without re-running the matrix. The user explicitly bounded closeout to validation/records,
not more experiments or profiling. This round is complete; no more jobs are scheduled. See RESULTS.md. Nine native calls reused
the five listed child sessions; every specialist has finished. Actual per-worker models and reasoning usage are preserved
in session-verification.json (not just configuration declarations).

Security note: resolved model-list API responses include credentials. An early diagnostic printed such a field;
the retained tmp file was sanitized, no credentials are added to configuration/Git, and key rotation was recommended.
Subsequent evidence collection uses a whitelist (model/variant/status/token metadata), never raw provider settings.

Docs: https://opencode.ai/v2/docs/agents , https://opencode.ai/v2/docs/permissions ,
https://opencode.ai/v2/docs/providers , https://opencode.ai/v2/docs/models .
