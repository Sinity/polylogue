---
created: 2026-07-16
purpose: Operate bounded shared-worktree implementation waves from adjudicated dossiers
status: implementation-ready-after-realized-baseline-reconciliation
project: polylogue
---

# Shared-worktree orchestration

`wave.py` is a thin adapter over Sinnix's existing attested
`run_agent_prompt.sh`. It owns four operations only:

- `validate`: reject ambiguous fields, broad paths/tests, missing required
  reads, same-wave write/read collisions, and Sol-as-worker routing;
- `render`: write byte-stable targeted prompts and run metadata;
- `run`: execute sequential waves with at most four Codex workers in parallel,
  passing model/effort and structured-output schema to the attested launcher;
- `status`: aggregate launcher attestations, structured completion receipts,
  and assigned-file before/after hashes.

It is not a scheduler DSL, agent framework, filesystem sandbox, coverage
database, or correctness authority. Completion receipts remain advisory; Sol
reviews the actual source and combined diff.

It is also not the mutation-certification lane. Successful certification must
temporarily change production and finish with no delta; this runner correctly
requires non-blocked implementation jobs to leave an assigned-file delta.
Certification therefore uses a separate direct attested Terra/high job in a
disposable worktree.

## Manifest contract

A manifest is a JSON array. Every job contains exactly:

`id`, `wave`, `model`, `effort`, `mission`, `required_reads`, `write_files`,
`avoid_files`, `acceptance`, and `focused_tests`.

All paths are exact repository-relative files. `required_reads` must exist at
validation time. `write_files` may name planned files, but never directories or
globs. Writes are disjoint within a wave. Only named `devtools test ...`
commands are accepted. Model and high effort are explicit on every job.

The prepared first-wave manifest intentionally fails validation until the
completed `polylogue-1xc.14.1`/`1xc.14` and `b054.1.1.3`–`.5` outcomes are
present and Sol has refreshed exact canary/receipt symbols. Its foundation job
must consume the realized workload profile, archive profile, named tiers,
C-03, and shared receipts. That is a real prerequisite rather than permission
to copy or partially reconstruct the implementation.

The deliberate missing read is
`.local/testsuite-diet/reconciliation/realized-baseline.json`. Sol creates it
only after checking the merged git head, upstream merge/receipt identities,
resolved canaries, and refreshed dossier hashes. Its presence prevents
accidental launch from the unmerged schema branch; it does not replace source
review.

Until the runner validates that receipt's contents against the current head and
interrupts timed-out workers through the attested job-control helper, real runs
must remain coordinator-supervised. These are prerequisites for unattended
fanout, not reasons to weaken the manifest or use raw PID kills.

## Commands

```bash
python .agent/scratch/testsuite_diet/orchestration/wave.py \
  validate .agent/scratch/testsuite_diet/orchestration/manifests/first-waves.after-workload-profiles.json

python .agent/scratch/testsuite_diet/orchestration/wave.py \
  render .agent/scratch/testsuite_diet/orchestration/manifests/first-waves.after-workload-profiles.json \
  --run-id review-001

python .agent/scratch/testsuite_diet/orchestration/wave.py \
  run .agent/scratch/testsuite_diet/orchestration/manifests/first-waves.after-workload-profiles.json \
  --run-id implementation-001 --max-concurrency 4

python .agent/scratch/testsuite_diet/orchestration/wave.py status implementation-001
```

Artifacts live under `.local/testsuite-diet/runs/<run-id>/`: normalized
manifest, prompts, logs, structured final receipts, assigned-file deltas,
attested launcher manifests, completion schema, and aggregate `run.json`.
The launcher is resolved from
`$SINNIX_ROOT/dots/_ai/skills/agent-orchestration/scripts/run_agent_prompt.sh`
with `/realm/project/sinnix` as the fallback root.

`run --dry-run` renders the same artifacts and records commands without
starting a model. The unit test uses a fake attested launcher to prove exact
model/effort propagation, structured receipts, sequential waves, and the
four-worker cap without spending agent calls.

`completion.schema.json` is for shared implementation/deletion workers.
`certification.schema.json` is deliberately separate and is passed directly to
the attested launcher for a certification job. It binds the law and dossier to
the frozen base commit, proves Terra/high prompt/model identity, captures the
historical and mutation checks, records exact certified deletions and retained
obligations, and requires matching restored hashes plus an empty final git
status. Store these receipts under
`.local/testsuite-diet/certification/<wave>/<cluster>.json`; do not put
certification jobs into the implementation manifest.

## Worker and coordinator boundary

Workers edit only their assigned files, run only named focused selectors, and
never use git, Beads, broad formatters, generated-surface sweeps, or broad
verification. If an unassigned edit or unresolved design decision is needed,
the worker returns a structured blocker and must leave its assigned files
unchanged. A non-blocked implementation receipt with no assigned-file delta is
classified invalid, as is a receipt claiming an unassigned file.

The receipt names production dependencies, actual and proposed deletions, and
whether sensitivity was already executed or still requires coordinator
certification. Survivor jobs normally report `sensitivity.executed=false` and
do not delete. A deletion job must name the independent certification artifact
in its mission and receipt.

`run` requires a clean git checkout. After each wave it compares the complete
dirty-path state before and after the wave and rejects any changed path outside
that wave's assignment union. A same-wave job may not read a file another job
writes. If any job blocks or becomes invalid, all later waves are recorded as
skipped rather than speculatively continuing. Because the scratch plan is
gitignored and invisible to git status, the runner separately hashes
`.agent/scratch/testsuite_diet` and rejects a worker that changes its own
control inputs.

The runner never auto-rolls back an invalid wave. It marks later waves skipped
and leaves the checkout quarantined for Sol to compare against the recorded
wave baseline and per-job deltas. Automatic restoration in a shared checkout
could erase concurrent or operator work.

After a survivor wave stops, editing freezes. Sol or an independent Terra/high
certifier uses a disposable worktree at the frozen survivor revision to run the
historical witness, representative production mutation, and
deletion-obligation review. It restores a clean tree, emits an attested
certification receipt, and creates no commit or merge. Exact subtraction is a
later shared wave.
After the certified subtraction wave, Sol:

1. reads every attestation, receipt, file delta, and source diff;
2. reconciles cross-cluster contracts and repairs composition issues;
3. reruns combined focused selectors;
4. runs `devtools verify --quick`;
5. runs `devtools verify --all` for harness/dependency waves, or ordinary
   testmon-selected `devtools verify` for later leaf-only waves;
6. stages, commits, updates Beads, pushes, and publishes.

Switch to isolated worktrees when two jobs must write the same file, a job
needs temporary mutation or architecture/durability adjudication, or the
shared checkout cannot provide a clean coordinator-owned branch. Prefer
serialization when overlapping jobs share one hotspot and isolation would only
add merge work. Exact prompt boundaries reduce collisions; they are not
filesystem enforcement.

## Model routing

Terra/high is the default implementation and semantic-certification lane. Luna
is absent from the first coding wave: calibrating it for one tiny deletion
cannot save time. Luna/high is admitted only when several certified bounded
jobs can amortize the three-packet read-only calibration and five-job
probation. Sol is the coordinator and is rejected as a manifest worker.

Sol/Ultra is the coordinator mode: it adjudicates packets, delegates, steers,
waits, synthesizes receipts, and performs integration. Ultra's native
subagents and this runner are alternative fanout mechanisms for a wave. Do not
launch the same jobs through both. Prefer this runner when receipts must attest
the exact Terra/Luna model; use native Ultra delegation for read-only dossier
review or bounded work where exact worker-model selection is not required.

See [`../14-holistic-execution-audit.md`](../14-holistic-execution-audit.md)
for the controlling survivor→certify→subtract lifecycle and
[`../15-law-execution-dag-and-model-routing.md`](../15-law-execution-dag-and-model-routing.md)
for per-law model, hotspot, wave, and worktree routing.
