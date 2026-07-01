# Polylogue Conductor Runbook

## Start Gate

Before a capability slice:

1. Run `.agent/scripts/devloop-status`.
2. Run `.agent/scripts/devloop-review`.
3. Read `README.md` and this `RUNBOOK.md`.
4. If ignored local `ACTIVE-LOOP.md` exists, read it.
5. If review warns, fix it or record the accepted warning in `ACTIVE-LOOP.md`
   when local state exists; otherwise initialize local state first.
6. Start the slice with `.agent/scripts/devloop-start "<slice>"`, or
   `.agent/scripts/devloop-start --focus Meta "<slice>"` when the slice itself
   begins as process/self-improvement work.

Use `.agent/scripts/devloop-status --json` when another script or report needs
structured state instead of human-readable text.

Do not start broad import/test/runtime work while daemon/root/process warnings
are unexplained.

`ACTIVE-LOOP.md` is current-slice state, not a history file. Keep its accepted
warnings section short and limited to live exceptions from the current slice.
Completed proofs, old commits, and prior demo notes belong in
`OPERATING-LOG.md` or `DEMO-RADAR.md`; `devloop-start` resets the warning field
when a new slice begins, and `devloop-review` warns if it accretes into a
historical ledger.

## Focus Modes

- `Direction`: choose the slice.
- `Evidence`: inspect source, archive, daemon, logs, issues, and demos.
- `Construction`: edit shared substrate or artifacts.
- `Proof`: run the narrow check that proves the claim.
- `Artifact`: make the result inspectable outside chat.
- `Velocity`: remove friction or record why it remains.
- `Meta`: audit agent/process failure modes and convert useful corrections into
  executable scaffold, observability, or tripwires.

Every material switch should name the trigger and decision.

Use the executable switch helper:

```bash
.agent/scripts/devloop-focus <from> <to> "<trigger>" "<decision>"
```

This appends the transition to ignored `OPERATING-LOG.md`, updates ignored
`ACTIVE-LOOP.md` when present, and refreshes the conductor packet. The valid
focus modes are exactly Direction, Evidence, Construction, Proof, Artifact,
Velocity, and Meta.

## One-Loop Protocol

1. **Direction**
   - Select one capability slice.
   - Brainstorm candidate demos or demo improvements before narrowing.
   - Check the workload radar: `ACTIVE-LOOP.md`, `DEMO-RADAR.md`, recent
     `OPERATING-LOG.md` next decisions, and any relevant audit note in
     `INDEX.md`.
   - Rank candidates by evidence urgency, user-visible truthfulness, substrate
     leverage, demo value, and velocity impact. Prefer the slice that improves
     the most categories without broadening into vague cleanup.
   - State out-of-scope items.
   - Write the slice contract before editing: demo value, reusable substrate,
     proof ladder, non-goals, and first concrete action.
   - Start the slice with `.agent/scripts/devloop-start "<slice>"`. Use
     `--focus Meta` for process-improvement slices so the first logged
     transition is truthful instead of a synthetic Direction entry.
   - Run `.agent/scripts/devloop-demo` when the slice should create, refresh,
     retire, or caveat an artifact.

2. **Evidence**
   - Gather authoritative current state from code, archive, daemon, history,
     docs, and demo artifacts.
   - Separate stored/source evidence from derived projections, candidates,
     assertions, and narrative judgment.
   - Identify coverage gaps that would make a demo misleading.

3. **Construction**
   - Make the smallest coherent change that advances the slice.
   - Prefer shared acquisition/query/projection/rendering substrate.
   - Avoid one-off recovery/export/demo/report silos.

4. **Proof**
   - Use the proof ladder: source review -> unit/parser -> package -> CLI/live
     archive -> real-data artifact -> broad gate.
   - State what the proof supports and what it does not.
   - Default exit is `Proof -> Artifact` when there is an artifact/demo/log to
     update, or `Proof -> Velocity` when the main result is process/resource
     learning. If exiting straight to `Direction`, record why no artifact or
     velocity closure was implicated.

5. **Artifact**
   - Update the repo-local `.agent/demos` shelf or conductor packet with
     something inspectable.
   - Treat `.agent/demos` as the best current demo set, not an append-only
     archive. Replace, consolidate, or move stale demos to `.agent/archive`
     whenever a better demo supersedes them.
   - Include rerun commands, data source/root/schema, proof/caveat, and limits.
   - Use `.agent/scripts/devloop-demo` to record candidates considered,
     selected/improved artifact, and next demo question.

6. **Velocity**
   - Record speedup, drag, next acceleration, and risk.
   - Run `.agent/scripts/devloop-velocity` when the loop feels slow, a
     maintenance/diagnostic command stalls, or before choosing a process
     improvement. Prefer making a measured drag visible over adding broad
     process prose.
   - Run `.agent/scripts/devloop-sync`.
   - Run `.agent/scripts/devloop-review` before claiming a clean checkpoint.

7. **Meta**
   - Run when the operator corrects process behavior, repeated friction appears,
     or a loop feels vague/stalled.
   - Start pure process/self-improvement slices with
     `.agent/scripts/devloop-start --focus Meta "<slice>"`; this records
     `Meta -> Meta` until a deliberate transition moves the loop back to
     Evidence, Construction, Proof, Artifact, Velocity, or Direction.
   - Use `.agent/scripts/devloop-meta "<trigger>" "<failure-hypothesis>" "<evidence>" "<change-considered>" "<change-made>" "<change-deferred>" "<next-tripwire>"`.
   - Fill every field; use `none` only when a field is genuinely empty.
   - Prefer one concrete scaffold/tooling/observability change over broad
     apology prose.
   - Ask whether demo generation lagged; if yes, run `devloop-demo` and improve
     the tripwire.

## Workload Radar

The devloop backlog is evidence-shaped, not a static ticket queue. Maintain it
as a radar during Direction and at substantial checkpoints:

1. `ACTIVE-LOOP.md` says what is live now.
2. `DEMO-RADAR.md` says which demos are current, missing, stale, or next.
3. `OPERATING-LOG.md` carries recent "next decision" lines and proof caveats.
4. `.agent/includes/` carries durable conventions and architecture direction.
5. `.agent/archive/conductor-history/` carries older audit/debt notes for
   archaeology only.

Use this scoring order when choosing between candidates:

- **Truthfulness risk:** Will leaving this alone cause the CLI, docs, demos, or
  reports to lie about source evidence, archive counts, fields, or semantics?
- **Substrate leverage:** Does the slice collapse a recovery/export/demo/report
  silo into query, acquisition, projection, rendering, or insight substrate?
- **Demo value:** Will the slice produce or improve an artifact that a future
  operator or agent can inspect without chat context?
- **Velocity impact:** Does the slice remove repeated latency, root confusion,
  daemon confusion, test friction, or scaffold drift?
- **Scope discipline:** Can the slice be proven with a narrow command and a
  live artifact without turning into an unbounded refactor?

If the current active slice is vague, do a Meta transition and repair the radar:
write a sharper `ACTIVE-LOOP.md` next action, add a `DEMO-RADAR.md` entry, or
turn a recurring friction into an executable review/velocity check.

## Heavy Work

Before a daemon/import/test run, state the proof claim and expected duration.
While it runs, use `.agent/scripts/devloop-ahead`; do not stack conflicting
heavy jobs. If a command is slow, first check archive root, duplicate daemons,
schema, stage timings, and `devloop-status` pressure output.

If `devloop-status` reports `live_performance_proof_blocked`, do not claim
latency or throughput from broad archive probes in that window. Use source
review, focused tests, query plans, or wait until borg/materialization/D-state
pressure clears.

If a command may run longer than a minute, record an active wait state:

```bash
.agent/scripts/devloop-wait "<job-or-command>" "<proof-claim>" "<poll-in>" "<mode-task>"
```

Waiting is active loop time. Rotate focus deliberately: Proof -> Velocity for
resource/tool friction, Proof -> Artifact for artifact writing, Proof ->
Evidence for adjacent source/runtime inspection, or Proof -> Direction when the
pending result is likely to close the slice.

## Runtime Baselines

Capture lightweight baselines before and after non-trivial runtime, archive, or
resource-sensitive work:

```bash
.agent/scripts/devloop-baseline "short-label"
```

Baselines live under `.agent/scratch/live-baselines/` and are local evidence for
comparison, not product demos. They should state archive root, schema, daemon
state, memory/pressure, and loop-affecting processes.

For a faster speed read without creating a baseline artifact:

```bash
.agent/scripts/devloop-velocity
```

Use it to spot long logged gaps, stale focus, archive-root drift, and active
heavy processes before starting another archive-heavy command.

The focus transition audit in `devloop-velocity` is a tripwire. Treat repeated
`Proof -> Direction` exits, long unlabeled dwell, or stale ACTIVE-LOOP focus as
process debt; fix the scaffold or log the deliberate exception before broadening
work.

## External Inbox Routing

`/realm/inbox` is an external staging area, not a Polylogue default. The
devloop must remain functional if that tree is wiped. Default conductor state
lives under `.agent/conductor-devloop`; default demo artifacts live under
`.agent/demos`.

When explicitly asked to read older briefs, prompt exports, patch packs, raw
devloop exports, or reports from the inbox, consult the current routing shelves
first:

- `/realm/inbox/project-devloops/README.md`
- `/realm/inbox/project-artifacts/README.md`

Do not mirror Polylogue conductor state or demo shelves into `/realm/inbox`
unless an explicit task names that destination.

## Git / Branch Protocol

- Treat this as a long-lived development branch unless the operator asks for a
  new branch or a PR-shaped split.
- Commit logical chunks proactively after focused proof, using staged paths.
  Avoid broad staging sweeps.
- Prefer continuing in this checkout over creating worktrees. Use worktrees only
  when true isolation is needed for concurrent risky edits, a separate agent
  lane, or an experiment that must not perturb the active branch.
- Use compile/test/daemon wait time for ahead work in this checkout. If a proof
  fails after ahead work, diagnose the whole failure shape and batch the fix.
- Do not push unless explicitly asked.

## End Gate

Before ending:

1. The latest operating-log entry is filled.
2. `ACTIVE-LOOP.md` names focus, accepted warnings, and next action.
3. `.agent/conductor-devloop` ignored generated state has been refreshed with
   `devloop-sync`.
4. `devloop-review` warnings are fixed or explicitly accepted.
5. `DEMO-RADAR.md` is current for substantial demo-facing work, or the log says
   why no demo artifact was implicated.
6. Any needed background process is complete, stopped, or named with a poll path.

## Durability

The reusable scaffold under `.agent/` is tracked. Current devloop state under
`.agent/conductor-devloop/` is ignored unless explicitly unignored in
`.gitignore`; this includes `ACTIVE-LOOP.md`, logs, generated manifests, and
event sidecars. Treat tracked docs plus ignored local state together as the
source of truth for contextless resumption; publish or copy the local packet
elsewhere only on an explicit task.

`devloop-review` enforces this boundary: scaffold paths such as
`.agent/DEVLOOP.md`, `.agent/includes/`, `.agent/scripts/`, and tracked
conductor protocol docs must be force-included by `.gitignore`, while current
state and demos must remain ignored. `devloop-status` reports branch, HEAD,
tracked-change count, and untracked-change count so a fresh agent sees whether
it is resuming a clean branch, a local scaffold edit, or a product slice.
