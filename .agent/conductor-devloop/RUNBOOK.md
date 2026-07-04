# Polylogue Conductor Runbook

## Start Gate

Before a capability slice:

1. Run `.agent/scripts/devloop-status`.
2. Run `.agent/scripts/devloop-review`.
3. Read `README.md` and this `RUNBOOK.md`.
4. If ignored local `ACTIVE-LOOP.md` exists, read it.
5. Run `bd prime` if not already done this session, then `bd ready` — beads
   is the durable backlog; the status output includes a beads summary.
6. If review warns, fix it or record the accepted warning in `ACTIVE-LOOP.md`
   when local state exists; otherwise initialize local state first.
7. Start the slice with `.agent/scripts/devloop-start "<slice>"`, or
   `.agent/scripts/devloop-start --meta "<slice>"` when the slice itself
   begins as process/self-improvement work. Claim the matching bead with
   `bd update <id> --claim`; create one first if the slice is untracked.

Use `.agent/scripts/devloop-status --json` when another script or report needs
structured state instead of human-readable text.

Every `.agent/scripts/devloop-*` primitive supports `--help` as a read-only
discovery path. Do not probe a script with placeholder positional arguments just
to learn its interface; a prior `--help` sweep accidentally rewrote active loop
state, so `devloop-review` now executes all help paths and verifies the active
state hashes are unchanged.

Do not start broad import/test/runtime work while daemon/root/process warnings
are unexplained.

`ACTIVE-LOOP.md` is current-slice state, not a history file. Keep its accepted
warnings section short and limited to live exceptions from the current slice.
Completed proofs, old commits, and prior demo notes belong in
`OPERATING-LOG.md` or `DEMO-RADAR.md`; `devloop-start` resets the warning field
when a new slice begins, and `devloop-review` warns if it accretes into a
historical ledger.

## Post-Compaction Discipline

Conversation compaction is a lossy state snapshot, not a new operator
instruction. After compaction:

- obey the newest real user message first;
- use the summary to recover state, proofs, files, jobs, commits, and
  unfinished work, not to invent a fresh priority;
- do not infer that repeated themes in the summary were re-requested at resume
  time;
- if the summary overweights process/meta/velocity work, spend at most one
  bounded Meta pass to correct the scaffold, then return to the object-level
  slice unless the newest user message says otherwise;
- update `ACTIVE-LOOP.md` if it disagrees with the actual current slice before
  committing or widening work.

## Focus Modes

- `Direction`: choose the slice.
- `Evidence`: inspect source, archive, daemon, logs, issues, and demos.
- `Construction`: edit shared substrate or artifacts.
- `Proof`: run the narrow check that proves the claim.
- `Artifact`: make the result inspectable outside chat.
- `Velocity`: remove friction or record why it remains.
- `Meta`: audit agent/process failure modes and convert useful corrections into
  executable scaffold, observability, or positive state checks.

Every material switch should name the trigger and decision.

Use the executable switch helper:

```bash
.agent/scripts/devloop-focus <from> <to> "<trigger>" "<decision>"
```

This appends the transition to ignored `OPERATING-LOG.md`, updates ignored
`ACTIVE-LOOP.md` when present, and refreshes the conductor packet. The valid
focus modes are exactly Direction, Evidence, Construction, Proof, Artifact,
Velocity, and Meta. The helper also validates the transition edge against the
shared mode graph used by review and velocity audits. It also checks continuity:
the `<from>` mode must match the active loop's current target mode, so stale
manual transitions fail before they rewrite local state. Use `--force` only when
a rare edge or continuity break is genuinely correct and the trigger/decision
explains why it should not be routed through the normal state machine.

## Greedy Batch / PR Cadence

Default development unit: one complete bead. Finish the capability claim total
and publish that as the normal PR boundary. A coherent phase is an exception
that must justify itself before publication: the bead must be genuinely too
large or risky to close as one PR, and the phase must honestly satisfy a named
acceptance-criteria subset with a clear residual matrix. Do not open or publish
a PR for every small projection, helper, construct declaration, or proof artifact
merely because it is locally green.

Prefer a single branch/PR for the whole bead when the work:

- belongs to one bead and one capability claim;
- touches the same shared acquisition/query/projection/rendering substrate;
- can be verified by one focused test family plus one live/demo artifact;
- would otherwise force reviewers and future agents to reconstruct intent
  across several thin PRs.

Split only when there is a real boundary:

- the bead is too large to review safely as one phase;
- independent parts have different risk, owners, or deployment timing;
- one part is a prerequisite unblocking other active work;
- verification cost or failure isolation would become materially worse;
- a partial PR can close a named bead or named acceptance-criteria phase, not
  just land a convenient substep.

Before publishing, audit the bead acceptance criteria. The default answer should
be "this PR closes the bead." If the PR does not close it, the body and bead
notes must say exactly why a phase split is justified, which criteria are
satisfied, which are deferred, and which follow-up bead owns the remainder. This
is a velocity rule: fewer, more complete integration boundaries beat a chain of
locally-correct but strategically thin slices.

If the remaining work is within the same bead, same substrate, and same proof
family, keep working. Do not stop merely because the current diff is already
mergeable.

## One-Loop Protocol

1. **Direction**
   - Select one capability slice.
   - Brainstorm candidate demos or demo improvements before narrowing.
   - Check the workload radar: `bd ready --json` (priorities encode the
     operator tier frame — P0 campaign epics outrank everything),
     `ACTIVE-LOOP.md`, `DEMO-RADAR.md`, recent `OPERATING-LOG.md` next
     decisions, and any relevant audit note in `INDEX.md`.
   - Run `devtools workspace frontier` when choosing from more than one
     candidate, when a sibling agent is active, or while waiting on a long proof.
     Use its subsystem/proof-cost/runtime-risk/subagent columns to batch
     adjacent work, avoid schema-lane collisions, and select read-only audit
     lanes that can update Beads without touching the current checkout.
   - Read `bd show <id>` for the chosen item — description, design notes,
     and acceptance criteria carry pre-made judgment; do not re-derive it.
   - Rank candidates by evidence urgency, user-visible truthfulness, substrate
     leverage, demo value, and velocity impact. Prefer the slice that improves
     the most categories without broadening into vague cleanup.
   - State out-of-scope items.
   - Write the slice contract before editing: demo value, reusable substrate,
     proof ladder, non-goals, and first concrete action.
   - Start the slice with `.agent/scripts/devloop-start "<slice>"`. Use
     `--meta` for process-improvement slices so the first logged
     transition is truthful instead of a synthetic Direction entry.
     The helper also infers `Meta` for obvious process titles beginning with
     `meta ...`, `process ...`, `scaffold ...`, `devloop ...`, or
     `conductor ...`, and for titles containing `convention`; use `--meta` or
     `--focus Meta` for other process slices.
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
   - Every loop must leave a Velocity/Meta record: a no-op with reason, a
     frontier batch grouping, a delegated read-only lane, a removed friction
     point, or a linked follow-up Bead.

7. **Meta**
   - Run when the operator corrects process behavior, repeated friction appears,
     or a loop feels vague/stalled.
   - Start pure process/self-improvement slices with
     `.agent/scripts/devloop-start --meta "<slice>"`; this records
     `Meta -> Meta` until a deliberate transition moves the loop back to
     Evidence, Construction, Proof, Artifact, Velocity, or Direction.
   - Run `.agent/scripts/devloop-velocity --record` near the start of a
     Meta-focused slice. This records focus cadence, long gaps, packet growth,
     task-history friction, and active heavy work under `.agent/task-history/`
     and leaves a compact `velocity-audit` entry in the operating log.
   - Use `.agent/scripts/devloop-meta "<trigger>" "<failure-hypothesis>" "<evidence>" "<change-considered>" "<change-made>" "<change-deferred>" "<next-safeguard>"`.
   - Fill every field; use `none` only when a field is genuinely empty.
   - Prefer one concrete scaffold/tooling/observability change over broad
     apology prose.
   - Ask whether demo generation lagged; if yes, run `devloop-demo` and improve
     the default next-loop check.

8. **Integration**
   - Before ending a slice, make the handoff explicit: committed files or
     intentionally uncommitted local state, Beads updates, PR/replay status, and
     verification. Use `.agent/scripts/devloop-integration` when branch replay
     or PR grouping matters.
   - Integration does not replace Velocity/Meta closure; after the handoff,
     record the mandatory speed/process outcome described above.

## Workload Radar

The devloop backlog is evidence-shaped, not a static ticket queue. Maintain it
as a radar during Direction and at substantial checkpoints:

1. Beads (`bd ready`, `bd blocked`, `bd stats`) is the durable backlog,
   dependency graph, and operator directive channel. Keep it truthful:
   claim on start, close with reason + proof on completion, create linked
   beads (`discovered-from:<id>`) for discovered work, and re-priority
   items when evidence changes their tier.
2. `ACTIVE-LOOP.md` says what is live now.
3. `DEMO-RADAR.md` says which demos are current, missing, stale, or next.
4. `OPERATING-LOG.md` carries recent "next decision" lines and proof caveats.
5. `.agent/includes/` carries durable conventions and architecture direction.
6. `.agent/archive/conductor-history/` carries older audit/debt notes for
   archaeology only.
7. `.agent/scripts/devloop-integration` shows how far the long-running branch is
   ahead of master and emits the read-heavy subagent prompt for PR clustering.

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
- **Integration pressure:** Are there enough ahead commits that the workbench
  should be clustered into PR-shaped replay groups before more unrelated work
  lands?

If the current active slice is vague, do a Meta transition and repair the radar:
write a sharper `ACTIVE-LOOP.md` next action, add a `DEMO-RADAR.md` entry, or
turn a recurring friction into an executable review/velocity check.

## Integration Lane

The long-running devloop branch is an integration workbench, not the final
history shape. Keep integration planning active enough that work can move to
master en masse without archaeology.

Use:

```bash
.agent/scripts/devloop-integration
.agent/scripts/devloop-integration --subagent-prompt
```

The default report must show both sides of the lane:

- workbench state: merge-base, current head, commits ahead of `origin/master`;
- local replay state: PR-shaped worktrees under the integration root, their
  branches, heads, dirty-path counts, and the current replay-plan artifact when
  present.

The branch integration target is PR-shaped:

1. Start candidate PR branches from current `origin/master`.
2. Group related workbench commits by product/change intent.
3. Replay each group in order, squashing only within that coherent group.
4. Write durable PR titles and bodies with Summary, Problem, Solution, and
   Verification.
5. Prove the PR branches compose back to the workbench state, or state the exact
   residual diff/dependency.

This lane is suitable for a read-heavy subagent: it can cluster commits, draft
PR bodies, identify dependencies, flag overbroad claims, and write a dry-run
replay script. The main devloop owns branch creation, conflict resolution,
verification, pushing, and PR creation.

Run the integration lane during Direction or Velocity when commits ahead of
master are accumulating across unrelated slices, and during wait windows when
the pending proof does not need the same checkout or archive.

Before pushing, opening PRs, continuing to the next replay group, or deciding a
local PR branch is ready, run `.agent/scripts/devloop-integration` and record
the exact branch heads plus verification in `OPERATING-LOG.md` or the replay
plan. Do not rely on chat memory for integration state.

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

Waiting is active loop time. Immediately pick one foreground lane from
`.agent/scripts/devloop-ahead`: adjacent source audit, artifact/demo, backlog
radar, subagent/audit prompt, integration planning, next verification, or
velocity/meta. Rotate focus deliberately: Proof -> Velocity for resource/tool
friction, Proof -> Artifact for artifact writing, Proof -> Evidence for
adjacent source/runtime inspection, or Proof -> Direction when the pending
result is likely to close the slice.

Ahead work must not create hidden contention. Do not start another heavy
archive/test/build job against the same checkout/archive while a proof is
running. Prefer light reads, demo writing, prompt/backlog shaping, source review,
or a non-overlapping subagent audit with explicit owned files and expected
artifact.

Subagents are useful for backlog growth and independent audits, not for
unbounded parallel churn. Use them when the work is read-heavy, separable, and
can return a concrete artifact: schema construct-validity audit, CLI surface
audit, route/contract drift audit, provider-field harmonization audit, or demo
candidate ranking. Do not spawn one just because a command is pending.

## Runtime Baselines

Capture lightweight baselines before and after non-trivial runtime, archive, or
resource-sensitive work:

```bash
.agent/scripts/devloop-baseline "short-label"
```

Baselines live under `.agent/task-history/live-baselines/` and are local
process/runtime evidence for comparison, not product demos and not scratch
research. They should state archive root, schema, daemon state,
memory/pressure, and loop-affecting processes.

For a faster speed read without creating a baseline artifact:

```bash
.agent/scripts/devloop-velocity
```

Use it to spot long logged gaps, stale focus, archive-root drift, and active
heavy processes before starting another archive-heavy command. It also prints
active packet size and event/log sidecar sizes so conductor growth is visible
instead of becoming hidden ignored-state clutter.

The focus transition audit in `devloop-velocity` is a process-health check. Treat repeated
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
- Push and PR creation are authorized devloop actions for integration/publication
  slices. Open PRs ready-for-review by default; use draft only with an explicit
  recorded reason. Do not push directly to `master`.

## End Gate

Before ending:

1. The latest operating-log entry is filled.
2. `ACTIVE-LOOP.md` names focus, accepted warnings, and next action.
3. Beads reflect reality: the slice bead is closed with reason + proof (or
   updated with `--notes` on partial progress), discovered work exists as
   linked beads, and nothing is left `in_progress` that this session is not
   actually progressing.
   After any Beads write, run `bd export --output .beads/issues.jsonl` before
   staging the tracked Beads snapshot.
4. `.agent/conductor-devloop` ignored generated state has been refreshed with
   `devloop-sync`.
5. `devloop-review` warnings are fixed or explicitly accepted.
6. `DEMO-RADAR.md` is current for substantial demo-facing work, or the log says
   why no demo artifact was implicated.
7. Any needed background process is complete, stopped, or named with a poll path.

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
