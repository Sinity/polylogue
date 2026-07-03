# Polylogue Devloop

This is the first stop for a contextless agent asked to continue the
Polylogue devloop. It names the executable process, the active state location,
and the durable memory surfaces.

## Start

From `/realm/project/polylogue`:

```bash
.agent/scripts/devloop-status
.agent/scripts/devloop-review
```

Use `.agent/scripts/devloop-status --focus` for a very fast context refresh
when only the active slice, focus, next action, git state, and packet size are
needed. Use `--quick` when host pressure is high and slower detailed
ops/worktree probes would add friction.

Then run `bd prime` (Beads workflow context) and read, in order:

1. `.agent/conductor-devloop/README.md`
2. `.agent/conductor-devloop/RUNBOOK.md`
3. `.agent/conductor-devloop/INTEGRATION.md`
4. `.agent/conductor-devloop/PROCESS.md`
5. `.agent/conductor-devloop/VELOCITY.md`
6. `.agent/conductor-devloop/TACTICS.md`
7. `.agent/conductor-devloop/INDEX.md`
8. `.agent/conductor-devloop/ACTIVE-LOOP.md` when present
9. `.agent/conductor-devloop/OPERATING-LOG.md` tail
10. `.agent/conductor-devloop/DEMO-RADAR.md`
11. `.agent/includes/README.md`

If `devloop-review` warns, fix the warning or record the conscious exception in
the active loop state before broad work.

## Post-Compaction Discipline

Conversation compaction is a lossy state snapshot, not a new operator
instruction. After compaction or resume:

- obey the newest real user message first;
- use the summary only to recover files, commits, proofs, jobs, and unfinished
  work;
- do not treat repeated themes in the summary as freshly re-requested;
- if the summary overweights process/meta work, spend at most one bounded Meta
  pass to repair the scaffold, then return to the object-level slice unless the
  newest user message says otherwise;
- update `ACTIVE-LOOP.md` when it disagrees with live state before committing
  or widening work.

## Shape

- `.agent/conductor-devloop/` is the active loop packet. Tracked files explain
  the process; ignored files hold current local state.
- `.agent/includes/` holds tracked durable project/devloop knowledge that should
  survive checkout and context loss.
- `.agent/scripts/` holds tracked executable primitives. Do not copy these into
  the conductor packet.
- `.agent/demos/` is ignored and current-curated. It is the best current demo
  set, not an append-only archive.
- `.agent/scratch/` is ignored supporting research only. It is not active loop
  state.

## Process

Use the shared focus modes exactly:

```text
Direction, Evidence, Construction, Proof, Artifact, Velocity, Meta
```

Record material transitions with:

```bash
.agent/scripts/devloop-focus <from> <to> "<trigger>" "<decision>"
```

`devloop-focus` validates transitions against the shared mode graph and checks
that `<from>` matches the active loop's current target mode. Use `--force` only
for a rare edge or deliberate continuity break whose rationale is explicit in
the trigger and decision; otherwise switch through a normal intermediate mode.

Start one concrete slice with:

```bash
.agent/scripts/devloop-start "<slice title>"
```

For process/self-improvement work, start in Meta and stay there until a
deliberate transition:

```bash
.agent/scripts/devloop-start --meta "<slice title>"
.agent/scripts/devloop-start --focus Meta "<slice title>"
```

`devloop-start` also infers `Meta` for obvious process titles beginning with
`meta ...`, `process ...`, `scaffold ...`, `devloop ...`, or
`conductor ...`, and for titles containing `convention`; use `--meta` or
`--focus Meta` for other process slices.

Refresh generated local state with:

```bash
.agent/scripts/devloop-sync
```

`devloop-sync` also keeps the active conductor packet compact: if
`OPERATING-LOG.md` grows past the rolling-window budget, older entries move to
ignored `.agent/archive/conductor-history/` and `EVENTS.jsonl` is regenerated
from the active window.
It also refreshes `devloop-script-hashes.tsv` for every `devloop-*` primitive
and the shared `lib-devloop` helper; `devloop-review` treats that manifest as a
positive freshness check for scaffold drift.

Every `.agent/scripts/devloop-*` primitive must support a side-effect-free
`--help` path. Use that for discovery instead of trying scripts with placeholder
arguments; `devloop-review` checks that help probing does not mutate active
state files.

## Task State: Beads

Beads (`bd`) is the durable backlog, dependency graph, and directive channel
for this devloop. `ACTIVE-LOOP.md`, `OPERATING-LOG.md`, and `DEMO-RADAR.md`
are the narrative/current-slice state; the work items themselves live in
beads.

Core moves:

- `bd prime` — recover workflow context after compaction/resume.
- `bd ready --json` — candidate slices, blockers already filtered out.
- `bd show <id>` — full item: description, design notes, acceptance,
  dependencies. Design fields carry pre-made implementation judgment; read
  them before re-deriving.
- `bd update <id> --claim` when a slice starts; `bd close <id>
  --reason "..."` with the proof when it lands.
- Discovered follow-up work: `bd create` with
  `--deps discovered-from:<current-id>`.
- `bd remember "<insight>"` / `bd memories <kw>` — durable repo-operational
  lore that must survive context loss.
- `bd human <id>` — flag a decision only the operator can make.

Conventions:

- Priorities encode the operator tier frame: P0 = proof campaigns,
  P1 = campaign enablers + live-trust correctness, P2 = correctness/features,
  P3 = surface hygiene, P4 = parked/blocked.
- Campaign epics are protected: closing a slice bead never closes the
  campaign epic; the epic closes only when its terminal state
  (cold-reader-gated external artifact) is recorded.
- GitHub-mirrored beads carry `external-ref gh-NNNN`. Closing the bead does
  not close the GitHub issue — that is a separate explicit act under the
  resolver-keyword discipline. When a mirrored bead closes, leave a note on
  whether the GH issue is also satisfied.
- The bead graph is state, not narrative. Operating-log entries still record
  triggers/decisions/proofs; beads record what work exists, what blocks it,
  and what it produced.

## Current Goal

Conduct the Polylogue dogfood/demo devloop indefinitely: continuously choose the
highest-value live-archive capability slice, produce inspectable artifacts
proving Polylogue improves agents with real history, collapse silos into shared
acquisition/query/projection/rendering substrate, verify on the active archive
or live capture, maintain logs and handoffs, and reprioritize by evidence.

## Defaults

- Default archive root: `/home/sinity/.local/share/polylogue`.
- Production `polylogued.service` should stay inactive during this devloop.
- The intended daemon is the branch-local `devtools workspace dev-loop`
  launcher on the canonical archive. Start/restart it with an explicit archive
  root and ports, for example:

  ```bash
  devtools workspace dev-loop \
    --archive-root /home/sinity/.local/share/polylogue \
    --api-port 8766 \
    --browser-capture-port 8765 \
    --prepare \
    --launch-daemon \
    --json
  ```

  `devloop-status` prints the daemon run directory and spool path; `devloop-review`
  warns if the daemon still points at a stale commit's run directory.
- Always state archive root, schema version, and relevant counts when quoting
  live archive facts.
- Treat `devloop-status` convergence fields as claim guards. A matching
  `index.schema_version` means the index can be opened; it does not mean the
  archive is fully converged. `convergence.raw_materialization_join_gaps`
  counts raw rows that do not join to index rows by `raw_id`; those gaps may be
  classified aliases or non-session artifacts. `convergence.raw_materialization_debt`
  counts unresolved actionable/open/blocked materialization work. Do not claim
  full archive convergence while materialization debt is nonzero, and quote
  classified join gaps separately when they remain.
- Treat `devloop-status` git fields as part of the start gate: branch, HEAD,
  tracked-change count, and untracked-change count tell you whether you are
  resuming a clean branch, local process edit, or product slice.
- Treat `devloop-status` pressure fields as proof routing, not trivia. If
  `live_performance_proof_blocked` is true, do not claim live latency or
  throughput from broad archive probes; use source review, focused tests, query
  plans, or wait until borg/materialization pressure clears.
- Treat `devloop-review` ignore-policy checks as load-bearing: tracked scaffold
  must survive checkout, while active loop state and demos stay local/current.
- `/realm/inbox` is staging only. The devloop must not depend on it.

## Integration Lane

This devloop runs on a long-lived workbench branch. That is acceptable only if
integration stays visible and PR-shaped.

Use:

```bash
.agent/scripts/devloop-integration
.agent/scripts/devloop-integration --subagent-prompt
```

The default report shows the workbench branch state and any local PR-shaped
replay worktrees under the integration root. Treat that as the source of truth
for integration status before deciding to push/open PRs, continue replaying
groups, or ask another subagent for clustering.

The intended flow is:

1. cluster ahead commits by related product/change intent;
2. create candidate PR branches from current `origin/master`;
3. cherry-pick or replay each cluster, squash within that coherent cluster, and
   write strong PR titles/bodies;
4. prove the PR branches compose back to the workbench state, or preserve the
   exact residual diff/dependency.

The clustering/planning part is a good read-heavy subagent lane. Final replay,
verification, push, and PR creation are normal authorized devloop actions; PRs
open ready-for-review by default unless a concrete draft reason is recorded.
The main devloop owns publication unless the operator explicitly delegates that
write authority to another agent.
