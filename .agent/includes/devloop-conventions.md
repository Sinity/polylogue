# Devloop Conventions

Polylogue uses the shared conductor conventions agreed with the Sinex devloop.
The goal is not identical repositories; it is transferable operating behavior.

The shared target combines Polylogue's active-state model with Sinex's durable
knowledge model:

- Polylogue's `conductor-devloop/` model is the source of truth for active
  state.
- Sinex's explicit cold-start and durable include model is the source of truth
  for project knowledge.
- Both projects should expose the same primitive names so an agent can transfer
  habits without relearning scripts or state semantics.

## Canonical Layout

```text
.agent/
  README.md
  DEVLOOP.md
  conductor-devloop/
  includes/
  scripts/
  demos/
  scratch/
  tools/
```

`conductor-devloop/` is the canonical active loop root. Do not maintain a
second active state mirror under `scratch/current` or `handoff/`.

## State File Roles

Required active packet files:

- `README.md` — contextless resume entrypoint.
- `RUNBOOK.md` — loop protocol, focus modes, proof ladder, heavy-work rules,
  git protocol, and end gate.
- `ACTIVE-LOOP.md` — ignored current slice/focus/next action.
- `OPERATING-LOG.md` — ignored timestamped decisions, evidence, actions, proof,
  and next decisions.
- `DEMO-RADAR.md` — ignored demo candidates, selected artifact, proof/caveat,
  and next demo question.
- `PROCESS.md`, `TACTICS.md`, `VELOCITY.md` — compact process, tactical, and
  speed rules.
- `ADVERSARIAL-REVIEW.md` — known failure modes and checks.
- `INDEX.md` — startup order and packet routing.

Recommended generated/local packet files:

- `EVENTS.jsonl` — generated from the active operating-log window.
- `SELF-PROMPTS.md` — durable self-prompt and process-goal material.
- `MANIFEST.md` — generated packet inventory.
- `devloop-script-hashes.tsv` — generated script freshness manifest.

Do not copy scripts, README snapshots, demos, or scratch material into the
packet for portability. A packaging tool can bundle those when needed; the
active packet should stay small and canonical.

## Script Names

Every devloop repo should converge on these primitive names:

```text
devloop-status
devloop-review
devloop-start
devloop-checkpoint
devloop-log
devloop-focus
devloop-demo
devloop-baseline
devloop-wait
devloop-ahead
devloop-meta
devloop-handoff
devloop-sync
devloop-velocity
devloop-refresh-demos
devloop-refresh-events
```

The exact internals may be repo-specific, but the operator-facing contract
should remain stable across projects.

Shared script contracts:

| Script | Contract |
| --- | --- |
| `devloop-status` | Fast live status: git, active loop, runtime/archive health, packet size, active heavy work. Provide `--focus` and `--quick` when useful. |
| `devloop-review` | Adversarial scaffold/state audit; fail on hard drift, warn on soft clutter. |
| `devloop-start` | Start one named slice and write active/log state. |
| `devloop-checkpoint` | Record reassessment without changing slice identity. |
| `devloop-log` | Append a timestamped structured operating-log entry. |
| `devloop-focus` | Record focus transition with trigger and decision. |
| `devloop-demo` | Update demo radar and, when applicable, the log. |
| `devloop-baseline` | Capture lightweight runtime/resource/git baseline. |
| `devloop-wait` | Record wait state plus foreground work during long commands. |
| `devloop-ahead` | Print useful foreground work suggestions. |
| `devloop-meta` | Record process failure/self-improvement audit. |
| `devloop-handoff` | Write latest human/agent handoff note inside the conductor packet. |
| `devloop-sync` | Regenerate derived packet files only; do not mirror to `/realm/inbox`. |
| `devloop-velocity` | Summarize cadence, proof gaps, friction, and slow commands. |
| `devloop-refresh-demos` | Regenerate demo manifests/catalog/index. |
| `devloop-refresh-events` | Regenerate `EVENTS.jsonl` from `OPERATING-LOG.md`. |

Every `devloop-*` script must support a side-effect-free `--help` path.

The focus-mode graph has one executable source of truth:
`.agent/scripts/lib-devloop`. Scripts that validate or report transitions must
consume `devloop_focus_modes`, `devloop_focus_edges`, or
`devloop_focus_edge_allowed`; do not duplicate the edge set in Python or shell
snippets inside individual scripts.

## Task State: Beads

Beads (`bd`) is the durable task/backlog/dependency substrate for repos that
carry a `.beads/` workspace (Polylogue since 2026-07-03). Contract:

- Work items, blockers, dependencies, priorities, and operator directives
  live in beads. A deferred directive is a P0/P1 bead, or a `bd human` flag
  when it needs an operator decision.
- The narrative plane stays in the conductor packet: `ACTIVE-LOOP.md`,
  `OPERATING-LOG.md`, and `DEMO-RADAR.md` record the current slice, focus
  transitions, proofs, and demo state. Beads records what work exists and
  what blocks it; the packet records how the work went.
- Loop integration: `bd ready` during Direction, `bd update <id> --claim` at
  slice start, `bd close <id> --reason` with proof at slice end,
  `bd create --deps discovered-from:<id>` for discovered work,
  `bd remember` for durable repo-operational lore.
- Priorities encode the operator tier frame (P0 proof campaigns, P1
  enablers/trust-correctness, P2 correctness/features, P3 hygiene,
  P4 parked). Campaign epics close only on recorded terminal state, never
  by slice completion.
- Scripts treat `bd` as optional (`command -v bd` guard) so the loop
  degrades gracefully where beads is absent.
- Sinex parity is pending: Sinex still uses `devloop-checkpoint --queue` as
  its directive channel. Until Sinex adopts beads, the shared contract
  differs deliberately; do not remove the Sinex queue primitive from this
  spec without changing both repos (tracked as a Polylogue bead).

## Greedy Batch / PR Cadence

Default development unit: one complete bead, or one coherent phase that can
honestly close a named bead acceptance-criteria subset with a clear residual
matrix. Do not open or publish a PR for every small projection, helper,
construct declaration, renderer field, or proof artifact merely because that
substep is locally green.

Prefer a single branch/PR phase when the work:

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

Before publishing, audit the bead acceptance criteria. If the PR does not close
the bead, the PR body and bead notes must say exactly which criteria are
satisfied, which are deferred, and which follow-up bead owns the remainder.
This is a velocity rule: fewer, more complete integration boundaries beat a
chain of locally-correct but strategically thin slices.

## Boundaries

- Tracked scaffold explains how to resume.
- Tracked conductor protocol includes `README.md`, `INDEX.md`, `RUNBOOK.md`,
  process docs, and self-prompts.
- Ignored conductor state explains what is happening now.
- Ignored demos are current curated artifacts, not historical dumps.
- Ignored scratch is supporting research, not active loop state or generated
  dump storage.
- Ignored archive/task-history shelves are support surfaces only. They may keep
  compact archaeology, local execution ledgers, runtime baselines, and generated
  process evidence, but they must not become startup packets or handoff mirrors.
- Generated manifests are acceptable; duplicated script/readme snapshots are
  not.
- The active conductor packet may contain ignored logs and generated sidecars,
  but growth must be visible. `devloop-status --json` exposes
  `agent_packet.bytes`, root file counts, `OPERATING-LOG.md` bytes, and
  `EVENTS.jsonl` bytes; `devloop-review` warns only when the active packet
  crosses a soft clutter budget.

## Scratch And Demo Policy

`.agent/scratch/` is not active loop state. It may contain `README.md` and
supporting research notes, but not generated JSON/JSONL/CSV/log dumps, old
handoff packets, copied exports, or active logs.

`.agent/demos/` is current-curated rather than append-only. Demos should be
regenerated, consolidated, or retired when a better artifact supersedes them.
Required demo shelf files are:

- `README.md`
- `SUMMARY_INDEX.json`
- `MANIFEST.readable.json`

`CURATED_CATALOG.md` is recommended when the shelf contains multiple inspectable
demo packets. Avoid checked-in or generated `CONCATENATED_READABLE.md`; external
bundle tooling owns portable concatenation.

## Gitignore Policy

The repo should track reusable scaffold and ignore live local state. The
canonical shape is:

```gitignore
.agent/*
!.agent/README.md
!.agent/DEVLOOP.md
!.agent/includes/
!.agent/includes/**
!.agent/scripts/
!.agent/scripts/**
!.agent/tools/
!.agent/tools/**
!.agent/conductor-devloop/
.agent/conductor-devloop/*
!.agent/conductor-devloop/README.md
!.agent/conductor-devloop/INDEX.md
!.agent/conductor-devloop/RUNBOOK.md
!.agent/conductor-devloop/PROCESS.md
!.agent/conductor-devloop/TACTICS.md
!.agent/conductor-devloop/VELOCITY.md
!.agent/conductor-devloop/SELF-PROMPTS.md
!.agent/conductor-devloop/ADVERSARIAL-REVIEW.md
!.agent/scratch/
.agent/scratch/*
!.agent/scratch/README.md
```

Tracked scaffold must be present on checkout. Ignored `ACTIVE-LOOP.md`,
`OPERATING-LOG.md`, `EVENTS.jsonl`, `DEMO-RADAR.md`, demos, archive, and
task-history are local current state.

## Archive Claim Discipline

- A matching active index schema means the index tier can be opened; it is not
  proof that all acquired source rows are materialized.
- `devloop-status --json` must expose archive convergence signals needed for
  live claims. In Polylogue that includes raw/index join gaps, unresolved raw
  materialization debt, replayable raw rows, and core read-model counts such as
  `index.observed_events`.
- Treat `convergence.raw_materialization_join_gaps` as the broad raw/index
  join-gap count and `convergence.raw_materialization_replayable` as the
  narrower acquired-but-unparsed replay queue. Join gaps are not automatically
  repair debt: `convergence.raw_materialization_debt` is the unresolved
  actionable/open/blocked materialization count after classification.
- If raw materialization debt is nonzero, demos may still be valid for the rows
  they query, but agents must not claim full archive convergence until the debt
  is repaired or classified. Classified join gaps should be quoted separately
  rather than hidden or called pending materialization.

## Polylogue Migration Target

- Keep `.agent/conductor-devloop/` as active state.
- Keep `EVENTS.jsonl` generated from `OPERATING-LOG.md`.
- Keep `.agent/demos/` regenerable and current-curated.
- Split durable process/architecture memory into `.agent/includes/`.
- Stop copying scripts into `.agent/conductor-devloop/scripts/`.
- Use `devloop-review` to catch drift from these conventions.

## Schelling Points

Both Polylogue and Sinex agents can converge independently on these rules:

1. Implement the canonical script name set.
2. Use `.agent/conductor-devloop/` as active loop root.
3. Keep `.agent/scratch/` out of startup state.
4. Stop maintaining duplicated mirrors or script snapshots.
5. Make `devloop-review` enforce local convention drift.
6. Make `devloop-status` print the canonical root and next action.
7. Keep external bundling separated from active conductor/demo state.
