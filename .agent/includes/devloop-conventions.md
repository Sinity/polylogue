# Devloop Conventions

Polylogue uses the shared conductor conventions agreed with the Sinex devloop.
The goal is not identical repositories; it is transferable operating behavior.

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

## Boundaries

- Tracked scaffold explains how to resume.
- Tracked conductor protocol includes `README.md`, `INDEX.md`, `RUNBOOK.md`,
  process docs, and self-prompts.
- Ignored conductor state explains what is happening now.
- Ignored demos are current curated artifacts, not historical dumps.
- Ignored scratch is supporting research, not active loop state.
- Generated manifests are acceptable; duplicated script/readme snapshots are
  not.
- The active conductor packet may contain ignored logs and generated sidecars,
  but growth must be visible. `devloop-status --json` exposes
  `agent_packet.bytes`, root file counts, `OPERATING-LOG.md` bytes, and
  `EVENTS.jsonl` bytes; `devloop-review` warns only when the active packet
  crosses a soft clutter budget.

## Archive Claim Discipline

- A matching active index schema means the index tier can be opened; it is not
  proof that all acquired source rows are materialized.
- `devloop-status --json` must expose archive convergence signals needed for
  live claims. In Polylogue that includes `convergence.raw_materialization_debt`
  and core read-model counts such as `index.observed_events`.
- Treat `convergence.raw_materialization_debt` as the broad raw/index join-gap
  count and `convergence.raw_materialization_replayable` as the narrower
  acquired-but-unparsed replay queue. The latter is a repair actuator hint; the
  former still needs debt classification before claiming convergence.
- If raw materialization debt is nonzero, demos may still be valid for the rows
  they query, but agents must not claim full archive convergence until the debt
  is repaired or classified.

## Polylogue Migration Target

- Keep `.agent/conductor-devloop/` as active state.
- Keep `EVENTS.jsonl` generated from `OPERATING-LOG.md`.
- Keep `.agent/demos/` regenerable and current-curated.
- Split durable process/architecture memory into `.agent/includes/`.
- Stop copying scripts into `.agent/conductor-devloop/scripts/`.
- Use `devloop-review` to catch drift from these conventions.
