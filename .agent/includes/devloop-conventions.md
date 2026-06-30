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
- Ignored conductor state explains what is happening now.
- Ignored demos are current curated artifacts, not historical dumps.
- Ignored scratch is supporting research, not active loop state.
- Generated manifests are acceptable; duplicated script/readme snapshots are
  not.

## Polylogue Migration Target

- Keep `.agent/conductor-devloop/` as active state.
- Keep `EVENTS.jsonl` generated from `OPERATING-LOG.md`.
- Keep `.agent/demos/` regenerable and current-curated.
- Split durable process/architecture memory into `.agent/includes/`.
- Stop copying scripts into `.agent/conductor-devloop/scripts/`.
- Use `devloop-review` to catch drift from these conventions.

