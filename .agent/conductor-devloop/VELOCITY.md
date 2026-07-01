# Polylogue Devloop Velocity

This is the cadence rubric for the indefinite Polylogue conductor loop.

## Cadence

- Every status report names the archive root, schema version, session count,
  message count, and whether the daemon is converging, idle, or blocked.
- Every long-running daemon/import/test action records a start line in
  `OPERATING-LOG.md`, then a result line with counts or failure shape.
- Run `.agent/scripts/devloop-review` before broad status claims and after any
  archive-root/daemon move.
- Run `.agent/scripts/devloop-sync` after changing the current conductor notes
  so the local `.agent/conductor-devloop` manifest, event sidecar, and helper
  snapshots remain coherent.
- Prefer one active daemon per archive root. Stop or quarantine stale daemons
  before interpreting counts.
- Prefer narrow verification while the host is under RAM/IO pressure. Record
  killed/interrupted runs as host/runtime facts, not product proof.
- Treat `devloop-status` pressure output as the live-timing gate. If borg,
  materialization, broad tests, or D-state I/O are active, live latency and
  throughput probes are not admissible proof; use query plans/focused tests or
  wait for the pressure window to close.

## Acceleration Rules

- Make stale states impossible to confuse with current states: put obsolete
  databases in a named quarantine with a manifest, or delete them only after
  their value is clearly superseded.
- Do not make bespoke silos permanent. Demos may be bespoke artifacts, but
  durable capability should land in query, acquisition, projection, rendering,
  or insight substrate.
- Use live archive proof, but never quote a count without the root and schema.
- Collapse public compatibility trash decisively when the replacement substrate
  exists; do not leave hidden rot under old names.
- When a demo is useful, update `.agent/demos` as the current curated demo
  shelf. Use `/realm/inbox` only when an explicit task names it.
- Use `.agent/scripts/devloop-velocity` for measured process reflection. The
  useful signal is not "more mode switches"; it is whether Proof closes through
  Artifact/Velocity, whether dwell times are explained by useful work, and
  whether heavy archive commands are being stacked under host I/O pressure.
- `Direction -> Proof` is valid for very small test/regression slices where
  evidence and construction are already trivial from the selected diff.
  `Velocity -> Artifact` is valid for shelf/presentation polish that directly
  improves inspectability of an already-produced artifact. Both should stay
  rare; if either becomes routine, switch back to Direction and ask what slice is
  being avoided.

## Archive Count Vocabulary

- `canonical active archive`: `/home/sinity/.local/share/polylogue`; as of the
  2026-06-30 collapse this is the promoted v18 dev archive.
- `old dev archive`: `/realm/tmp/polylogue-dev/archive`; should not exist as a
  live archive after the collapse. If it reappears, treat it as drift.
- `historical backup`: archive-db-backups under XDG/captures, schema v6-v8,
  useful for archaeology but not a current runtime target.
