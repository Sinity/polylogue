# Polylogue Devloop Process

## Startup

Run:

```bash
.agent/scripts/devloop-review
```

Read `README.md`, `RUNBOOK.md`, `VELOCITY.md`, `PROCESS.md`, `TACTICS.md`,
and `ADVERSARIAL-REVIEW.md` before trusting memory. If ignored local state
exists, also read `ACTIVE-LOOP.md`, `OPERATING-LOG.md`, and `DEMO-RADAR.md`.
Then inspect the active archive root and daemon state. Counts are never
meaningful without root, schema version, session count, message count, and
whether convergence is running.

## Loop Shape

1. Pick the highest-value live archive capability slice.
2. Record a timestamped operating-log entry with `.agent/scripts/devloop-log`.
3. Gather evidence from source, archive, daemon logs, and existing notes.
4. Make the smallest shared-substrate change that advances the slice.
5. Prove the exact claim with a focused command or real archive artifact.
6. Update `.agent/demos` or the conductor packet with useful readable proof.
7. Run `.agent/scripts/devloop-sync` after changing scaffold/current notes.
8. During Meta/process slices, run `.agent/scripts/devloop-velocity --record`
   before choosing the process change. This turns speed/focus reflection into
   an operating-log artifact instead of an unrecorded chat judgment.

Value and substrate are interleaved, not alternatives. A visible demo/finding is
the forcing function, but substrate repair is the correct slice when broken
archive state, query semantics, rendering, or daemon convergence would make the
artifact false, stale, or fragile. The smell is generic cleanup without a
specific artifact or invariant it protects, not repair itself.

Use `.agent/scripts/devloop-focus <from> <to> "<trigger>" "<decision>"` for
material focus changes so the role switch is logged, reflected in
`ACTIVE-LOOP.md`, and reflected in the conductor packet.

Because current state is ignored locally, `.agent/conductor-devloop/` contains
tracked process docs plus ignored local state. `devloop-sync` refreshes the
ignored event sidecar, demo indexes, packet manifest, and tracked script-hash
list; it does not mirror state to `/realm/inbox` or copy script snapshots into
the conductor packet.

## Non-Negotiables

- One canonical archive root. Quarantine obsolete DB roots instead of leaving
  them discoverable as current.
- One daemon writing the canonical archive. Prod and dev daemons must not both
  run against competing roots.
- No permanent one-off recovery/export/report silos when a query/projection/
  rendering primitive can own the capability.
- Public flags and compatibility fronts are removed decisively once the DSL or
  projection substrate replaces them.
- Insight/pathology/postmortem outputs must say whether evidence is structural,
  source-reported, derived, heuristic, candidate, or unsupported.
- Meta/scaffold work must leave an executable consequence: a review check,
  corrected synced packet, sharper stop rule, or materially better next-slice
  choice. Process prose alone is not a completed slice.
- Meta/scaffold work must record velocity/focus evidence with
  `devloop-velocity --record`; `devloop-review` treats a missing audit as drift.

## Proof Ladder

Source review proves shape. Focused tests prove parser/storage semantics. CLI,
MCP, API, and daemon probes prove surface contracts. Real archive demos prove
operator value. Broad `devtools verify` proves phase readiness, not every edit.
