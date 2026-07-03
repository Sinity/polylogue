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

1. Pick the highest-value live archive capability slice. Start from
   `bd ready --json` — beads is the durable backlog and dependency graph;
   priorities encode the operator tier frame (P0 campaigns first). Claim the
   chosen bead (`bd update <id> --claim`); if the slice is not tracked yet,
   create it first.
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

## External-Proof Campaigns

A campaign is a bounded goal whose terminal state is an externally legible
artifact — one a stranger with no repo context can read, believe, and (via the
seeded demo path) run. Campaigns outrank open-ended substrate slices; enabler
substrate work is in scope only when the specific campaign artifact would be
false or fragile without it.

Current campaign sequence (operator direction; supersede only with recorded
evidence, never delete as duplicate):

1. Claim-vs-evidence finding — commit `af4915d11` was the first bounded
   slice, not the terminal state. Finding-grade requires: stated sample
   frame; Codex/GPT coverage (the action evidence lane, not only
   `tool_result` rows); ambiguous-bucket characterization; benign-recovery
   vs consequential-silence split; acknowledge-later sensitivity window;
   marker precision/recall calibrated on a hand-labeled sample; seeded
   stranger-runnable demo.
2. agent-forensics regeneration with all-provider repricing (per-provenance
   or all-provider headline; cache-inclusion disambiguated).
3. Handoff-pack two-arm uplift experiment (pack arm vs raw-ref arm, same
   continuation task, measured).

Campaign rules:

- **Capabilities may not be silos; demos may.** A demo/report/finding
  artifact is a derivative product — a particular application of Polylogue —
  and its packaging (README, charts, narrative, a devtools wrapper) is a
  legitimate one-off. The FACTS it relies on are not: any fact a finding
  needs (e.g. action outcome fields, failure follow-up classification) must
  land as composable product capability — a unit field, projection, or
  rendering primitive the product keeps — not as bespoke table-reads that
  leave the product no more expressive than before. Test: after the demo
  ships, can the next differently-shaped question about the same facts be
  answered by composition, without another script?
- **Cold-reader gate.** A campaign artifact reaches terminal state only
  after a fresh agent, given ONLY the artifact directory, can state what it
  proves, name its sample frame and caveats, and reproduce it. Record the
  cold-read as part of the demo packet.
- **Slice closure is not campaign closure.** Committing a bounded slice
  does not retire the campaign; the campaign stays the top of the priority
  frame until its terminal state is recorded.
- **Operator-direction preservation.** Sections marked as operator
  direction (priority frames, queued campaign directives, operator-sourced
  backlog items) are superseded with a recorded rationale in the operating
  log, never deleted as duplicates during sync/cleanup.
- **Beads is the durable directive channel.** Campaigns are P0 epic beads
  with dependency-encoded sequencing; operator directives arrive as P0/P1
  beads or `bd human` flags, and survive any amount of packet cleanup.
  Closing a slice bead never closes its campaign epic; the epic closes only
  on recorded terminal state. Discovered work becomes a linked bead
  (`--deps discovered-from:<id>`).
