# Agent Forensics

A longitudinal agent-usage forensics packet, over the deterministic seeded
demo archive (`polylogue demo seed`).

This packet replaces the retired schema-v23 full-report packet for current
cardinality and headline token/cost claims. It uses product analysis
surfaces, not the deleted standalone `scripts/agent_forensics.py` path.

**This packet is private-data-free.** A prior version of this packet was
generated against a live operator archive and committed real corpus size,
token totals, per-model spend in USD, and the operator's archive path to
this public repo (polylogue-0bgr). It has been regenerated from scratch
against the deterministic seeded fixture archive; every number below
describes the ~19-session synthetic fixture, not any operator's real usage.

## What This Proves

Polylogue can regenerate an agent-usage forensics packet from an archive
using normal analysis commands. The result keeps distinct evidence claims
separate:

- physical-session archive totals;
- logical-session high-water totals;
- priced vs origin-reported cost lanes;
- current origin coverage;
- month/origin/model usage timeline rows.

## Current Headline (seeded fixture)

Generated: 2026-08-02T13:52:20Z
Archive root: `/path/to/demo-archive`
Index schema: v54

- physical sessions: 19
- messages: 71
- blocks: 121
- materialized session profiles: 19
- origin coverage rows: 8
- usage timeline rows: 5
- physical-session tokens accounted: 388,872
- logical-session high-water tokens accounted: 388,872
- replay-chain gap: 0
- Claude Code physical/logical tokens: 388,500 / 388,500
- Codex physical/logical tokens: 0 / 0 (no priced Codex sessions in the fixture)
- stored provider-priced cost: $2.84
- catalog API-equivalent cost: $2.84
- logical catalog API-equivalent cost: $2.84

## Caveats

This is not provider billing truth. It does not query provider accounts. Cost
figures are archive/provider-reported lanes plus catalog API-equivalent pricing
where the pricing catalog matches the model.

This is not a resurrection of the old standalone forensics script. The old
script was intentionally folded into product analysis surfaces. This packet is
the current demo/finding layer over those surfaces.

This is not evidence about any operator's real usage or spend. The headline
numbers above describe the deterministic seeded fixture only.

The cost-rollups drilldown command completes immediately on this fixture (see
`current/cost-rollups-timeout.txt`). The prior live-archive version of this
packet recorded a genuine 120-second timeout on that command at live-archive
scale; that is a real product-performance finding, tracked as a separate
follow-up, and does not reproduce on a fixture this small.

Structured failure follow-up behavior is covered by the current
`claim-vs-evidence` packet. This packet links to that current demo rather than
duplicating bounded failure-follow-up samples here.

## Regenerate

```bash
polylogue demo seed --root ./demo-archive --force --with-overlays
export POLYLOGUE_ARCHIVE_ROOT="$PWD/demo-archive"

polylogue --plain ops diagnostics workload --json \
  > .agent/demos/agent-forensics/current/archive-workload.json

polylogue --plain analyze usage --detail headline --format json --limit 0 \
  > .agent/demos/agent-forensics/current/usage-headline-all.json

polylogue --plain analyze usage --detail headline --origin claude-code-session --format json --limit 0 \
  > .agent/demos/agent-forensics/current/usage-headline-claude-code.json

polylogue --plain analyze usage --detail headline --origin codex-session --format json --limit 0 \
  > .agent/demos/agent-forensics/current/usage-headline-codex.json

polylogue --plain analyze insights coverage --group-by origin --format json --limit 1000 \
  > .agent/demos/agent-forensics/current/coverage-origin.json

polylogue --plain analyze insights usage-timeline --group-by month-origin-model --format json --limit 500 \
  > .agent/demos/agent-forensics/current/usage-timeline-month-origin-model.json

devtools workspace demo-shelf
```

Never regenerate this packet with `POLYLOGUE_ARCHIVE_ROOT` pointed at a live
operator archive -- this is a committed public-repo artifact, and doing so is
exactly the mistake polylogue-0bgr fixed.

## Files

- `current/summary.json` — claim/non-claim, headline numbers, caveats, command proofs.
- `current/archive-workload.json` — fixture-archive tier/cardinality snapshot.
- `current/usage-headline-*.json` — all-provider and provider-specific usage lanes.
- `current/coverage-origin.json` — origin coverage table.
- `current/usage-timeline-month-origin-model.json` — month/origin/model timeline.
- `current/cost-rollups-timeout.txt` — cost-rollups drilldown proof (completes on the fixture; documents the live-archive timeout finding it originally captured).
