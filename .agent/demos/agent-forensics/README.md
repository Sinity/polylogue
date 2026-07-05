# Agent Forensics

Current v24 longitudinal agent-usage finding over the live Polylogue archive.

This packet replaces the retired schema-v23 full-report packet for current
cardinality and headline token/cost claims. It uses product analysis surfaces,
not the deleted standalone `scripts/agent_forensics.py` path.

## What This Proves

Polylogue can regenerate a current agent-usage forensics packet from the live
archive using normal analysis commands. The result keeps distinct evidence
claims separate:

- physical-session archive totals;
- logical-session high-water totals;
- priced vs origin-reported cost lanes;
- current origin coverage;
- month/origin/model usage timeline rows.

## Current Headline

Generated: 2026-07-05T07:36:52Z
Archive root: `/home/sinity/.local/share/polylogue`
Index schema: v24

- physical sessions: 16,816
- messages: 4,364,655
- blocks: 665,890
- materialized session profiles: 16,816
- origin coverage rows: 8
- usage timeline rows: 329
- physical-session tokens accounted: 399.9B
- logical-session high-water tokens accounted: 292.9B
- replay-chain gap: 107.0B
- Claude Code physical/logical tokens: 176.4B / 153.7B
- Codex physical/logical tokens: 223.4B / 139.2B
- stored provider-priced cost: $247,949.99
- catalog API-equivalent cost: $344,935.46
- logical catalog API-equivalent cost: $285,519.69

## Caveats

This is not provider billing truth. It does not query provider accounts. Cost
figures are archive/provider-reported lanes plus catalog API-equivalent pricing
where the pricing catalog matches the model.

This is not a resurrection of the old standalone forensics script. The old
script was intentionally folded into product analysis surfaces. This packet is
the current demo/finding layer over those surfaces.

The cost-rollups drilldown command timed out after 120 seconds in this run:

```bash
POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue \
  polylogue --plain analyze insights cost-rollups --format json --limit 100
```

The current cost claims therefore cite `analyze usage --detail headline`, which
completed and carries the pricing lanes used in `summary.json`. The timeout is a
product-performance follow-up, not hidden evidence.

Structured failure follow-up behavior is covered by the current
`claim-vs-evidence` packet. This packet links to that current demo rather than
duplicating bounded failure-follow-up samples here.

## Regenerate

```bash
POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue \
  polylogue --plain ops diagnostics workload --json \
  > .agent/demos/agent-forensics/current/archive-workload.json

POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue \
  polylogue --plain analyze usage --detail headline --format json --limit 0 \
  > .agent/demos/agent-forensics/current/usage-headline-all.json

POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue \
  polylogue --plain analyze usage --detail headline --origin claude-code-session --format json --limit 0 \
  > .agent/demos/agent-forensics/current/usage-headline-claude-code.json

POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue \
  polylogue --plain analyze usage --detail headline --origin codex-session --format json --limit 0 \
  > .agent/demos/agent-forensics/current/usage-headline-codex.json

POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue \
  polylogue --plain analyze insights coverage --group-by origin --format json --limit 1000 \
  > .agent/demos/agent-forensics/current/coverage-origin.json

POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue \
  polylogue --plain analyze insights usage-timeline --group-by month-origin-model --format json --limit 500 \
  > .agent/demos/agent-forensics/current/usage-timeline-month-origin-model.json

devtools workspace demo-shelf
```

## Files

- `current/summary.json` — claim/non-claim, headline numbers, caveats, command proofs.
- `current/archive-workload.json` — live archive tier/cardinality snapshot.
- `current/usage-headline-*.json` — all-provider and provider-specific usage lanes.
- `current/coverage-origin.json` — origin coverage table.
- `current/usage-timeline-month-origin-model.json` — month/origin/model timeline.
- `current/cost-rollups-timeout.txt` — timed-out drilldown evidence.
