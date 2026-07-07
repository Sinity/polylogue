# Agent Forensics

Historical live-archive report generated from
`/home/sinity/.local/share/polylogue` on 2026-07-03. This packet is retained as
the latest full forensics report, but it is not the current archive
cardinality source after the v24 rebuild. Use
`../temporal-archive-aggregates/current/archive-cardinality.json` and
`../archive-debt-summary/summary.json` for current v24 archive counts until the
full forensics packet is regenerated after session-profile convergence.

```bash
export POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue
polylogue --plain analyze usage --detail headline --format json --limit 0 \
  > /realm/tmp/polylogue-cost-reconciliation/provider-usage-all-logical-current.json
polylogue --plain analyze usage --origin claude-code-session --format json --limit 0 \
  > /realm/tmp/polylogue-cost-reconciliation/provider-usage-claude-logical-current.json
polylogue --plain analyze usage --origin codex-session --format json --limit 0 \
  > /realm/tmp/polylogue-cost-reconciliation/provider-usage-codex-logical-current.json
polylogue --plain analyze insights cost-rollups --format json \
  > /realm/tmp/polylogue-cost-reconciliation/cost-rollups-current.json
```

## What This Proves

- The report archive opened at index schema v23 when generated.
- Physical archive scale at report generation time: 16,498 sessions and
  4,142,175 messages.
- Current v24 archive scale, from the refreshed demo shelf: 16,627 physical
  sessions, 8,730 logical root sessions, and 4,254,615 messages.
- Token aggregates can be audited from current `session_model_usage` and
  provider usage rows with explicit grain labels.
- `origin_reported` providers remain provenance-distinct, while the report
  also computes a separate catalog API-equivalent estimate from Polylogue's
  shared vendored LiteLLM pricing catalog.
- The SVG chart set renders as valid XML.

## Current Headline

- Physical archive token view: 395.3B, from physical-session
  `session_model_usage` rows.
- Logical-session high-water token view: 288.7B across 8 origins. The 106.6B
  all-provider replay gap is the current headline construct-validity caveat.
- Claude Code logical-session high-water view: 153.5B tokens, versus 175.8B
  in the Claude Code physical-session view. The 22.3B-token gap is live
  evidence that physical replay chains must not be used as the only headline.
- Codex logical-session high-water view: 135.2B tokens, versus 219.5B in the
  Codex physical-session view. This 84.2B-token gap is the largest current
  reason the all-provider physical headline is not a logical-work headline.
- Stored/provider-priced subset: $243,392.19.
- Catalog API-equivalent estimate across matched provenances: $337,565.03.
- Structured failure follow-up section is bounded to 5,000 failed outcomes and
  labels that bound in the report.

## Freshness Note

The v24 `polylogue analyze usage --detail headline` probe reports a current
physical-session token view of 397.2B, stored/provider-priced subset
`$244,117.12`, and catalog API-equivalent estimate `$339,994.76`. The same
probe's logical high-water output is not comparable to this packet yet because
session profile convergence is still partial; the temporal aggregate packet
currently reports 4,413 materialized profiles out of 16,627 physical sessions.
Do not cite the logical high-water numbers from this packet as current until
the full report is regenerated after profile convergence.

## Delta vs 2026-06-27 Packet

The retired 2026-06-27 report headline was 546.6B tokens and $89,367.84
priced cost. The current archive/report is not directly comparable as a clean
time-series point because archive deduplication, schema changes, provider
coverage, and pricing coverage changed between runs. The useful comparison is:

- physical-session token view is now 395.3B, down 151.3B from the old stale
  report;
- stored/provider-priced cost is now $243,392.19, up $154,024.35 from the old
  priced-subset headline;
- the new report adds a separate all-matched-provenance catalog API-equivalent
  estimate of $337,565.03, including $94,172.84 from `origin_reported` rows that
  previously had no dollar estimate.

## Caveats

Logical-session token attribution is still an open blocker for the all-provider
forensics headline. Treat the physical-session token total as an archive
measurement, and use logical-session high-water numbers when the claim is
"logical work performed" rather than "physical rows currently materialized."
Catalog API-equivalent cost remains an estimate, not final billing
reconciliation.

The run happened while `borgbackup-job-realm` was active. Timings printed by
the script showed the live archive was I/O pressured: scale/span 10.306s,
reasoning deltas 26.448s, and bounded failure follow-up 22.769s.

## Files

- `report.md` - human-readable report with embedded chart links and explicit
  token-grain caveats.
- `charts/*.svg` - standalone charts.
- `structured_failure_followups.json` - bounded ref-backed failure follow-up
  samples and rollups.
