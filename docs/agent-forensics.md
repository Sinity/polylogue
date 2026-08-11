# Agent Usage Forensics

Polylogue mines longitudinal agent usage through the normal archive analysis
surfaces, not a standalone report script. The useful outputs are composable:
coverage describes archive shape, cost rollups describe provider/model spend
evidence, usage timelines describe monthly token/cost movement, and the
action query surface exposes provider-reported tool outcomes and their
immediate follow-up classification.

All commands below are read-only against the archive.

## Queries

```bash
# Scale and adoption curve.
polylogue analyze insights coverage --group-by month --format json

# Provider/model cost and subscription-credit evidence.
polylogue analyze insights cost-rollups --format json
polylogue analyze insights cost-rollups --model gpt-5-codex --format json

# Provider usage audit with physical/logical token-grain labels.
polylogue analyze usage --format json --limit 0
polylogue analyze usage --origin claude-code-session --format json --limit 0
polylogue analyze usage --origin codex-session --format json --limit 0

# Monthly usage movement by origin and model.
polylogue analyze insights usage-timeline --group-by month-origin-model --format json

# Inspect structurally failed actions and their immediate follow-up class.
polylogue --format json actions where is_error:true
```

The same analysis can be reproduced against the deterministic demo archive:

```bash
polylogue demo seed --root /tmp/demo-archive --force --with-overlays --format json
POLYLOGUE_ARCHIVE_ROOT=/tmp/demo-archive \
  polylogue analyze insights usage-timeline --format json
POLYLOGUE_ARCHIVE_ROOT=/tmp/demo-archive \
  polylogue --format json actions where is_error:true
```

## What The Surfaces Report

- **Scale and span:** sessions, messages, origins, and time buckets through
  `coverage`.
- **Token economy:** input, output, cache-read, cache-write, total, and
  reasoning token lanes through `usage-timeline`.
- **Token grain:** physical-session and logical-session-model-high-water
  rollups through `analyze usage`; physical archive totals and logical work
  totals are distinct claims.
- **Cost evidence:** stored/provider-priced cost, catalog API-equivalent
  estimates, catalog coverage gaps, and subscription-credit estimates through
  `cost-rollups` and `usage-timeline`.
- **Model evolution:** usage buckets grouped by month, origin, and model.
- **Structured failure follow-up:** action queries anchor on provider-reported
  tool-result failures (`is_error=1` or non-zero `exit_code`) and expose the
  immediately following assistant turn's acknowledgment classification.

## Accuracy Notes

These surfaces read the archive's materialized analytics tables
(`session_model_usage`, `session_provider_usage_events`, `sessions`,
`actions`, `messages`, and `blocks`) and keep distinct evidence streams
distinct:

1. **Per-event deltas vs cumulative totals.** `session_provider_usage_events`
   carries both `last_*` per-event deltas and `total_*` cumulative running
   totals. Usage timelines sum the `last_*` deltas; summing cumulative columns
   would over-count.
2. **Cost provenance.** Only `priced` rows carry stored `cost_usd`.
   `origin_reported` rows can be catalog-priced as an API-equivalent estimate
   when their model matches the shared pricing catalog, but that derived
   estimate is not provider billing truth and does not change the underlying
   provenance.
3. **Subscription reality.** API-list-equivalent cost is not the same thing as
   a Claude Max/Pro subscription. Subscription-credit estimates use the shared
   dated pricing catalog and do not charge cache-read tokens.
4. **Token grain.** `session_model_usage` is a physical-session evidence stream.
   `polylogue analyze usage` also exposes `logical_session_model_high_water`,
   which collapses fork/resume/replay chains by logical session and model. Use
   the physical view for archive-materialization claims and the logical view for
   logical-work claims; do not silently substitute one for the other.
5. **Failure classification.** Action queries do not infer tool success or
   failure from assistant prose. Structured tool-result fields are the evidence
   anchor; prose is only a follow-up acknowledgment signal.

Current caveat: all-provider logical-session repricing is still being repaired.
Treat physical usage totals as current archive measurements, not final billing
reconciliation or logical-work totals.

## Privacy

Aggregate analysis outputs omit message content, session titles, and source
paths by default. Action queries can return archived content, so treat their
output according to the archive's privacy boundary.
