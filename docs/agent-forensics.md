# AI-Agent Forensics

`scripts/agent_forensics.py` mines a Polylogue archive for longitudinal usage
findings and emits a Markdown report plus standalone SVG charts. It is the kind
of analysis only a full personal AI-session archive can produce: token economy,
cost (API-equivalent *and* subscription-credit views), model evolution,
temporal rhythm, and workflow shape — over the whole corpus.

It is **pure standard library** (no third-party dependencies, no network), reads
the archive **read-only**, and writes nothing to it. Anyone can run it against
any archive and reproduce the report.

## Run it

```bash
# Against your archive (defaults to $POLYLOGUE_ARCHIVE_ROOT, then the XDG dir):
python scripts/agent_forensics.py --archive ~/.local/share/polylogue --out ./forensics

# Reproduce against the synthetic demo archive (no private data):
polylogue demo seed --root /tmp/demo-archive --force --with-overlays --format json
python scripts/agent_forensics.py --archive /tmp/demo-archive --out ./forensics-demo

open ./forensics/report.md      # report.md + charts/*.svg
```

The report renders in any Markdown viewer; the SVG charts embed inline on GitHub.

## What it reports

- **Scale & span** — sessions, messages, origins, date range.
- **Temporal rhythm** — sessions/month and tokens/month (the adoption curve).
- **Token economy** — input / output / cache-read / cache-write / reasoning,
  **split by cost provenance**. `priced` rows (Claude Code) carry a computed
  `cost_usd`; `origin_reported` rows (other providers) report token counts with
  no per-token cost. The two are never conflated.
- **Cache amplification** — within the priced subset, cache-read tokens vs fresh
  input. In agentic loops this is the dominant volume and the dominant cost
  driver; a token counter that ignores cache reads understates real usage by
  orders of magnitude.
- **Cost** — API-list-equivalent cost by model, per-session cost distribution,
  and provenance breakdown.
- **Subscription reality** — the API-list-equivalent cost is *not* what a Claude
  Max/Pro subscriber pays. On a plan, **cache reads are free** and usage is
  metered in credits (`(input + cache_write) × in_rate + output × out_rate`).
  The report estimates credits consumed (per the
  [she-llac.com](https://she-llac.com/claude-limits) rate analysis — an
  estimate, not official) and frames the API-vs-plan value gap.
- **Model evolution** — tokens/month by top model.
- **Workflow shape** — work-event types and session-length distribution.

## Accuracy notes

Numbers are read directly from the archive's materialized analytics tables
(`session_model_usage`, `session_provider_usage_events`, `session_work_events`,
`sessions`). Two accounting traps the tool handles explicitly:

1. **Per-event deltas vs cumulative totals.** `session_provider_usage_events`
   carries both `last_*` (per-event delta) and `total_*` (cumulative running
   total) columns. Reasoning/token sums use the `last_*` deltas; summing the
   cumulative columns over-counts by orders of magnitude.
2. **Cost provenance.** Only `priced` rows have a real `cost_usd`. Pairing that
   cost with the all-provenance token total would misstate both, so they are
   reported separately.

## Privacy

The report is **aggregate statistics only** — no message content, session
titles, or paths. Even so, an archive owner's usage totals and spend are
personal; run it against your own archive for your own numbers. The author's
real 3.5-year run is kept private (not committed). The committed, reproducible
example uses the synthetic demo archive.
