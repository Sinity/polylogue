# Long-Haul Benchmark Campaigns

Reproducible performance validation for Polylogue at various archive scales.

## Quick Start

```bash
# Run small campaign (~1K messages, ~30s)
python -m devtools.run_campaign --scale small

# Run medium campaign (~10K messages, ~5min)
python -m devtools.run_campaign --scale medium

# Run a specific campaign
python -m devtools.run_campaign --scale small --campaign fts-rebuild

# List available campaigns
python -m devtools.run_campaign --list

# Or use the named validation lane
python -m devtools.run_validation_lanes --lane long-haul-small
```

## Scale Levels

| Level | Messages | Conversations | Typical Duration |
|-------|----------|--------------|-----------------|
| small | 1,000 | 100 | ~30s |
| medium | 10,000 | 500 | ~5min |
| large | 100,000 | 2,000 | ~30min |
| stretch | 1,000,000 | 10,000 | ~4h |

## Campaigns

| Campaign | Description |
|----------|-------------|
| fts-rebuild | Full FTS5 index rebuild from scratch |
| incremental-index | Batch-wise incremental FTS index updates |
| filter-scan | Common filter query latency (provider, date, tool use) |
| startup-health | `check --runtime` startup speed (count, stats, latest run) |

## Reports

Campaign reports are written to this directory as:
- `<date>-<scale>.md` -- Human-readable Markdown
- `<date>-<scale>.json` -- Machine-readable JSON

## Architecture

The benchmark infrastructure consists of:

1. **`devtools/large_archive_generator.py`** -- Synthetic archive generator using polylogue's SyntheticCorpus engine
2. **`devtools/benchmark_campaigns.py`** -- Campaign runner with timing instrumentation
3. **`devtools/campaign_report.py`** -- Markdown/JSON report generator
4. **`devtools/run_campaign.py`** -- CLI entry point (`python -m devtools.run_campaign`)
5. **`tests/unit/storage/test_scale.py`** -- Pytest-integrated performance budget assertions

## Reproducibility

All campaigns accept a `--seed` flag for deterministic archive generation.
The default seed is 42. Reports include the scale level, timestamp, and
database statistics for meaningful cross-run comparisons.
