# Observability & Telemetry

Polylogue records every run, exposes live status summaries, and emits machine-readable data so you can wire the CLI into dashboards or schedulers without scraping interactive output.

## `polylogue status`

`polylogue status` reads the SQLite run history and now supports a richer set of export modes:

- `--providers drive,codex` filters the summaries (and any dumps/summaries) to the listed providers.
- `--dump runs.json --dump-limit 50` writes the last N raw run records to a file (use `-` for stdout).
- `--summary metrics.json` emits aggregate run/provider summaries as JSON so dashboards never have to parse tables. Combine with `--summary-only` for a purely machine-readable output.
- `--watch --interval 10` refreshes the status tables (and any requested dump/summary) on a cadence—ideal for terminal dashboards.

Examples:

```bash
# Filter to Drive runs and emit both human and JSON summaries
polylogue status --providers drive --summary metrics.json

# Headless summary streaming (stdout JSON every 30s)
polylogue status --providers codex --summary-only --summary - --watch --interval 30
```

For unattended jobs, call `polylogue status --dump /path/to/runs.json --dump-limit 50` (and optionally `--summary metrics.json`) immediately after your sync/import command so dashboards and alerts can consume fresh JSON without parsing terminal output.

## Where the Data Lives

- Run history + metadata: `$XDG_STATE_HOME/polylogue/polylogue.db` (use `polylogue status --dump runs.json` or `--dump -` when you need a JSON snapshot)
- Conversations/branches/messages (same SQLite DB): `$XDG_STATE_HOME/polylogue/polylogue.db`

Use these locations as backup targets or feed them into external monitoring stacks.

Tip: `polylogue env` prints all resolved paths (config, output directories, state DB) in either human-readable or JSON form, making it easy to confirm where data lands before wiring up automation.
