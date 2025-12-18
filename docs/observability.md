# Observability & Telemetry

Polylogue records every run, exposes live status summaries, and emits machine-readable data so you can wire the CLI into dashboards or schedulers without scraping interactive output.

## `polylogue browse status`

`polylogue browse status` reads the SQLite run history and supports a rich set of export modes:

- `--providers drive,codex` filters the summaries (and any dumps/summaries) to the listed providers.
- `--dump runs.json --dump-limit 50` writes the last N raw run records to a file (use `-` for stdout).
- `--summary metrics.json` emits aggregate run/provider summaries as JSON so dashboards never have to parse tables. Combine with `--summary-only` for a purely machine-readable output.
- `--watch --interval 10` refreshes the status tables (and any requested dump/summary) on a cadence—ideal for terminal dashboards.
- Credential visibility: `browse status --json` and `config show --json` include the credential/token paths plus any env overrides (`POLYLOGUE_CREDENTIAL_PATH`, `POLYLOGUE_TOKEN_PATH`), making it easy to confirm which files headless jobs will use. Add `--top N` to status to include ranked runs, and `--inbox` to surface pending inbox items.

Examples:

```bash
# Filter to Drive runs and emit both human and JSON summaries
polylogue browse status --providers drive --summary metrics.json

# Headless summary streaming (stdout JSON every 30s)
polylogue browse status --providers codex --summary-only --summary - --watch --interval 30
```

For unattended jobs, call `polylogue browse status --dump /path/to/runs.json --dump-limit 50` (and optionally `--summary metrics.json`) immediately after your sync/import command so dashboards and alerts can consume fresh JSON without parsing terminal output.

## `polylogue browse metrics`

For Prometheus-compatible metrics, use:

```bash
# Print Prometheus text format to stdout
polylogue browse metrics

# Serve /metrics over HTTP (localhost:8000 by default)
polylogue browse metrics --serve --host 127.0.0.1 --port 8000

# JSON payload for debugging or custom collectors
polylogue browse metrics --json
```

## Where the Data Lives

- Run history + metadata: `$XDG_STATE_HOME/polylogue/polylogue.db` (use `polylogue browse status --dump runs.json` or `--dump -` when you need a JSON snapshot)
- Conversations/branches/messages (same SQLite DB): `$XDG_STATE_HOME/polylogue/polylogue.db`

Use these locations as backup targets or feed them into external monitoring stacks.

Tip: `polylogue config show` prints all resolved paths (config, output directories, state DB) in either human-readable or JSON form (pass `--json`), making it easy to confirm where data lands before wiring up automation.
