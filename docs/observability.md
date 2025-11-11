# Observability & Telemetry

Polylogue records every run, exposes live status summaries, and emits machine-readable data so you can wire the CLI into dashboards or schedulers without scraping interactive output.

## Structured Run Logs

Every command writes a single JSON line to stderr when `POLYLOGUE_RUN_LOG` is unset or `1`:

```json
{"event":"polylogue_run","cmd":"sync drive","provider":"drive","count":6,"attachments":12,"duration":4.2}
```

Set `POLYLOGUE_RUN_LOG=0` to silence the log when embedding Polylogue inside other scripts. The same payloads back `polylogue status` so retry counts, attachment totals, and provider metadata stay consistent everywhere.

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

## Automation Hooks

`polylogue automation` already emits systemd/cron snippets. Add `--status-log /path/to/summary.json --status-limit 40` so every scheduled run writes `polylogue status --dump-only --dump …` after the main command. The new `--summary` options make it simple to append aggregate metrics alongside the raw run history.

## Where the Data Lives

- Raw runs: `$XDG_STATE_HOME/polylogue/runs.json`
- SQLite metadata (conversations, branches, messages, provider summaries): `$XDG_STATE_HOME/polylogue/polylogue.db`
- Structured run logs: stderr (`POLYLOGUE_RUN_LOG=0/1` toggles emission)

Use these locations as backup targets or feed them into external monitoring stacks.
