# Observability

Polylogue writes run metadata to disk and keeps a SQLite history so automation can consume results without scraping terminal output.

## Run ledger

- Every `run` writes `archive_root/runs/run-<timestamp>-<run_id>.json`.
- The same run records are stored in `$XDG_STATE_HOME/polylogue/polylogue.db` (table: `runs`).
- Render output lives under `render_root` (defaults to `archive_root/render`).

## Health checks

- `polylogue health` validates config, archive root, DB reachability, index status, and Drive credential/token presence.
- Results are cached in `archive_root/health.json` for roughly 10 minutes.

## Exports

- `polylogue export --out /path/to/export.jsonl` dumps conversations, messages, and attachments as JSONL.
- Use this for dashboards or downstream processing.

## Path inspection

- `polylogue config show` prints a short summary; use `polylogue config show --json` for raw config values and env overrides.
