# Observability

Polylogue writes run metadata to disk and keeps a SQLite history so automation can consume results without scraping terminal output.

## Run ledger

- Every `run` writes `archive_root/runs/run-<timestamp>-<run_id>.json`.
- The same run records are stored in `$XDG_STATE_HOME/polylogue/polylogue.db` (table: `runs`).
- Render output lives under `render_root` (defaults to `archive_root/render`).

## Health checks

- `polylogue check` validates config, archive root, DB reachability, index status, and Drive credential/token presence.
- `polylogue check --repair` fixes issues that can be auto-fixed.
- `polylogue check --vacuum` compacts the database and reclaims space.

## Path inspection

- `polylogue run --preview` prints resolved sources and output paths.
- Use `POLYLOGUE_RENDER_ROOT` to override render output without editing config.
