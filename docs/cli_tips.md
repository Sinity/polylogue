# CLI Tips

Polylogue switches to plain mode automatically when stdout/stderr are not TTYs. Use `--interactive` on a TTY to force prompts, or set `POLYLOGUE_FORCE_PLAIN=1` in CI.

## Drive auth

- Provide an OAuth client JSON at `~/.config/polylogue/polylogue-credentials.json` or set `POLYLOGUE_CREDENTIAL_PATH`.
- Tokens are stored at `~/.config/polylogue/token.json` (or `POLYLOGUE_TOKEN_PATH`).
- Drive auth requires `--interactive` for the browser authorization code.
- Run `polylogue auth` to initiate the OAuth flow.

## Source scoping

- Use `--source NAME` (repeatable) on `run` to avoid reprocessing everything.
- Use `--source last` to reuse the previous interactive selection.
- Example: `polylogue run --source gemini --stage ingest`.

## Preview

- Use `polylogue run --preview` to preview counts without writing.

## Search defaults

- Interactive runs open a picker when multiple results are returned, then open the selection.
- Omitting the query opens the latest render in interactive mode and prints the path in plain mode.
- Use `--list` to force the full list output (no picker or auto-open).
- Use `--open` to open the newest render without searching.
- Use `--verbose` to include snippets in list output.

## Index rebuild

- Search automatically rebuilds the index if it is missing.

## Path debugging

- Use `polylogue run --preview` to confirm resolved sources and output paths.
- Use `POLYLOGUE_RENDER_ROOT` to override render output without editing config.

## Health checks

- `polylogue check` verifies database integrity, FTS index, and render files.
- `polylogue check --repair` fixes issues that can be auto-fixed.
- `polylogue check --vacuum` compacts the database and reclaims space.
