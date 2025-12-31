# CLI Tips

Polylogue switches to plain mode automatically when stdout/stderr are not TTYs. Use `--interactive` on a TTY to force prompts, or set `POLYLOGUE_FORCE_PLAIN=1` in CI.

## Drive auth

- Provide an OAuth client JSON at `~/.config/polylogue/credentials.json` or set `POLYLOGUE_CREDENTIAL_PATH`.
- Tokens are stored at `~/.config/polylogue/token.json` (or `POLYLOGUE_TOKEN_PATH`).
- Drive auth requires `--interactive` for the browser authorization code.

## Source scoping

- Use `--source NAME` (repeatable) on `run` to avoid reprocessing everything.
- Use `--source last` to reuse the previous interactive selection.
- Example: `polylogue run --source gemini --stage ingest`.

## Preview

- Use `polylogue run --preview` to preview counts without writing.

## Search defaults

- Interactive runs open a picker when multiple results are returned.
- Omitting the query prints the latest render path.
- Use `--list` to force the full list output (no picker).
- Use `--open` to open the newest render without searching.
- Use `--verbose` if you need snippets in the interactive view.

## Config editing

- Use `polylogue config edit` (interactive) to add or change sources.
- `config set` only supports `archive_root` now.

## Index rebuild

- Search automatically rebuilds the index if it is missing.

## Exports

- `polylogue export --out /path/to/export.jsonl` writes a machine-readable snapshot of conversations, messages, and attachments.

## Path debugging

- `polylogue config show` prints a summary; `polylogue config show --json` shows raw values and env overrides.
