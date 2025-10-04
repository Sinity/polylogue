# Gemini Markdown (gmd)

Interactive-first tools to render Gemini chat JSON to Markdown and mirror a Google Drive folder into local, annotated Markdown with linked attachments.

## Quick Start
- Enable the direnv-managed dev shell: `direnv allow` (uses `.envrc` to call `nix develop`).
- If you prefer to launch manually: `nix develop` (installs Python, gum, skim, rich, bat, glow, etc.).
- Run `python3 gmd.py` and pick an action from the gum menu (Render, Sync, Local Syncs, List, Recent Runs, Help).
- When a directory has multiple JSON logs, the skim picker previews files with `bat`; press `Ctrl+G` for a live `glow` render before confirming.
- The first Drive action guides you through supplying a Google OAuth client JSON and runs auth automatically. Tokens are cached next to `gmd.py`.

## What You Can Do
- **Render local logs:** Choose a file or directory; skim previews JSON candidates, rich shows progress, and outputs land in `./gmd_out` by default. Add `--html` to emit a themed HTML preview alongside the Markdown file.
- **Sync Drive folder:** Connect to the default Drive folder (`AI Studio`) and pull chats to `./gemini_synced`, downloading attachments unless you opt to link only.
- **List Drive chats:** Browse remote chats with skim (fuzzy search + previews) or emit JSON for automation.
- **Sync Codex / Claude Code sessions:** Mirror local CLI transcripts from `~/.codex/sessions/` and `~/.config/claude/projects/` into Markdown with `gmd sync-codex` and `gmd sync-claude-code`, including optional HTML previews, pruning, and skim-powered selection.
- **Import other providers:** Convert ChatGPT exports, Claude.zip bundles, Claude Code sessions, or Codex CLI logs into Markdown via `gmd import …` subcommands. Interactively pick conversations with skim or pass `--all`/`--conversation-id` for batch mode.
- **View recent runs:** Inspect the last few renders/syncs recorded in the runtime log.
- **Doctor & Stats:** Run `gmd doctor` to sanity-check local exports and `gmd stats` for attachment/token dashboards across an output directory.

## Provider Cheat Sheet

### ChatGPT Exports
- Export a ZIP from chat.openai.com → Settings → Data Controls → Export.
- Render with `gmd import chatgpt EXPORT.zip --all --out chatgpt_out --html`.
- Metadata includes `sourcePlatform: chatgpt`, `conversationId`, `sourceExportPath`, and detected `sourceModel`.
- Attachments inside the export land in `<chat>_attachments/`; large tool outputs are truncated inline with the full text saved alongside the Markdown.

### Claude.ai Bundles
- Export a bundle from claude.ai settings, then run `gmd import claude EXPORT.zip --out claude_out --html`.
- Tool uses/results are rendered as paired call/result blocks; attachments copy from the bundle’s `attachments/` directory.
- Front matter records `sourcePlatform: claude.ai`, `conversationId`, `sourceModel`, and `sourceExportPath`.

### Claude Code Sessions
- Local IDE logs live under `~/.config/claude/projects/`. Use `gmd sync-claude-code --out claude_code_synced --html --diff` for continuous mirroring or `gmd import claude-code SESSION_ID` for one-offs.
- Each Markdown file captures summaries, tool invocations, shell transcripts, and provenance fields (`sourceSessionPath`, `sourceWorkspace`).

### OpenAI Codex CLI
- Session JSONL files live under `~/.codex/sessions/`. `gmd sync-codex --out codex_synced --html --diff` keeps them current.
- Tool call/output pairs are combined, oversized logs spill into `_attachments/`, and metadata records the absolute source path.

## Automation & Flags
Although the CLI is interactive by default, the same functionality is available non-interactively:
- `python3 gmd.py render PATH [--out DIR] [--links-only] [--dry-run] [--force] [--collapse-threshold N] [--json] [--plain]`
- `python3 gmd.py sync [--folder-name NAME] [--folder-id ID] [--out DIR] [--links-only] [--since RFC3339] [--until RFC3339] [--name-filter REGEX] [--dry-run] [--force] [--prune] [--json] [--plain]`
- `python3 gmd.py sync-codex [--base-dir DIR] [--out DIR] [--collapse-threshold N] [--html] [--html-theme THEME] [--force] [--prune] [--all]`
- `python3 gmd.py sync-claude-code [--base-dir DIR] [--out DIR] [--collapse-threshold N] [--html] [--html-theme THEME] [--force] [--prune] [--all]`
- `python3 gmd.py list [--folder-name NAME] [--folder-id ID] [--since RFC3339] [--until RFC3339] [--name-filter REGEX] [--json] [--plain]`
- `python3 gmd.py status`
- `python3 gmd.py import chatgpt EXPORT_PATH [--conversation-id ID ...] [--all] [--out DIR] [--collapse-threshold N] [--html] [--html-theme THEME] [--plain]`
- `python3 gmd.py import claude EXPORT_PATH [--conversation-id ID ...] [--all] [--out DIR] [--collapse-threshold N] [--html] [--html-theme THEME] [--plain]`
- `python3 gmd.py import claude-code SESSION_ID [--base-dir DIR] [--out DIR] [--collapse-threshold N] [--html] [--html-theme THEME] [--plain]`
- `python3 gmd.py import codex SESSION_ID [--out DIR] [--base-dir DIR] [--collapse-threshold N] [--html] [--plain]`
- `python3 gmd.py doctor [--codex-dir DIR] [--claude-code-dir DIR] [--limit N] [--json]`
- `python3 gmd.py stats [--dir DIR] [--json]`

`--plain` disables gum/skim/Rich styling for CI or scripts; `--json` prints machine-readable summaries.

## Tooling & UX Stack
The dev shell equips gmd with:
- `gum` for confirmations, menus, and formatted summaries.
- `skim (sk)` for fuzzy selection with previews (`bat` and `glow`).
- `rich` for progress bars, tables, and styled output.
- `bat`, `delta`, `fd`, `ripgrep`, `glow` as supporting CLIs.

Everything falls back gracefully when `--plain` is specified or stdout isn’t a TTY.

## Credentials & Tokens
1. Create an OAuth client for a “Desktop app” in the Google Cloud Console (we link you directly from the prompt).
2. Drop the downloaded JSON into the project root when prompted; gmd copies it to `credentials.json` and launches the OAuth flow.
3. `token.json` is generated automatically and refreshed as needed. No manual edits required.

## Formatting
- Markdown keeps attachments in per-chat `_attachments` folders when downloads are enabled.
- Responses are folded at 25 lines by default (configurable via flag or interactive setting per run).
- Summaries are shown both as rich panels and gum-formatted Markdown for easy copy/paste.
- Collapsible callouts stay as Markdown blockquotes; open the generated `.html` preview for interactive folding when terminal renderers (e.g., `glow`) don’t support it.
- Imported providers share the same Markdown pipeline, so chunk counts, token approximations, and attachment sniffing behave consistently across Gemini, ChatGPT, Claude, Claude Code, and Codex sources.

## Development Notes
- Code follows PEP 8 with type hints where practical.
- No test suite yet; validate with `python3 gmd.py render PATH --plain --dry-run` and `python3 gmd.py sync --plain --dry-run` against sample inputs.
- Credentials (`credentials.json`, `token.json`) stay out of version control.
