# Polylogue

Polylogue is an interactive-first toolkit for archiving AI/LLM conversations—rendering local exports to Markdown, syncing Drive sessions, and keeping local Codex/Claude Code transcripts tidy with diffs, stats, and diagnostics.

## Quick Start
- Enable the direnv-managed dev shell: `direnv allow` (uses `.envrc` to call `nix develop`).
- Prefer manual entry? `nix develop` installs Python plus gum, skim, rich, bat, glow, etc.
- Run `python3 polylogue.py` and pick an action from the gum menu (Render, Sync, Local Syncs, Doctor, Stats, etc.).
- When a directory has multiple JSON logs, the skim picker previews files with `bat`; press `Ctrl+G` for a live `glow` render before confirming.
- The first Drive action walks you through supplying a Google OAuth client JSON; tokens are cached next to `polylogue.py`.

## What You Can Do
- **Render local logs:** Choose a file or directory; skim previews candidates, rich shows progress, and outputs land in the configured render directory (default `/realm/data/chatlog/markdown/gemini-render`). Add `--html` for themed previews or `--diff` to see deltas when re-rendering.
- **Sync Drive folders:** Connect to the default Drive folder (`AI Studio`) and pull chats to Markdown, downloading attachments unless you opt to link only.
- **Sync Codex / Claude Code sessions:** Mirror local CLI transcripts from `~/.codex/sessions/` and `~/.config/claude/projects/` via `polylogue sync-codex` / `polylogue sync-claude-code`, with optional JSON summaries, pruning, diffs, and HTML previews.
- **Import exported providers:** Convert ChatGPT zips, Claude exports, Claude Code sessions, or Codex JSONLs via `polylogue import …` subcommands. Skim lets you cherry-pick conversations; `--all` batches them.
- **Doctor & Stats:** `polylogue doctor` sanity-checks source directories; `polylogue stats` aggregates attachment sizes, token counts, and provider summaries (with `--since/--until` filters).
- **View recent runs:** The status dashboard shows the last operations, including attachment MiB and diff counts per command.

## Provider Cheat Sheet

### ChatGPT Exports
- Export a ZIP from chat.openai.com → Settings → Data Controls → Export.
- Render with `polylogue import chatgpt EXPORT.zip --all --out /realm/data/chatlog/markdown/chatgpt --html`.
- Metadata includes `sourcePlatform: chatgpt`, `conversationId`, `sourceExportPath`, and detected `sourceModel`.
- Attachments land in `<chat>_attachments/`; oversized tool outputs are truncated inline with full payloads saved beside the Markdown.

### Claude.ai Bundles
- Export a bundle from claude.ai, then run `polylogue import claude EXPORT.zip --out /realm/data/chatlog/markdown/claude --html`.
- Tool use/result blocks are rendered as paired call/result sections; attachments copy from the bundle’s `attachments/` directory.
- Front matter records `sourcePlatform: claude.ai`, `conversationId`, `sourceModel`, and `sourceExportPath`.

### Claude Code Sessions
- Local IDE logs live under `~/.config/claude/projects/`. Use `polylogue sync-claude-code --out /realm/data/chatlog/markdown/claude-code --html --diff` for continuous mirroring or `polylogue import claude-code SESSION_ID` for one-offs.
- Each Markdown file captures summaries, tool invocations, shell transcripts, and provenance fields (`sourceSessionPath`, `sourceWorkspace`).

### OpenAI Codex CLI
- Session JSONL files live under `~/.codex/sessions/`. `polylogue sync-codex --out /realm/data/chatlog/markdown/codex --html --diff` keeps them current.
- Tool call/output pairs are merged, oversized logs spill into `_attachments/`, and metadata records the absolute source path.

## Automation & Flags
Although the CLI is interactive by default, the same functionality is available non-interactively:
- `python3 polylogue.py render PATH [--out DIR] [--links-only] [--dry-run] [--force] [--collapse-threshold N] [--json] [--plain] [--diff]`
- `python3 polylogue.py sync [--folder-name NAME] [--folder-id ID] [--out DIR] [--links-only] [--since RFC3339] [--until RFC3339] [--name-filter REGEX] [--dry-run] [--force] [--prune] [--json] [--plain] [--diff]`
- `python3 polylogue.py sync-codex [--base-dir DIR] [--out DIR] [--collapse-threshold N] [--html] [--html-theme THEME] [--force] [--prune] [--all] [--json] [--diff]`
- `python3 polylogue.py sync-claude-code [--base-dir DIR] [--out DIR] [--collapse-threshold N] [--html] [--html-theme THEME] [--force] [--prune] [--all] [--json] [--diff]`
- `python3 polylogue.py list [--folder-name NAME] [--folder-id ID] [--since RFC3339] [--until RFC3339] [--name-filter REGEX] [--json] [--plain]`
- `python3 polylogue.py status`
- `python3 polylogue.py import chatgpt EXPORT_PATH [--conversation-id ID ...] [--all] [--out DIR] [--collapse-threshold N] [--html] [--html-theme THEME] [--plain]`
- `python3 polylogue.py import claude EXPORT_PATH [--conversation-id ID ...] [--all] [--out DIR] [--collapse-threshold N] [--html] [--html-theme THEME] [--plain]`
- `python3 polylogue.py import claude-code SESSION_ID [--base-dir DIR] [--out DIR] [--collapse-threshold N] [--html] [--html-theme THEME] [--plain]`
- `python3 polylogue.py import codex SESSION_ID [--out DIR] [--base-dir DIR] [--collapse-threshold N] [--html] [--plain]`
- `python3 polylogue.py doctor [--codex-dir DIR] [--claude-code-dir DIR] [--limit N] [--json]`
- `python3 polylogue.py stats [--dir DIR] [--since DATE] [--until DATE] [--json]`

`--plain` disables gum/skim/Rich styling for CI or scripts; `--json` prints machine-readable summaries.

## Tooling & UX Stack
The dev shell equips Polylogue with:
- `gum` for confirmations, menus, and formatted summaries.
- `skim (sk)` for fuzzy selection with previews (`bat` and `glow`).
- `rich` for progress bars, tables, and styled output.
- `bat`, `delta`, `fd`, `ripgrep`, `glow` as supporting CLIs.

Everything falls back gracefully when `--plain` is specified or stdout isn’t a TTY.

## Configuration
- Polylogue looks for a config at `~/.polylogueconfig` or `$XDG_CONFIG_HOME/polylogue/config.json` (override via `$POLYLOGUE_CONFIG`).
- Copy `docs/polylogue.config.sample.jsonc` as a starting point to set default collapse thresholds, HTML preferences, and output directories once.
- `polylogue doctor` reports the expected locations when no config is detected.

## Credentials & Tokens
1. Create an OAuth client for a “Desktop app” in the Google Cloud Console (we link you directly from the prompt).
2. Drop the downloaded JSON into the project root when prompted; Polylogue copies it to `credentials.json` and launches the OAuth flow.
3. `token.json` is generated automatically and refreshed as needed. No manual edits required.

## Formatting
- Markdown keeps attachments in per-chat `_attachments` folders when downloads are enabled.
- Responses are folded at 25 lines by default (configurable via flag or interactive setting per run).
- Summaries are shown both as rich panels and gum-formatted Markdown for easy copy/paste.
- Collapsible callouts stay as Markdown blockquotes; open the generated `.html` preview for interactive folding when terminal renderers (e.g., `glow`) don’t support it.
- Imported providers share the same Markdown pipeline, so chunk counts, token approximations, and attachment sniffing behave consistently across Gemini, ChatGPT, Claude, Claude Code, and Codex sources.

## Development Notes
- Code follows PEP 8 with type hints where practical.
- No test suite yet; validate with `python3 polylogue.py render PATH --plain --dry-run` and `python3 polylogue.py sync --plain --dry-run` against sample inputs.
- Credentials (`credentials.json`, `token.json`) stay out of version control.
