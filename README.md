# Polylogue

Polylogue is an interactive-first toolkit for archiving AI/LLM conversations—rendering local exports to Markdown, syncing Drive sessions, and keeping local Codex/Claude Code transcripts tidy with diffs, stats, and diagnostics.

## Quick Start

## Quality Gates
- `pytest` is expected to stay green (`24 passed` in CI); add tests for new behaviours whenever possible.
- Newly generated Markdown is spot-checked with `polylogue stats --dir …` and quick greps (timestamps, footnotes, attachment counts) before shipping.
- Provider slugs are deterministic kebab-case so Git diffs stay tidy. Avoid manual renames unless conversation metadata truly changes.
- Run `python3 polylogue.py --plain import …`/`sync …` after importer changes to ensure real data still parses.

- Enable the direnv-managed dev shell: `direnv allow` (uses `.envrc` to call `nix develop`).
- Prefer manual entry? `nix develop` installs Python plus gum, skim, rich, bat, glow, etc.
- Run `python3 polylogue.py` and pick an action from the gum menu (Render, Sync, Local Syncs, Doctor, Stats, etc.).
- When a directory has multiple JSON logs, the skim picker previews files with `bat`; press `Ctrl+G` for a live `glow` render before confirming.
- The first Drive action walks you through supplying a Google OAuth client JSON; credentials and tokens are stored under `$XDG_CONFIG_HOME/polylogue/` (defaults to `~/.config/polylogue/`).

## What You Can Do
- **Render local logs:** Choose a file or directory; skim previews candidates, rich shows progress, and outputs land in the configured render directory (default `/realm/data/chatlog/markdown/gemini-render`, a legacy name that works for any provider JSON). Add `--html` for themed previews or `--diff` to see deltas when re-rendering.
- **Sync Drive folders:** Connect to the default Drive folder (`AI Studio`) and pull chats to Markdown in `/realm/data/chatlog/markdown/gemini-sync`, downloading attachments unless you opt to link only.
- **Sync Codex / Claude Code sessions:** Mirror local CLI transcripts from `~/.codex/sessions/` and `~/.claude/projects/` via `polylogue sync-codex` / `polylogue sync-claude-code`, with optional JSON summaries, pruning, diffs, and HTML previews. Outputs land in `/realm/data/chatlog/markdown/codex` and `/realm/data/chatlog/markdown/claude-code` by default.
- **Watch local sessions in real time:** `polylogue watch codex` and `polylogue watch claude-code` keep those directories synced automatically; adjust debounce, HTML, collapse settings per watcher, or run a single pass with `--once`. Every sync is logged to `polylogue status --json` so scheduled runs and watchers share the same telemetry.
- **Import exported providers:** Convert ChatGPT zips, Claude exports, Claude Code sessions, or Codex JSONLs via `polylogue import …` subcommands. Skim lets you cherry-pick conversations; `--all` batches them. Automation targets exist for Codex, Claude Code, Drive sync, render jobs, and ChatGPT imports (see `polylogue automation describe`). (A Gemini-specific importer is not bundled yet; use `render` if you have raw JSON.)
- **Doctor & Stats:** `polylogue doctor` sanity-checks source directories; `polylogue stats` aggregates attachment sizes, token counts, and provider summaries (with `--since/--until` filters).
- **View recent runs:** The status dashboard shows the last operations, including attachment MiB and diff counts per command.
- **Monitor automation:** `polylogue status --json --watch` now streams provider-level stats for dashboards or terminal monitoring.
- **Branch-aware transcripts:** Canonical Markdown still lives at `<slug>.md`, but every import now writes `<slug>/conversation.md`, `<slug>/conversation.common.md`, and `branches/<branch-id>/{<branch-id>.md, overlay.md}` so historical forks remain accessible.
- **SQLite/Qdrant indexing:** Every successful write updates `XDG_STATE_HOME/polylogue/polylogue.db` (and, optionally, a Qdrant collection) so downstream tooling can query or sync metadata without reparsing Markdown.

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
- Local IDE logs live under `~/.claude/projects/`. Use `polylogue sync-claude-code --out /realm/data/chatlog/markdown/claude-code --html --diff` for continuous mirroring or `polylogue import claude-code SESSION_ID` for one-offs.
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
- `python3 polylogue.py status [--json] [--watch] [--interval seconds]`
- `python3 polylogue.py import chatgpt EXPORT_PATH [--conversation-id ID ...] [--all] [--out DIR] [--collapse-threshold N] [--html] [--html-theme THEME] [--plain]`
- `python3 polylogue.py import claude EXPORT_PATH [--conversation-id ID ...] [--all] [--out DIR] [--collapse-threshold N] [--html] [--html-theme THEME] [--plain]`
- `python3 polylogue.py import claude-code SESSION_ID [--base-dir DIR] [--out DIR] [--collapse-threshold N] [--html] [--html-theme THEME] [--plain]`
- `python3 polylogue.py import codex SESSION_ID [--out DIR] [--base-dir DIR] [--collapse-threshold N] [--html] [--plain]`
- `python3 polylogue.py doctor [--codex-dir DIR] [--claude-code-dir DIR] [--limit N] [--json]`
- `python3 polylogue.py stats [--dir DIR] [--since DATE] [--until DATE] [--json]`

`--plain` disables gum/skim/Rich styling for CI or scripts; `--json` prints machine-readable summaries.
Use `--to-clipboard` on `render`/`import` commands to copy a single Markdown result directly to your system clipboard.

## Tooling & UX Stack
The dev shell equips Polylogue with:
- `gum` for confirmations, menus, and formatted summaries.
- `skim (sk)` for fuzzy selection with previews (`bat` and `glow`).
- `rich` for progress bars, tables, and styled output.
- `bat`, `delta`, `fd`, `ripgrep`, `glow` as supporting CLIs.

Everything falls back gracefully when `--plain` is specified or stdout isn’t a TTY.

See `docs/automation.md` for watcher usage, ready-made systemd/cron templates, and the `polylogue automation systemd|cron|describe` helper that prints the snippets or metadata for you.
Available automation targets: `codex`, `claude-code`, `drive-sync`, `gemini-render`, `chatgpt-import` (pass `--target <name>` for snippet generation or declarative modules). Per-target defaults (collapse thresholds, Drive folders, HTML flags) are baked into the metadata but can be overridden via CLI flags or the NixOS module.

## Nix Flake Usage
- This repository is a self-contained flake: add `inputs.polylogue.url = "github:yourname/polylogue";` (or your fork) to another flake and reference the packaged CLI as `inputs.polylogue.packages.${system}.polylogue` or `inputs.polylogue.apps.${system}.polylogue`.
- Import the NixOS module via `imports = [ inputs.polylogue.nixosModules.polylogue ];` to enable declarative automation. Example:
  ```nix
  services.polylogue = {
    enable = true;
    user = "polylogue";
    workingDir = "/var/lib/polylogue";
    targets.codex = {
      enable = true;
      outputDir = "/realm/data/chatlog/markdown/codex";
      timer.interval = "10m";
    };
    targets."claude-code".enable = true;
  };
  ```
- The module reads the same automation metadata as the CLI (see `polylogue automation describe --target codex`) and wires systemd timers around the packaged binary. Override `targets.<name>.extraArgs`, `collapseThreshold`, `html`, or set per-target `workingDir`/`outputDir` as needed.

## Configuration
- Polylogue reads configuration from `$POLYLOGUE_CONFIG`, `$XDG_CONFIG_HOME/polylogue/config.json`, or (legacy) `~/.polylogueconfig`.
- Copy `docs/polylogue.config.sample.jsonc` to `$XDG_CONFIG_HOME/polylogue/config.json` to customise collapse thresholds, HTML defaults, and per-provider output directories.
- Run state (`state.json`, `runs.json`) lives under `$XDG_STATE_HOME/polylogue/` so automation can tail recent activity.
- `polylogue doctor` reports the discovered paths when no config is detected.
- Drive behaviour can be tuned with environment variables such as `$POLYLOGUE_RETRIES`, `$POLYLOGUE_RETRY_BASE`, `$POLYLOGUE_AUTH_MODE`, and `$POLYLOGUE_TOKEN_PATH`.
- Indexing backends are controlled via `$POLYLOGUE_INDEX_BACKEND` (`sqlite`, `qdrant`, or `none`) alongside optional Qdrant knobs (`POLYLOGUE_QDRANT_URL`, `POLYLOGUE_QDRANT_API_KEY`, `POLYLOGUE_QDRANT_COLLECTION`).

## Credentials & Tokens
1. Create an OAuth client for a “Desktop app” in the Google Cloud Console (we link you directly from the prompt).
2. When prompted, point Polylogue at the downloaded JSON; it copies the file to `$XDG_CONFIG_HOME/polylogue/credentials.json` (created on demand) before launching the OAuth flow.
3. Tokens are written to `$XDG_CONFIG_HOME/polylogue/token.json` and refreshed automatically. Override locations with `$POLYLOGUE_TOKEN_PATH` if necessary.

## Formatting
- Markdown keeps attachments in per-chat `_attachments` folders when downloads are enabled.
- Responses are folded at 25 lines by default (configurable via flag or interactive setting per run).
- Summaries are shown both as rich panels and gum-formatted Markdown for easy copy/paste.
- Collapsible callouts stay as Markdown blockquotes; open the generated `.html` preview for interactive folding when terminal renderers (e.g., `glow`) don’t support it.
- Imported providers share the same Markdown pipeline, so chunk counts, token approximations, and attachment sniffing behave consistently across Gemini, ChatGPT, Claude, Claude Code, and Codex sources.

## Development Notes
- Code follows PEP 8 with type hints where practical.
- Run `pytest` for the automated test suite covering importers, sync flows, and HTML transforms.
- Credentials (`credentials.json`, `token.json`) stay out of version control.
