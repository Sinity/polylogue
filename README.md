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
- **Render local logs:** Choose a file or directory; skim previews candidates, rich shows progress, and outputs land in the configured render directory (defaults to something like `~/polylogue-data/render`). Add `--html` for themed previews or `--diff` to see deltas when re-rendering.
- **Sync provider archives:** `polylogue sync drive|codex|claude-code` unifies Drive pulls and local IDE session mirroring with consistent flags for collapse thresholds, HTML, pruning, diffs, and JSON output. Defaults from `polylogue.config` keep outputs in locations such as `~/polylogue-data/sync-drive`, `~/polylogue-data/codex`, and `~/polylogue-data/claude-code`.
- **Import provider exports:** `polylogue import chatgpt|claude|codex|claude-code …` normalises exports into Markdown/HTML, reusing provider metadata and letting you cherry-pick conversations interactively when desired.
- **Inspect archives:** `polylogue inspect branches` renders branch trees (and auto-writes HTML explorers), `polylogue inspect search` queries the SQLite FTS index with rich filters, and `polylogue inspect stats` summarises tokens/attachments per provider.
- **Watch local sessions in real time:** `polylogue watch codex` and `polylogue watch claude-code` keep those directories synced automatically; adjust debounce, HTML, collapse settings per watcher, or run a single pass with `--once`. Every sync is logged to `polylogue status --json` so scheduled runs and watchers share the same telemetry.
- **Tweak session defaults on the fly:** The interactive menu's `Settings` pane toggles HTML previews, switches light/dark themes, and resets to config defaults. Adjustments flow through render, sync, import, and watch commands for the duration of the session.
- **Doctor & Stats:** `polylogue doctor` sanity-checks source directories; `polylogue stats` aggregates attachment sizes, token counts, and provider summaries (with `--since/--until` filters).
- **View recent runs:** The status dashboard shows the last operations, including attachment MiB and diff counts per command.
- **Monitor automation:** `polylogue status --json --watch` now streams provider-level stats for dashboards or terminal monitoring.
- **Branch-aware transcripts:** Canonical Markdown now lives at `<slug>/conversation.md`, with `<slug>/conversation.common.md` capturing shared context and `branches/<branch-id>/{<branch-id>.md, overlay.md}` preserving every alternate path.
- **Explore branch graphs:** `polylogue inspect branches` renders a skim-driven branch picker, prints the tree view, and auto-writes an HTML explorer when branches diverge (override output with `--html-out`, disable via `--html off`).
- **Search transcripts:** `polylogue inspect search` queries the SQLite FTS index with filters for provider, model, date range, and attachment metadata; add `--no-picker` to skip the skim preview or `--json` for automation.
- **Prune legacy outputs:** `polylogue prune` cleans up flat `<slug>.md` files and `_attachments/` folders left behind by older releases, keeping only the canonical conversation directories.
- **SQLite/Qdrant indexing:** Every successful write updates `XDG_STATE_HOME/polylogue/polylogue.db` (and, optionally, a Qdrant collection) so downstream tooling can query or sync metadata without reparsing Markdown.

## Provider Cheat Sheet

### ChatGPT Exports
- Export a ZIP from chat.openai.com → Settings → Data Controls → Export.
- Render with `polylogue import chatgpt EXPORT.zip --all --out ~/polylogue-data/chatgpt --html`.
- Metadata includes `sourcePlatform: chatgpt`, `conversationId`, `sourceExportPath`, and detected `sourceModel`.
- Attachments land in `<chat>/attachments/`; oversized tool outputs are truncated inline with full payloads saved beside the Markdown.

### Claude.ai Bundles
- Export a bundle from claude.ai, then run `polylogue import claude EXPORT.zip --out ~/polylogue-data/claude --html`.
- Tool use/result blocks are rendered as paired call/result sections; attachments copy from the bundle’s `attachments/` directory.
- Front matter records `sourcePlatform: claude.ai`, `conversationId`, `sourceModel`, and `sourceExportPath`.

### Claude Code Sessions
- Local IDE logs live under `~/.claude/projects/`. Use `polylogue sync claude-code --out ~/polylogue-data/claude-code --html --diff` for continuous mirroring or `polylogue import claude-code [SESSION_ID]` for one-offs.
- Each Markdown file captures summaries, tool invocations, shell transcripts, and provenance fields (`sourceSessionPath`, `sourceWorkspace`).

### OpenAI Codex CLI
- Session JSONL files live under `~/.codex/sessions/`. `polylogue sync codex --out ~/polylogue-data/codex --html --diff` keeps them current.
- Tool call/output pairs are merged, oversized logs spill into `attachments/`, and metadata records the absolute source path.

## Automation & Flags
Although the CLI is interactive by default, the same functionality is available non-interactively:
- `python3 polylogue.py render PATH [--out DIR] [--links-only] [--dry-run] [--force] [--collapse-threshold N] [--json] [--plain] [--html [on|off|auto]] [--diff]`
- `python3 polylogue.py sync drive|codex|claude-code [--out DIR] [--links-only] [--dry-run] [--force] [--prune] [--collapse-threshold N] [--html [on|off|auto]] [--diff] [--json]`
  - Drive extras: `--folder-name`, `--folder-id`, `--since`, `--until`, `--name-filter`, `--list-only`
  - Local extras: `--base-dir`, `--all`
- `python3 polylogue.py import chatgpt|claude|codex|claude-code SOURCE … [--out DIR] [--collapse-threshold N] [--html [on|off|auto]] [--force] [--all] [--conversation-id ID ...] [--base-dir DIR] [--json]`
- `python3 polylogue.py inspect branches [--provider NAME] [--slug SLUG] [--conversation-id ID] [--branch BRANCH_ID] [--diff] [--html [on|off|auto]] [--html-out PATH] [--theme light|dark] [--no-picker]`
- `python3 polylogue.py inspect search QUERY [--limit N] [--provider NAME] [--slug SLUG] [--conversation-id ID] [--branch BRANCH_ID] [--model MODEL] [--since RFC3339] [--until RFC3339] [--with-attachments|--without-attachments] [--no-picker] [--json]`
- `python3 polylogue.py inspect stats [--dir DIR] [--since DATE] [--until DATE] [--json]`
- `python3 polylogue.py watch codex|claude-code [--base-dir DIR] [--out DIR] [--collapse-threshold N] [--html [on|off|auto]] [--debounce seconds] [--once]`
- `python3 polylogue.py status [--json] [--watch] [--interval seconds]`
- `python3 polylogue.py prune [--dir DIR] [--dry-run]`
- `python3 polylogue.py doctor [--codex-dir DIR] [--claude-code-dir DIR] [--limit N] [--json]`

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
- Markdown keeps attachments in per-chat `attachments/` folders when downloads are enabled.
- Responses are folded at 25 lines by default (configurable via flag or interactive setting per run).
- Summaries are shown both as rich panels and gum-formatted Markdown for easy copy/paste.
- Collapsible callouts stay as Markdown blockquotes; open the generated `.html` preview for interactive folding when terminal renderers (e.g., `glow`) don’t support it.
- Imported providers share the same Markdown pipeline, so chunk counts, token approximations, and attachment sniffing behave consistently across Gemini, ChatGPT, Claude, Claude Code, and Codex sources.

## Development Notes
- Code follows PEP 8 with type hints where practical.
- Run `pytest` for the automated test suite covering importers, sync flows, and HTML transforms.
- Credentials (`credentials.json`, `token.json`) stay out of version control.
