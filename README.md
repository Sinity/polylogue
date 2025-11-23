# Polylogue

Polylogue is a CLI toolkit for archiving AI/LLM conversations—rendering local exports to Markdown, syncing Drive sessions, and keeping local Codex/Claude Code transcripts tidy with diffs, stats, and diagnostics.

## Quick Start

## Quality Gates
- `pytest` is expected to stay green; add tests for new behaviours whenever possible.
- Newly generated Markdown is spot-checked with `polylogue inspect stats --dir …` and quick greps (timestamps, footnotes, attachment counts) before shipping.
- Provider slugs are deterministic kebab-case so Git diffs stay tidy. Avoid manual renames unless conversation metadata truly changes.
- Run `python3 polylogue.py import …`/`sync …` after importer changes to ensure real data still parses (set `POLYLOGUE_FORCE_PLAIN=1` if you need deterministic plain output in CI).

- Enable the direnv-managed dev shell: `direnv allow` (uses `.envrc` to call `nix develop`).
- Prefer manual entry? `nix develop` installs Python plus gum, skim, rich, bat, glow, etc.
- Use `python3 polylogue.py <command>` (or `polylogue --help`) directly—every workflow is exposed as a CLI subcommand, and interactive pickers only appear when stdout/stderr are TTYs (pass `--interactive` to re-enable prompts in non-interactive shells). Run `polylogue help <command>` for focused per-command usage details.
- When a directory has multiple JSON logs, commands such as `polylogue render` use skim to preview files with `bat`; press `Ctrl+G` for a live `glow` render before confirming.
- The first Drive action walks you through supplying a Google OAuth client JSON; credentials and tokens are stored under `$XDG_CONFIG_HOME/polylogue/` (defaults to `~/.config/polylogue/`).

## What You Can Do
- **Render local logs:** Choose a file or directory; skim previews candidates, rich shows progress, and outputs land in the configured render directory (defaults to something like `~/polylogue-data/render`). Add `--html` for themed previews or `--diff` to see deltas when re-rendering.
- **Sync provider archives:** `polylogue sync drive|codex|claude-code|chatgpt|claude` unifies Drive pulls, local IDE sessions, and export bundles with consistent flags for collapse thresholds, HTML, pruning, diffs, and JSON output—now with Rich progress bars so you can see throughput as chats and sessions stream in. Defaults from `polylogue.config` keep outputs in locations such as `~/polylogue-data/sync-drive`, `~/polylogue-data/codex`, `~/polylogue-data/claude-code`, `~/polylogue-data/chatgpt`, and `~/polyologue-data/claude`. Drive runs accept explicit chat IDs via `--chat-id file-id --chat-id other-id`, while local providers can point at specific sessions via `--session path/to/session.jsonl` or skip pickers entirely with `--all`. ChatGPT/Claude exports are treated like local providers: drop ZIPs (or extracted directories containing `conversations.json`) under `$XDG_DATA_HOME/polylogue/exports/{chatgpt,claude}` and rerun `polylogue sync chatgpt` / `polylogue sync claude` as each export arrives.
- **Import provider exports:** `polylogue import chatgpt|claude|codex|claude-code …` normalises exports into Markdown/HTML, reusing provider metadata and letting you cherry-pick conversations interactively when desired.
- **Search transcripts:** `polylogue search` queries the SQLite FTS index with rich filters for provider, model, date range, and attachment metadata.
- **Browse archives:** `polylogue browse branches` renders branch trees (and auto-writes HTML explorers), and `polylogue browse stats` summarises tokens/attachments per provider.
- **Watch local sessions in real time:** `polylogue sync codex --watch` and `polylogue sync claude-code --watch` keep IDE logs mirrored automatically, and `polylogue sync chatgpt --watch` / `polylogue sync claude --watch` tail the `$XDG_DATA_HOME/polylogue/exports/{chatgpt,claude}` directories so every new ZIP or freshly extracted `conversations.json` triggers an incremental sync. Adjust debounce, HTML, and pruning per watcher, or run a single pass with `--once` when you just want to sweep directories without staying attached.
- **Doctor & Stats:** `polylogue maintain doctor` sanity-checks source directories, verifies SQLite/Qdrant indexes, and surfaces Drive retry/failure rates from recent runs; `polylogue maintain index` lets you re-run the index validation/repair cycle in isolation; `polylogue browse stats` aggregates attachment sizes, token counts, and provider summaries (with `--since/--until` filters). `polylogue browse status` renders a Rich overview of provider health and the latest runs for at-a-glance monitoring.
- **Settings:** `polylogue config set --html on --theme dark` updates the default render/sync preferences so scripted runs inherit the same HTML behaviour without extra flags.
- **View recent runs:** The status dashboard shows the last operations, including attachment MiB, diff counts, and Drive retry/failure stats per command.
- **Monitor status non-interactively:** `polylogue browse status --json --watch` streams provider-level stats for dashboards or terminal monitoring, and `polylogue browse status --dump <path> --dump-limit N` (optionally `--dump-only`) writes a JSON snapshot without reprinting the tables—perfect for cron/systemd hooks.
- **Observability exports:** Narrow status output with `--providers drive,codex`, stream summaries with `--watch`, emit newline-delimited snapshots via `polylogue browse status --watch --json-lines`, and dump machine-readable aggregates via `polylogue browse status --summary metrics.json` / `--summary-only`.
- **Inspect environment:** `polylogue config show` prints the resolved config/output directories plus the state/runs DB paths; pass `--json` when you need to feed the same information into scripts.
- **Run history:** Every render/sync/import/watch operation records a row in the SQLite database at `$XDG_STATE_HOME/polylogue/polylogue.db`. Use `polylogue browse runs --limit 20 --providers drive --since 2024-01-01 --json` for ad-hoc inspection (with `--until` to cap the window), or `polylogue browse status --dump runs.json` (pass `--dump -` for stdout) when you need a raw JSON export.
- **Branch-aware transcripts:** Canonical Markdown now lives at `<slug>/conversation.md`, with `<slug>/conversation.common.md` capturing shared context and `branches/<branch-id>/{<branch-id>.md, overlay.md}` preserving every alternate path.
- **Explore branch graphs:** `polylogue browse branches` renders a skim-driven branch picker, prints the tree view, and auto-writes an HTML explorer when branches diverge (override output with `--out`, disable via `--html off`).
- **Search transcripts:** `polylogue search` queries the SQLite FTS index with filters for provider, model, date range, and attachment metadata; add `--no-picker` to skip the skim preview or `--json` for CI.
- **Prune legacy outputs:** `polylogue maintain prune` cleans up flat `<slug>.md` files and `_attachments/` folders left behind by older releases, keeping only the canonical conversation directories.
- **SQLite/Qdrant indexing:** Every successful write updates `XDG_STATE_HOME/polylogue/polylogue.db` (and, optionally, a Qdrant collection) so downstream tooling can query or sync metadata without reparsing Markdown.

For deeper observability notes (status filters, JSON summaries), see `docs/observability.md`.

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

## Commands & Flags
Every workflow is available directly via the CLI:
- `python3 polylogue.py render PATH [--out DIR] [--links-only] [--dry-run] [--force] [--collapse-threshold N] [--json] [--html [on|off|auto]] [--diff]`
- `python3 polylogue.py sync drive|codex|claude-code|chatgpt|claude [--out DIR] [--links-only] [--dry-run] [--force] [--prune] [--collapse-threshold N] [--html [on|off|auto]] [--diff] [--json]` (note: `--diff` is only valid for drive/codex/claude-code).
  - Drive extras: `--folder-name`, `--folder-id`, `--since`, `--until`, `--name-filter`, `--list-only`
  - Local extras: `--base-dir`, `--all`, `--session PATH` (repeatable) to bypass pickers
- `python3 polylogue.py import chatgpt|claude|codex|claude-code SOURCE … [--out DIR] [--collapse-threshold N] [--html [on|off|auto]] [--dry-run] [--force] [--all] [--conversation-id ID ...] [--base-dir DIR] [--json]`
- `python3 polylogue.py search QUERY [--limit N] [--provider NAME] [--slug SLUG] [--conversation-id ID] [--branch BRANCH_ID] [--model MODEL] [--since RFC3339] [--until RFC3339] [--with-attachments|--without-attachments] [--no-picker] [--json] [--open]`
- `python3 polylogue.py browse branches [--provider NAME] [--slug SLUG] [--conversation-id ID] [--branch BRANCH_ID] [--diff] [--html [on|off|auto]] [--out PATH] [--theme light|dark] [--no-picker]`
- `python3 polylogue.py browse stats [--dir DIR] [--provider NAME] [--since DATE] [--until DATE] [--json]`
- `python3 polylogue.py browse status [--json] [--json-lines] [--watch] [--interval seconds] [--dump path] [--dump-only]`
- `python3 polylogue.py browse runs [--limit N] [--providers list] [--commands list] [--json]`
- `python3 polylogue.py maintain prune [--dir DIR] [--dry-run]`
- `python3 polylogue.py maintain doctor [--codex-dir DIR] [--claude-code-dir DIR] [--limit N] [--json]`
- `python3 polylogue.py maintain index [--dir DIR] [--repair]`
- `python3 polylogue.py config init`
- `python3 polylogue.py config set [--html on|off] [--theme light|dark] [--reset] [--json]`
- `python3 polylogue.py config show [--json]`
- `python3 polylogue.py help [COMMAND]`
- `python3 polylogue.py completions --shell bash|zsh|fish`

Plain mode is selected automatically whenever stdout/stderr aren't TTYs (or when `POLYLOGUE_FORCE_PLAIN=1` is set), and `--json` prints machine-readable summaries.
Use `--to-clipboard` on `render`/`import` commands to copy a single Markdown result directly to your system clipboard.

## Tooling & UX Stack
The dev shell equips Polylogue with:
- `gum` for confirmations, pickers, and formatted summaries.
- `skim (sk)` for fuzzy selection with previews (`bat` and `glow`).
- `rich` for progress bars, tables, and styled output.
- `bat`, `delta`, `fd`, `ripgrep`, `glow` as supporting CLIs.

Interactive features assume the gum/skim/Rich toolchain from the dev shell; for raw stdout or CI-friendly output rely on the automatic plain-mode detection or export `POLYLOGUE_FORCE_PLAIN=1`.

### Shell Completions
Generate a completion script with `polylogue completions --shell bash|zsh|fish` and source it from your shell profile. The zsh variant calls back into `polylogue _complete …` so it can surface live data (known providers, branch slugs, session paths, etc.) while you tab through arguments.


## Nix Flake Usage
- This repository is a self-contained flake: add `inputs.polylogue.url = "github:yourname/polylogue";` (or your fork) to another flake and reference the packaged CLI as `inputs.polylogue.packages.${system}.polylogue` or `inputs.polylogue.apps.${system}.polylogue`.

## Configuration
- Polylogue reads configuration from `$POLYLOGUE_CONFIG`, `$XDG_CONFIG_HOME/polylogue/config.json`, or (legacy) `~/.polylogueconfig`, and validates the contents at startup so typos (bad themes, non-numeric thresholds, mis-typed output paths) fail fast with actionable messages.
- Copy `docs/polylogue.config.sample.jsonc` to `$XDG_CONFIG_HOME/polylogue/config.json` to customise collapse thresholds, HTML defaults, and per-provider output directories.
- Run metadata and run history live in `$XDG_STATE_HOME/polylogue/polylogue.db` (SQLite); `polylogue status --dump` provides machine-readable exports for dashboards or scripts, and `polylogue env` prints the resolved config/output paths for quick inspection.
- `polylogue doctor` reports the discovered paths when no config is detected.
- Drive behaviour can be tuned with environment variables such as `$POLYLOGUE_RETRIES`, `$POLYLOGUE_RETRY_BASE`, `$POLYLOGUE_AUTH_MODE`, and `$POLYLOGUE_TOKEN_PATH`.
- Indexing backends are controlled via `$POLYLOGUE_INDEX_BACKEND` (`sqlite`, `qdrant`, or `none`) alongside optional Qdrant knobs (`POLYLOGUE_QDRANT_URL`, `POLYLOGUE_QDRANT_API_KEY`, `POLYLOGUE_QDRANT_COLLECTION`).

## Credentials & Tokens
1. Create an OAuth client for a “Desktop app” in the Google Cloud Console (we link you directly from the prompt).
2. When prompted, point Polylogue at the downloaded JSON; it copies the file to `$XDG_CONFIG_HOME/polylogue/credentials.json` (created on demand) before launching the OAuth flow.
3. Tokens are written to `$XDG_CONFIG_HOME/polylogue/token.json` and refreshed automatically. Override locations with `$POLYLOGUE_TOKEN_PATH` if necessary.

## Formatting
- Markdown keeps attachments in per-chat `attachments/` folders when downloads are enabled.
- Responses are folded at 25 lines by default (configurable via flag or `polylogue settings`).
- Summaries are shown both as rich panels and gum-formatted Markdown for easy copy/paste.
- Collapsible callouts stay as Markdown blockquotes; open the generated `.html` preview for interactive folding when terminal renderers (e.g., `glow`) don’t support it.
- Imported providers share the same Markdown pipeline, so chunk counts, token approximations, and attachment sniffing behave consistently across Gemini, ChatGPT, Claude, Claude Code, and Codex sources.

## Contributing

This project uses **merge commits with clean feature branches** to preserve complete history while maintaining readability.

**Quick start:**
1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make commits freely while developing (WIP commits are fine)
3. **Before PR:** Clean commits with `git rebase -i main`
4. Push and open PR

**See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for detailed Git workflow.**

**Key points:**
- Use conventional commit format: `feat:`, `fix:`, `test:`, `docs:`, etc.
- Clean up WIP commits before opening PR
- Update branch via `git rebase main`, not `git merge main`
- View history cleanly with `git log --first-parent` (or use `git lg` alias from `.gitconfig`)

## Development Notes
- Code follows PEP 8 with type hints where practical.
- Run `pytest` for the automated test suite covering importers, sync flows, and HTML transforms.
- Credentials (`credentials.json`, `token.json`) stay out of version control.
- Run history lives in the SQLite database at `$XDG_STATE_HOME/polylogue/polylogue.db`; use `polylogue status --dump runs.json` (or `--dump -` for stdout) when you need a JSON export.
- See [AGENTS.md](AGENTS.md) for AI agent development guidelines.
