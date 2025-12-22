# CLI Tips

Polylogue’s CLI bundles a handful of ergonomic helpers that smooth out repetitive workflows. The Nix/devshell environment ships every required dependency (Rich, questionary, pyperclip, etc.), so these helpers are always available unless stdout/stderr aren’t TTYs (in which case Polylogue automatically switches to a plain UI—export `POLYLOGUE_FORCE_PLAIN=1` to force that behaviour in CI).
Plain/non-TTY runs never auto-prompt; if a prompt would be required (e.g., Drive auth, picker selection) the command exits with a warning so cron jobs don’t hang—pass explicit flags like `--all`/IDs or run with `--interactive` on a TTY when you do want prompts.

Interactive workflows require the external binaries `sk` (skim), `bat`, and `glow` to be available on `PATH`; Polylogue now treats them as hard dependencies rather than optional helpers. Image OCR is also enabled by default—install Tesseract plus the Pillow/pytesseract bindings (or use `polylogue[ocr]`) to make sure attachment text is indexed, and pass `--no-attachment-ocr` when you want to skip OCR for a run.

## Clipboard & Credential Workflows

- **Drive onboarding**: During the first Drive sync, Polylogue checks the system clipboard for an OAuth client JSON. If found (and confirmed), it saves the payload to `$XDG_CONFIG_HOME/polylogue/credentials.json`, avoiding manual copy/paste steps.
- **Auth path visibility**: `polylogue config show --json` and `polylogue doctor status --json` include credential/token paths plus env overrides (`POLYLOGUE_CREDENTIAL_PATH`/`POLYLOGUE_TOKEN_PATH`), so you can validate which files headless jobs will read. Drive OAuth requires a TTY; non-interactive shells fail fast instead of hanging on input.
- **Manual credential import**: When no clipboard payload exists, the CLI guides you through selecting a local file or opening Google’s setup guide. Credentials and tokens always land under `$XDG_CONFIG_HOME/polylogue/`.
- **Copy rendered Markdown**: Pass `--to-clipboard` to render/import commands. When exactly one Markdown file is produced, Polylogue copies it via `pyperclip` and reports success or a warning if the OS clipboard rejects the write.
- **Clipboard failures**: `pyperclip` ships with the devshell, but if the OS clipboard rejects writes Polylogue simply reports the warning while continuing the import.

## Branch Explorer & Search

- **Branch picker**: `polylogue browse branches` lists branch-aware conversations, opens a skim picker when `sk` is available, prints the tree, and writes a shareable HTML explorer automatically when forks exist (override with `--out`, disable via `--html off`). Inline prompts can queue a branch diff or on-demand HTML write/directory reveal without rerunning the command.
- **FTS search**: `polylogue search` queries the SQLite `messages_fts` index. Filters include provider, slug, conversation/branch IDs, model, date range (`--since/--until`), and attachment presence. Interactive previews default to skim; add `--no-picker` in CI or scripts, or `--json` when you need machine-readable results.

## Session Settings & Themes

- **Adjust defaults**: Run `polylogue config set --html on|off --theme light|dark` to change the HTML preview preference or theme for the current environment. Use `--reset` to restore config defaults.
- **Script defaults**: Non-interactive runs respect `polylogue.config`. Update the config file or call `polylogue config set --html on --theme dark` once so cron jobs pick up the same behaviour without extra flags.

## Discoverability & Completions

- **Quick help**: `polylogue help` prints the command overview; append a command name (e.g., `polylogue help sync`) to see detailed flags without digging into the docs.
- **Environment summary**: `polylogue config show` prints the resolved config path, output directories, and state database locations. Pass `--json` when you need the same information for scripts.
- **Shell completions**: Run `polylogue completions --shell bash|zsh|fish` and source the output. All shells (bash/fish/zsh) now use dynamic completions via `polylogue _complete`, so they can suggest known providers, branch slugs, recent Drive IDs, and local session files as you tab through arguments.
- **Status & Run History**: `polylogue doctor status` renders provider-level health (runs, attachments, failure counts) alongside the most recent sync/import runs. Pair it with `--json` to feed monitoring dashboards, use `polylogue browse runs --since 2024-01-01 --until 2024-02-01` when you need a focused history slice, and run `polylogue doctor index check --repair` if you ever need to rebuild SQLite/Qdrant indexes without running the full doctor.
- **Streaming status**: Combine `polylogue doctor status --watch --json-lines` with `jq`, `tee`, or log forwarders to collect newline-delimited JSON snapshots without scraping Rich tables.
