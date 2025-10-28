# CLI Tips

Polylogue’s CLI bundles a handful of ergonomic helpers that smooth out repetitive workflows. These features are optional and fail gracefully when the environment lacks the required dependencies.

## Clipboard & Credential Workflows

- **Drive onboarding**: During the first Drive sync, Polylogue checks the system clipboard for an OAuth client JSON. If found (and confirmed), it saves the payload to `$XDG_CONFIG_HOME/polylogue/credentials.json`, avoiding manual copy/paste steps.
- **Manual credential import**: When no clipboard payload exists, the CLI guides you through selecting a local file or opening Google’s setup guide. Credentials and tokens always land under `$XDG_CONFIG_HOME/polylogue/`.
- **Copy rendered Markdown**: Pass `--to-clipboard` to render/import commands. When exactly one Markdown file is produced, Polylogue copies it via `pyperclip` and reports success or a warning if clipboard support is unavailable.
- **Graceful fallbacks**: Missing `pyperclip` or clipboard backends never block imports—they only suppress the convenience prompts.

## Branch Explorer & Search

- **Branch picker**: `polylogue inspect branches` lists branch-aware conversations, opens a skim picker when `sk` is available, prints the tree, and writes a shareable HTML explorer automatically when forks exist (override with `--html-out`, disable via `--html off`). Inline prompts can queue a branch diff or on-demand HTML write/directory reveal without rerunning the command.
- **FTS search**: `polylogue inspect search` queries the SQLite `messages_fts` index. Filters include provider, slug, conversation/branch IDs, model, date range (`--since/--until`), and attachment presence. Interactive previews default to skim; add `--no-picker` in CI or scripts, or `--json` for automation.
