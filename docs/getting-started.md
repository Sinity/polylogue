[← Back to Docs](README.md)

# Getting Started

A 5-minute quickstart for Polylogue.

## Enter the source checkout

```bash
git clone https://github.com/Sinity/polylogue.git
cd polylogue
nix develop
```

Verify:

```bash
polylogue --version
# polylogue X.Y.Z (commit 0000000)
```

No packaged PyPI/Homebrew/container install path is documented as current until
release artifacts exist and have been smoke-tested. From the dev shell,
Polylogue follows XDG conventions and works without a mandatory config file.

## First search

Point Polylogue at your AI chat exports, then search:

```bash
polylogue "css refactor"
```

Polylogue is query-first: a quoted expression (as above) or one after the
`find` keyword is a full-text search. A bare *unquoted* word is treated as a
command name and prints a hint instead of searching (#1842), so quote
multi-word queries. Results are rendered in your pager.

## First list

List recent sessions with natural-language time filters:

```bash
polylogue --since yesterday read --all
polylogue --since "last week" --origin claude-code-session read --all
```

Limit and format:

```bash
polylogue --since 2026-01 --limit 5 read --all --format json
```

## First cost check

Review estimated API costs:

```bash
polylogue analyze --cost-outlook --plan claude-pro
```

Polylogue includes a 23-model pricing catalog covering Anthropic, OpenAI, and
Google models. Session-level and provider-wide rollups are available.

## Running the daemon

The daemon watches your chat directories and ingests new sessions in real
time:

```bash
polylogued run
```

It auto-discovers Claude Code sessions, Codex sessions, and configured watch
roots. Optionally exposes a local HTTP API and a browser-capture receiver.

Check daemon health:

```bash
polylogued status
```

## Key commands

| Command | Purpose |
|---------|---------|
| `polylogue <terms>` | Full-text search |
| `polylogue --since <when> read --all` | List matched sessions |
| `polylogue --origin <origin> analyze --count` | Count matched sessions |
| `polylogue analyze --by origin` | Grouped statistics |
| `polylogue analyze --cost-outlook --plan claude-pro` | Subscription/quota cost outlook |
| `polylogue --id <id> read` | Export one session |
| `polylogue --since yesterday read --all` | Batch export |
| `polylogued run` | Start the daemon |
| `polylogued status` | Daemon health check |
| `polylogue --id <id> read --view transcript` | Display full session |

## Optional terminal note widget (zsh)

To turn the last shell command into an editable capture candidate, add this
optional widget to `~/.zshrc`. It has no Polylogue runtime dependency; it only
prefills the command line, leaving you to edit and submit it normally.

```zsh
typeset -g POLYLOGUE_NOTE_LAST_STATUS=0
precmd_functions+=('_polylogue_note_remember_status')

_polylogue_note_remember_status() {
  POLYLOGUE_NOTE_LAST_STATUS=$?
}

_polylogue_note_prefill() {
  local last_command="$(fc -ln -1)"
  BUFFER="polylogue note --ref last \"${last_command} [exit ${POLYLOGUE_NOTE_LAST_STATUS}]\""
  zle end-of-line
}

zle -N _polylogue_note_prefill
bindkey '^Xn' _polylogue_note_prefill
```

Press `Ctrl-X`, then `n`, review the prefilled text, and press Enter to capture
it as a candidate for later judgment.

## Configuration

No config file is required. Sensible defaults cover everything.

Polylogue auto-discovers these directories:

```
~/.claude/projects/       Claude Code sessions
~/.codex/sessions/         Codex sessions
```

Custom watch roots can be given to the daemon with `--root`. For an explicit
one-time import request, keep the daemon running and use:

```bash
polylogue import /path/to/exports
```

## Next steps

- [Search reference](search.md) -- query grammar, filters, verbs, output formats
- [Daemon guide](daemon.md) -- `polylogued`, HTTP API, systemd integration
- [Insights](insights.md) -- session profiles, work events, phases, costs
- [Export](export.md) -- markdown, JSONL, and query-set reads
- [Schema](schema.md) -- database tables, FTS5, embeddings, versioning
- [Providers](providers/index.md) -- provider detection, parser locations
- [CLI Reference](cli-reference.md) -- generated command reference
- [Configuration](configuration.md) -- XDG paths and environment variables
