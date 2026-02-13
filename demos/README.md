# Demo Generation

Animated terminal screencasts for polylogue documentation and README.

## Quick Start

```bash
# Generate demos with synthetic data (no real exports needed)
./demos/generate.sh --synthetic

# Generate demos with real fixture data
./demos/generate.sh
```

## Prerequisites

- **VHS**: Terminal recorder from Charmbracelet. Available in the Nix devshell (`nix develop`), or install via `brew install vhs` / `go install github.com/charmbracelet/vhs@latest`.
- **Python environment**: `uv sync` or `nix develop` for polylogue dependencies.

## Using Real Fixtures

The demo system uses the same `tests/fixtures/real/` directory as the test suite. Symlink your actual provider exports there for richer demos:

```bash
# ChatGPT export
ln -s ~/Downloads/conversations.json tests/fixtures/real/chatgpt/

# Claude web export
ln -s ~/Downloads/claude-export.json tests/fixtures/real/claude/

# Claude Code sessions
ln -s ~/.claude/projects tests/fixtures/real/claude-code/

# Regenerate demos with real data
./demos/generate.sh
```

## Tape Files

| Tape | Shows | Output |
|------|-------|--------|
| `01-overview.tape` | `polylogue` (no args) → stats panels | `01-overview.gif` |
| `02-run.tape` | `polylogue run --preview` → plan, then run | `02-run.gif` |
| `03-search.tape` | Query mode with filters | `03-search.gif` |
| `04-dashboard.tape` | TUI dashboard browsing | `04-dashboard.gif` |
| `05-site.tape` | Static HTML site generation | `05-site.gif` |

## Workflow

```
generate.sh
  ├── 1. seeds demo database (scripts/seed_demo.py)
  ├── 2. exports env vars for isolation
  ├── 3. runs each .tape file via vhs
  └── 4. copies key GIFs to docs/assets/
```

## Re-recording a Single Tape

```bash
# Skip seeding, just re-record one tape
./demos/generate.sh --skip-seed --tape 03-search
```

## CI

The GitHub Actions workflow (`.github/workflows/demos.yml`) runs with `--synthetic` since CI doesn't have real exports. Updated GIFs are auto-committed.
