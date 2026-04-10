# Local Cache Layout

This directory is for disposable cache state that improves local development
speed but does not carry durable meaning.

Examples:

- pytest cache data
- Hypothesis example database
- Ruff and mypy caches
- benchmark plugin scratch storage
- coverage working state

Treat `.cache/` as recreatable. If it becomes suspicious, delete it and rerun
the relevant command.

Tool-owned roots such as `.venv/`, `.direnv/`, and Nix `result*` symlinks stay
where their tools expect them. `.cache/` is for repo-chosen cache state.
