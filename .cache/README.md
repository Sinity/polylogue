# Local Cache Layout

This directory is for disposable cache state that improves local development
speed but does not carry durable meaning.

Examples:

- pytest cache data
- Hypothesis example database
- Python bytecode via `PYTHONPYCACHEPREFIX`
- Ruff and mypy caches
- benchmark plugin scratch storage
- coverage working state

Treat `.cache/` as recreatable. If it becomes suspicious, delete it and rerun
the relevant command.

Roots such as `.venv/` and `.direnv/` stay at the repo top because their tools
expect those locations. Repo-chosen cache state belongs under `.cache/`.
