# Repository Guidelines

## Project Structure & Modules
- `gmd.py`: interactive CLI entrypoint for rendering local Gemini JSON and syncing Drive chats.
- `chatmd/`: implementation modules (commands, importers, rendering, UI utilities).
- `nix/devshell.nix`: dev shell defining Python deps plus gum, skim, rich, bat, glow, etc.
- `gemini/`, `sinex_md/`: sample inputs and generated Markdown artifacts.

## Development Workflow
- Use `nix develop` to enter the environment with all required tools.
- Launch `python3 gmd.py` for the interactive menu (render, sync, list, status, help).
- First Drive action will ask for a Google OAuth client JSON and perform auth automatically; token is cached locally.

## Automation & Testing
- Non-interactive paths use `--plain` to disable gum/skim/Rich; combine with `--json` for machine-readable summaries.
- Smoke test with `python3 gmd.py render data --plain --dry-run` and `python3 gmd.py sync --plain --dry-run`.
- If you add automated tests, place them under `tests/` and use `pytest`.

## Style & Naming
- Python code follows PEP 8, 4-space indentation, snake_case identifiers.
- CLI flags are kebab-case (e.g., `--links-only`, `--dry-run`).
- Keep inline comments concise and purposeful.

## Credentials & Security
- Never commit `credentials.json` or `token.json`.
- The CLI guides users through supplying an OAuth client; ensure documentation reflects this.
- Drive access errors should surface clear, actionable prompts.
