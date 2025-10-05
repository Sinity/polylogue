# Repository Guidelines

## Project Structure & Modules
- `polylogue.py`: interactive CLI entrypoint for rendering AI chat exports and syncing provider archives.
- `polylogue/`: implementation modules (commands, importers, rendering, UI utilities).
- `nix/devshell.nix`: dev shell defining Python deps plus gum, skim, rich, bat, glow, etc.
- `gemini/`, `sinex_md/`: sample inputs and generated Markdown artifacts.

## Development Workflow
- Use `nix develop` to enter the environment with all required tools.
- Launch `python3 polylogue.py` for the interactive menu (render, sync, list, stats, doctor, help).
- First Drive action will ask for a Google OAuth client JSON and perform auth automatically; token is cached locally.

## Automation & Testing
- Non-interactive paths use `--plain` to disable gum/skim/Rich; combine with `--json` for machine-readable summaries.
- Smoke test with `python3 polylogue.py render data --plain --dry-run` and `python3 polylogue.py sync --plain --dry-run`.
- If you add automated tests, place them under `tests/` and use `pytest`.

## Style & Naming
- Python code follows PEP 8, 4-space indentation, snake_case identifiers.
- CLI flags are kebab-case (e.g., `--links-only`, `--dry-run`).
- Keep inline comments concise and purposeful.

## Credentials & Security
- Never commit `credentials.json` or `token.json`.
- The CLI guides users through supplying an OAuth client; ensure documentation reflects this.
- Drive access errors should surface clear, actionable prompts.
