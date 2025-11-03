# Repository Guidelines

## Project Structure & Modules
- `polylogue.py`: interactive CLI entrypoint for rendering AI chat exports and syncing provider archives.
- `polylogue/`: implementation modules (commands, importers, rendering, UI utilities).
- `nix/devshell.nix`: dev shell defining Python deps plus gum, skim, rich, bat, glow, etc.
- Provider walkthroughs and sample workflows live under `docs/` (see `docs/providers/`).

## Development Workflow
- Use `nix develop` to enter the environment with all required tools.
- Launch `python3 polylogue.py` for the interactive menu (render, sync, list, stats, doctor, help).
- The first Drive action requests a Google OAuth client JSON and stores credentials/tokens under `$XDG_CONFIG_HOME/polylogue/`.

## Automation & Testing
- Non-interactive paths use `--plain` to disable gum/skim/Rich; combine with `--json` for machine-readable summaries.
- Smoke test with `python3 polylogue.py render data --plain --dry-run` and `python3 polylogue.py sync --plain --dry-run`.
- Run `pytest` regularly; new tests should live under `tests/`.

## Style & Naming
- Python code follows PEP 8, 4-space indentation, snake_case identifiers.
- CLI flags are kebab-case (e.g., `--links-only`, `--dry-run`).
- Keep inline comments concise and purposeful.

## Credentials & Security
- Never commit the files created in `$XDG_CONFIG_HOME/polylogue/` (credentials, tokens).
- The CLI guides users through supplying an OAuth client; ensure documentation reflects the XDG storage path.
- Drive access errors should surface clear, actionable prompts.
