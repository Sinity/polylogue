# Repository Guidelines

## Project Structure & Module Organization
- `convert_gemini_chatlog.py`: Main CLI to convert Gemini (and similar) chat JSON into Markdown, optionally downloading linked Google Drive attachments.
- `gemini/`: Sample inputs and generated artifacts (text, media, PDFs). Useful for local testing.
- `format_log.jq`: jq filter that renders JSONL logs into readable Markdown.
- `flake.nix`, `shell.nix`: Nix dev shells providing Python + Google API deps.
- Other folders (`sinex/`, `sinex_output/`): auxiliary content and generated docs.

## Build, Test, and Development Commands
- Enter dev shell (recommended): `nix develop` (flakes) or `nix-shell`.
- Convert a JSON chat log (no downloads):
  `python3 convert_gemini_chatlog.py ai_sinity_analysis_gemini_2_5_flash.json --output-dir out --attachment-summary --no-download`
- Convert with Drive downloads (first run triggers OAuth):
  `python3 convert_gemini_chatlog.py INPUT.json --output-dir out --credentials credentials.json --download-dir attachments --force-download`
- Preview without writing files: `python3 convert_gemini_chatlog.py INPUT.json --dry-run -v`
- Format an agent JSONL log: `cat log.jsonl | jq -f format_log.jq -r > log.md`

## Coding Style & Naming Conventions
- Python: follow PEP 8, 4-space indentation, type hints where practical.
- Naming: modules and functions in `snake_case`; CLI flags use kebab-case (e.g., `--no-download`).
- Formatting/Linting: no enforced tool in repo; if used locally, keep diffs minimal.

## Testing Guidelines
- No formal unit tests yet. Validate changes by running the converter against a real file (e.g., `ai_sinity_analysis_gemini_2_5_flash.json`) with `--dry-run` and with downloads disabled/enabled.
- If adding tests, place them under `tests/` with `test_*.py` and use `pytest`.

## Commit & Pull Request Guidelines
- Commit messages: prefer Conventional Commits style (e.g., `feat: add --link-text-attachments flag`, `fix: robust token file handling`).
- Pull requests: include a short description, example command(s) showing the change in action, and any before/after Markdown snippets.
- Link related issues or notes; add screenshots only when the change affects Markdown rendering.

## Security & Configuration Tips
- Keep `credentials.json` and generated `token.json` out of version control. Store them locally and reference with `--credentials`.
- Google Drive: ensure the account has access to linked files; if downloads fail, try `--force-download` and verify tester permissions.

