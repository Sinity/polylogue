# Repository Guidelines

## Project Structure & Modules
- `polylogue.py`: CLI entrypoint for rendering AI chat exports and syncing provider archives.
- `polylogue/`: implementation modules (commands, importers, rendering, UI utilities).
- `nix/devshell.nix`: dev shell defining Python deps plus gum, skim, rich, bat, glow, etc.
- Provider walkthroughs and sample workflows live under `docs/` (see `docs/providers/`).

## Development Workflow
- Use `nix develop` to enter the environment with all required tools.
- Run `python3 polylogue.py --help` (or a specific subcommand) directly; every workflow is exposed via the CLI with skim/gum prompts only when needed.
- The first Drive action requests a Google OAuth client JSON and stores credentials/tokens under `$XDG_CONFIG_HOME/polylogue/`.
- Assume dependencies are always present: do **not** add graceful-degradation branches for missing CLI tools or libraries. Our NixOS devshell supplies gum, skim, rich, etc., so code should hard-require them.
- **Never add graceful-degradation fallbacks.** We run on NixOS and can guarantee every dependency; if a tool is missing it should be treated as a hard failure, not a best-effort path.

## Automation & Testing
- Non-interactive paths automatically drop into a plain UI when stdout/stderr aren’t TTYs. Set `POLYLOGUE_FORCE_PLAIN=1` when you need deterministic plain mode in CI, or pass `--interactive` to re-enable gum/skim prompts even without a TTY.
- Smoke test with `POLYLOGUE_FORCE_PLAIN=1 python3 polylogue.py render data --dry-run` and `POLYLOGUE_FORCE_PLAIN=1 python3 polylogue.py sync --dry-run`.
- Run `pytest` regularly; new tests should live under `tests/`.
- Use `polylogue env` (with `--json` when needed) to confirm resolved config/output paths while debugging CI or support issues.

## Style & Naming
- Python code follows PEP 8, 4-space indentation, snake_case identifiers.
- CLI flags are kebab-case (e.g., `--links-only`, `--dry-run`).
- Keep inline comments concise and purposeful.

## Credentials & Security
- Never commit the files created in `$XDG_CONFIG_HOME/polylogue/` (credentials, tokens).
- The CLI guides users through supplying an OAuth client; ensure documentation reflects the XDG storage path.
- Drive access errors should surface clear, actionable prompts.

## Git Workflow (Important!)

This project uses **merge commits with clean feature branches**. Before creating a PR:

### 1. Make Clean Commits From the Start

**Best practice for AI agents:** Make each commit meaningful and well-formatted:

```bash
# ✅ Good commits (keep these)
git commit -m "feat: add progress bars to sync operations"
git commit -m "test: add tests for progress tracking"
git commit -m "docs: update README with progress bars"

# ❌ Avoid messy commits like:
# - "WIP"
# - "fix typo"
# - "oops"
# - "trying something"
```

**If you need to clean up commits (non-interactive methods for agents):**

```bash
# Option A: Reset and re-commit cleanly
# Use when you have multiple messy commits
git reset --soft HEAD~3  # Undo last 3 commits, keep changes staged
git commit -m "feat: implement complete feature X

Detailed description of all changes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Option B: Amend the last commit
# Use when only the last commit needs fixing
git commit --amend -m "feat: corrected commit message"

# Option C: Use fixup commits (advanced)
# Create fixup commits, then auto-squash them
git commit --fixup=HEAD~2  # Marks as fixup of earlier commit
git rebase --autosquash main  # Auto-squashes fixups (non-interactive)
```

**For human reviewers only:**
```bash
git rebase -i main  # Opens editor - NOT usable by agents
```

### 2. Use Conventional Commits

Format: `type: description`

**Types:** `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`

**Examples:**
```
feat: add progress bars to sync operations
fix: resolve JSONModeError in import commands
test: add coverage for config validation
docs: update README with new --json-lines flag
```

### 3. Include Co-Authored-By Trailer

All AI-generated commits should include attribution:

```
feat: implement new feature

Detailed description of what was done.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### 4. Update Branch With Rebase, Not Merge

```bash
# ✅ Correct - rebase on main
git fetch origin
git rebase origin/main

# ❌ Wrong - creates merge commits in feature branch
git merge main
```

### 5. Viewing History

```bash
# Clean view (daily use)
git log --first-parent --oneline  # or: git lg

# Full view (investigation)
git log --graph --all --oneline   # or: git lga
```

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for detailed Git workflow.
