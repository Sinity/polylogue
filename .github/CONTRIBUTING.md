# Contributing to Polylogue

## Git Workflow

This project uses **merge commits with discipline** to preserve complete history while maintaining readability.

### Core Principles

1. **Branch structure is preserved** - We use merge commits (not squash/rebase) when merging PRs
2. **Clean commits are required** - Before merging, your feature branch must have clean, meaningful commits
3. **History is queryable** - Use `git log --first-parent` for clean view, full history available when needed

### Why This Approach?

**Benefits:**
- Complete traceability of how features were developed
- Can view history simply (`git lg`) or in detail (`git lga`)
- Can revert entire features by reverting merge commit
- Preserves information that squash/rebase destroys

**Requirements:**
- You must clean up your branch before merging
- No "WIP", "fix typo", "oops" commits in final PR
- Each commit should be atomic and meaningful

## Daily Workflow

### 1. Starting a New Feature

```bash
# Update main
git checkout main
git pull

# Create feature branch
git checkout -b feature/my-feature

# Or for AI agent branches:
git checkout -b claude/my-feature
```

### 2. Working on Your Branch

Make commits freely while developing:

```bash
git commit -m "WIP: trying approach X"
git commit -m "oops forgot file"
git commit -m "fix typo"
git commit -m "this works!"
```

**This is fine during development!** You'll clean it up later.

### 3. Cleaning Your Branch (IMPORTANT!)

Before opening a PR, **clean up your commits**:

```bash
# Interactive rebase from main
git rebase -i main
```

This opens an editor. Transform messy commits into clean ones:

**Before (messy):**
```
pick a1b2c3d WIP: trying approach X
pick d4e5f6g oops forgot file
pick h7i8j9k fix typo
pick l0m1n2o this works!
pick p3q4r5s add tests
pick t6u7v8w fix test
```

**After (clean):**
```
pick a1b2c3d feat: implement feature X core logic
squash d4e5f6g oops forgot file
squash h7i8j9k fix typo
squash l0m1n2o this works!
pick p3q4r5s test: add comprehensive tests for feature X
squash t6u7v8w fix test
```

**Result:** Two clean commits instead of six messy ones.

### 4. Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description>

[optional body]

[optional footer]
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code restructuring without behavior change
- `test:` - Adding/updating tests
- `docs:` - Documentation changes
- `chore:` - Maintenance tasks
- `perf:` - Performance improvements

**Examples:**
```
feat: add progress bars to sync operations

Implements Rich progress bars for Drive, Codex, and Claude Code
sync operations. Progress is suppressed in --json mode and plain mode.

Closes #123
```

### 5. Keeping Branch Updated

**Don't use `git merge main`!** This creates merge commits in your branch.

Instead, rebase:

```bash
# Get latest main
git fetch origin

# Rebase your branch on top of main
git rebase origin/main

# If there are conflicts, resolve them:
# 1. Fix conflicts in files
# 2. git add <resolved-files>
# 3. git rebase --continue
```

### 6. Opening a Pull Request

1. Push your cleaned branch:
   ```bash
   git push origin feature/my-feature
   ```

2. Open PR on GitHub

3. Fill out the PR template

4. Request review

### 7. After Approval

**Maintainer will merge with:**
```bash
git merge --no-ff feature/my-feature
```

This creates a merge commit preserving branch structure:

```
main:  A---B---C-------M
            \         /
feature:     D---E---F
```

Where D, E, F are your clean commits.

## Viewing History

### Clean View (Daily Use)

```bash
# Short log showing just merge points and direct commits
git lg

# With graph
git lgg
```

This hides feature branch details, looks like squashed commits.

### Detailed View (Investigation)

```bash
# Full history with all commits
git lga

# See commits in a specific feature
git log main..feature/my-feature
```

### Git Blame

```bash
# Simple blame (shows merge commits)
git blamef file.py

# Detailed blame (shows individual commits)
git blame file.py
```

## Common Scenarios

### Scenario: I Have 10 Messy Commits

**Clean them up:**

```bash
git rebase -i main

# In the editor, use:
# pick - keep commit
# squash - merge into previous commit
# fixup - like squash, but discard commit message
# reword - change commit message
# drop - delete commit
```

### Scenario: Main Has Advanced Since I Branched

**Rebase onto latest main:**

```bash
git fetch origin
git rebase origin/main
```

**If conflicts occur:**
```bash
# 1. Resolve conflicts in files
# 2. git add <files>
# 3. git rebase --continue

# Or abort if you want to start over:
git rebase --abort
```

### Scenario: I Already Pushed Messy Commits

**After cleaning with rebase, force push:**

```bash
git push --force-with-lease
```

**Warning:** Only do this on feature branches, never on main!

### Scenario: I Accidentally Merged Main Into My Branch

**Undo the merge:**

```bash
# Find the commit before the merge
git log --oneline

# Reset to it
git reset --hard <commit-before-merge>

# Rebase instead
git rebase main

# Force push to update remote
git push --force-with-lease
```

### Scenario: How Do I Test My Changes?

```bash
# Run tests
pytest

# Run specific test
pytest tests/test_my_feature.py -v

# Test CLI commands
uv run polylogue <command> --dry-run

# Enter dev shell first if needed
nix develop
```

## Git Configuration

We provide recommended Git config in `.gitconfig`. To use it:

```bash
# Include in your local repo config
git config --local include.path ../.gitconfig

# Or copy specific aliases to your global config
cat .gitconfig >> ~/.gitconfig
```

**Useful aliases from our config:**
- `git lg` - Clean, first-parent log
- `git lga` - Full history with graph
- `git tidy` - Interactive rebase from upstream
- `git blamef` - Blame with first-parent

## CI/Testing

All PRs must:
- Pass `pytest` (all tests)
- Have no linting errors
- Include tests for new functionality

## Questions?

- Check AGENTS.md for development setup
- Check README.md for user-facing documentation
- Open an issue for questions

## For AI Agents

If you're Claude/GPT/etc working on this codebase:

1. **Always clean commits before creating PR** - Use `git rebase -i` to squash WIP commits
2. **Follow conventional commits format** - All commits should be `type: description`
3. **Include Co-Authored-By trailer** - Add your contribution attribution
4. **Run tests** - `pytest` must pass before pushing

Example commit:
```
feat: add JSON error handling for import commands

Wrap import handlers with @safe_json_handler decorator to catch
JSONModeError exceptions and emit structured JSON in --json mode.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

See AGENTS.md for more development guidelines.
