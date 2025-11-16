# Git Workflow Cheat Sheet

Quick reference for the merge-commits-with-discipline workflow.

## Philosophy

> Preserve complete history with merge commits, hide complexity with `--first-parent` flag.

**Benefits:**
- ✅ Complete traceability of feature development
- ✅ Clean view when you want it (`git lg`)
- ✅ Full details when you need them (`git lga`)
- ✅ Can revert entire features easily
- ✅ Preserves information that squash/rebase destroy

**Requirements:**
- ❗ Clean commits before merging (no WIP junk)
- ❗ Use conventional commit format
- ❗ Rebase branch, don't merge main into it

## Daily Commands

### Starting Work

```bash
# Update main
git checkout main
git pull

# Create feature branch
git checkout -b feature/my-feature
```

### While Developing

Make commits freely - WIP commits are fine:

```bash
git commit -m "WIP: trying something"
git commit -m "oops forgot file"
git commit -m "fix typo"
```

### Before Opening PR

**1. Clean up commits:**

```bash
git rebase -i main
```

In the editor, use:
- `pick` - keep commit as-is
- `squash` - merge into previous commit, keep both messages
- `fixup` - merge into previous, discard this message
- `reword` - change commit message
- `drop` - delete commit

**2. Push (force if you already pushed):**

```bash
# First push
git push origin feature/my-feature

# After cleaning (if already pushed)
git push --force-with-lease
```

### Keeping Branch Updated

**Don't merge main!** Use rebase:

```bash
git fetch origin
git rebase origin/main

# If conflicts:
# 1. Fix conflicts in files
# 2. git add <files>
# 3. git rebase --continue
```

### Viewing History

```bash
# Clean view (default for daily use)
git lg                      # Short, first-parent only
git lgg                     # With graph

# Full view (investigation)
git lga                     # All commits with graph
git log --graph --all

# Your PR commits
git review                  # What's in your PR
```

## Commit Message Format

```
<type>: <short description>

[Optional longer description explaining why, not what]

[Optional footers like issue references]

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Types

- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code restructuring, no behavior change
- `test:` - Add/update tests
- `docs:` - Documentation
- `chore:` - Maintenance, tooling
- `perf:` - Performance improvement

### Examples

```
feat: add progress bars to sync operations

Implements Rich progress bars for Drive and local provider sync.
Progress is automatically suppressed in --json mode and plain mode.

Closes #123
```

```
fix: resolve JSONModeError not being caught in import

Apply @safe_json_handler decorator to all import command handlers
to properly catch and emit JSON errors instead of tracebacks.

Addresses Codex review feedback.
```

## Common Scenarios

### I have 10 messy commits

```bash
git rebase -i main
# Squash/fixup the messy ones
git push --force-with-lease
```

### Main advanced since I branched

```bash
git fetch origin
git rebase origin/main
```

### I merged main into my branch by mistake

```bash
# Find commit before the merge
git log --oneline

# Reset to it
git reset --hard <commit-before-merge>

# Rebase instead
git rebase origin/main

# Force push
git push --force-with-lease
```

### View what changed in a PR

```bash
# On GitHub, or locally:
git diff main...feature/my-feature

# Just the commits
git log main..feature/my-feature
```

### Undo last commit (keep changes)

```bash
git reset HEAD~1
```

### Undo last commit (discard changes)

```bash
git reset --hard HEAD~1
```

## Git Aliases

Add these to `~/.gitconfig` or use the project's `.gitconfig`:

```ini
[alias]
    lg = log --first-parent --oneline --decorate
    lgg = log --first-parent --graph --oneline --decorate
    lga = log --graph --oneline --decorate --all
    review = log --first-parent --graph --oneline --decorate @{upstream}..HEAD
    tidy = rebase -i @{upstream}
    mergeff = merge --no-ff
```

## For Maintainers

### Merging a PR

```bash
# Checkout PR branch
git fetch origin
git checkout feature/contributor-branch

# Verify commits are clean
git log main..HEAD

# Switch to main and merge
git checkout main
git merge --no-ff feature/contributor-branch

# Push
git push origin main
```

**Note:** GitHub's "Merge pull request" button does this automatically when set to "Create a merge commit."

### Reverting a Feature

```bash
# Find the merge commit
git log --first-parent --oneline

# Revert it
git revert -m 1 <merge-commit-sha>
```

The `-m 1` tells Git to revert to the first parent (main), undoing the feature branch.

## Configuration

### Local Repository

```bash
# Include project config
git config --local include.path ../.gitconfig

# Verify
git config --list --local
```

### Global Configuration

Copy settings you like from `.gitconfig` to `~/.gitconfig`:

```bash
# For aliases
git config --global alias.lg "log --first-parent --oneline --decorate"

# For merge behavior
git config --global merge.ff false

# For better conflict markers
git config --global merge.conflictStyle zdiff3
```

## Quick Reference Card

| What                  | Command                        |
|-----------------------|--------------------------------|
| Clean view            | `git lg`                       |
| Full view             | `git lga`                      |
| Clean commits         | `git rebase -i main`           |
| Update branch         | `git rebase origin/main`       |
| Force push (safe)     | `git push --force-with-lease`  |
| View PR commits       | `git review`                   |
| Undo last commit      | `git reset HEAD~1`             |
| Merge with commit     | `git merge --no-ff <branch>`   |

## Further Reading

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Book - Rewriting History](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History)
- [Pro Git - Interactive Rebase](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History#_changing_multiple)
