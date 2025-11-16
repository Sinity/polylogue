# Git Workflow Setup Guide

Complete guide to setting up the merge-commits-with-discipline workflow.

## Quick Setup (1 minute)

```bash
# 1. Enable Git hooks
git config core.hooksPath .githooks

# 2. Include project Git config
git config --local include.path ../.gitconfig

# 3. Verify setup
git config --list --local | grep -E "(hooks|alias.lg)"
```

Done! You now have:
- ✅ Git aliases (`git lg`, `git lga`, `git review`, etc.)
- ✅ Hooks that warn about messy commits
- ✅ Better conflict markers (`zdiff3`)
- ✅ Automatic merge commit creation (`--no-ff`)

## What Was Configured?

### Files Created

```
polylogue/
├── .gitconfig              # Git aliases and settings
├── .githooks/              # Git hooks directory
│   ├── README.md           # Hooks documentation
│   ├── commit-msg          # Validates commit format
│   └── pre-push            # Warns about messy commits
├── .github/
│   ├── CONTRIBUTING.md     # Detailed Git workflow guide
│   ├── GIT_WORKFLOW.md     # Quick reference cheat sheet
│   ├── pull_request_template.md  # PR template
│   └── SETUP.md            # This file
├── AGENTS.md               # Updated with Git workflow
└── README.md               # Updated with Contributing section
```

### Git Aliases Added

| Alias       | Command                                       | Purpose                     |
|-------------|-----------------------------------------------|-----------------------------|
| `git lg`    | `log --first-parent --oneline --decorate`    | Clean history view          |
| `git lgg`   | `log --first-parent --graph ...`             | Clean history with graph    |
| `git lga`   | `log --graph --oneline --decorate --all`     | Full history                |
| `git review`| `log --first-parent ... @{upstream}..HEAD`   | View PR commits             |
| `git tidy`  | `rebase -i @{upstream}`                      | Clean up commits            |
| `git mergeff`| `merge --no-ff`                             | Force merge commit          |

### Git Settings Configured

```ini
[merge]
    ff = false                # Always create merge commit
    conflictStyle = zdiff3    # Better conflict markers

[pull]
    rebase = true             # Rebase when pulling

[rebase]
    autoStash = true          # Auto-stash when rebasing
    autoSquash = true         # Auto-squash fixup commits

[diff]
    algorithm = histogram     # Better diff algorithm
    colorMoved = default      # Show moved code differently

[fetch]
    prune = true              # Auto-prune deleted branches
```

## Testing Your Setup

### 1. Test Git Aliases

```bash
# Should show clean history
git lg

# Should show full history with graph
git lga

# Should work even though no upstream commits yet
git review
```

### 2. Test Commit Hook

```bash
# This should trigger a reminder
git commit --allow-empty -m "WIP testing"

# This should be clean
git commit --allow-empty -m "test: verify commit hook works"
```

### 3. Test Pre-Push Hook

```bash
# Create a WIP commit
git commit --allow-empty -m "WIP: test"

# Try to push (should warn)
git push origin HEAD

# Clean it up
git reset HEAD~1
```

## Daily Workflow

### Starting a Feature

```bash
git checkout main
git pull
git checkout -b feature/my-feature
```

### While Developing

Make commits freely:

```bash
git add .
git commit -m "WIP: trying something"
# ... more WIP commits ...
```

### Before Opening PR

Clean up commits:

```bash
# Interactive rebase
git tidy  # or: git rebase -i main

# In the editor, squash WIP commits
# Save and close

# Push (force if you already pushed)
git push origin feature/my-feature --force-with-lease
```

### Viewing History

```bash
# Clean view (what most people should see)
git lg

# Full details (when investigating)
git lga

# What's in my PR
git review
```

## Customization

### Global vs Local Config

The `.gitconfig` file contains **local** settings for this project.

**To use these settings globally:**

```bash
# Copy individual aliases you like
git config --global alias.lg "log --first-parent --oneline --decorate"

# Or copy entire alias section
cat .gitconfig | grep -A 20 "\[alias\]" >> ~/.gitconfig
```

**To keep project-local:**

```bash
# Already set with: git config --local include.path ../.gitconfig
# Settings only apply to this repo
```

### Modify Hooks

```bash
# Edit hooks
$EDITOR .githooks/commit-msg
$EDITOR .githooks/pre-push

# Make executable
chmod +x .githooks/*
```

### Disable Hooks Temporarily

```bash
# Skip commit-msg validation
git commit --no-verify -m "message"

# Skip pre-push check
git push --no-verify
```

## Troubleshooting

### Hooks Not Running

```bash
# Check hooks path
git config core.hooksPath
# Should show: .githooks

# If not set:
git config core.hooksPath .githooks

# Verify hooks are executable
ls -l .githooks/
# Should show: -rwxr-xr-x
```

### Aliases Not Working

```bash
# Check if config is included
git config --list --local | grep include.path
# Should show: include.path=../.gitconfig

# If not:
git config --local include.path ../.gitconfig

# Test alias
git lg
```

### Merge Commits Not Created

```bash
# Check merge.ff setting
git config merge.ff
# Should show: false

# If not:
git config merge.ff false
```

## Advanced: GitHub Settings

### Branch Protection Rules

Configure on GitHub for `main` branch:

**Settings → Branches → Branch protection rules → Add rule**

```yaml
Branch name pattern: main

✅ Require pull request reviews before merging
  - Required approving reviews: 1

✅ Require status checks to pass before merging
  - Require branches to be up to date before merging
  - Status checks: pytest

✅ Do not allow bypassing the above settings
```

### Default Merge Strategy

**Settings → General → Pull Requests**

```
✅ Allow merge commits
❌ Allow squash merging (disable)
❌ Allow rebase merging (disable)

Default to merge commit
```

This enforces merge commits at the repository level.

## Learning Resources

### Documentation

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Detailed Git workflow guide
- **[GIT_WORKFLOW.md](GIT_WORKFLOW.md)** - Quick reference cheat sheet
- **[.githooks/README.md](../.githooks/README.md)** - Hooks documentation

### External Resources

- [Conventional Commits](https://www.conventionalcommits.org/) - Commit message format
- [Git Interactive Rebase](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History) - Official Git book
- [Pro Git Book](https://git-scm.com/book/en/v2) - Complete Git reference

## Getting Help

**For Git workflow questions:**
- Check [CONTRIBUTING.md](CONTRIBUTING.md) first
- Check [GIT_WORKFLOW.md](GIT_WORKFLOW.md) for quick answers
- Open an issue if still stuck

**For Git config issues:**
- Run `git config --list --local` to see settings
- Check `.gitconfig` file directly
- Verify hooks with `ls -l .githooks/`

## Next Steps

1. ✅ Complete setup above
2. ✅ Read [CONTRIBUTING.md](CONTRIBUTING.md)
3. ✅ Try the workflow with a test branch
4. ✅ Open your first PR using clean commits

Happy coding! 🚀
