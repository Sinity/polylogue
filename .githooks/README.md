# Git Hooks

This directory contains Git hooks to help enforce clean commit hygiene.

## Installation

Enable these hooks in your local repository:

```bash
git config core.hooksPath .githooks
```

## Available Hooks

### `commit-msg`

Validates commit message format and reminds you to use conventional commits.

**Checks:**

- Conventional commit format: `type: description`
- Valid types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`

**Note:** This is a soft reminder - commits will still proceed.

### `pre-push`

Checks for messy commits before pushing to remote.

**Checks for problematic commits:**

- WIP commits
- "oops", "typo", "debug" messages
- Temporary/test commits

**Also checks:**

- Non-conventional commit format

**Behavior:**

- Warns about messy commits
- Prompts to confirm push
- Can cancel push to clean up commits first

## Disabling Hooks

### Temporarily (single commit/push)

```bash
# Skip commit-msg hook
git commit --no-verify -m "message"

# Skip pre-push hook
git push --no-verify
```

### Permanently

```bash
# Disable hooks
git config --unset core.hooksPath

# Re-enable later
git config core.hooksPath .githooks
```

## Customization

Feel free to modify these hooks for your workflow:

```bash
# Edit hooks
$EDITOR .githooks/commit-msg
$EDITOR .githooks/pre-push

# Make sure they're executable
chmod +x .githooks/*
```

## Why These Hooks?

This project uses **merge commits with discipline**:

- We preserve branch structure with merge commits
- But we require clean commits before merging
- These hooks help catch messy commits early

See [CONTRIBUTING.md](../.github/CONTRIBUTING.md) for the full Git workflow.
