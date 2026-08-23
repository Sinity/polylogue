# Beads - AI-Native Issue Tracking

This repository uses **Beads** for durable task tracking. Live state is stored
in the shared Dolt database and published with `bd dolt push`; it is not
imported from or exported into feature branches.

## What is Beads?

Beads is issue tracking that lives in your repo, making it perfect for AI coding agents and developers who want their issues close to their code. No web UI required - everything works through the CLI and integrates seamlessly with git.

**Learn more:** [github.com/steveyegge/beads](https://github.com/steveyegge/beads)

## Quick Start

### Essential Commands

```bash
# Create new issues
bd create "Add user authentication"

# View all issues
bd list

# View issue details
bd show <issue-id>

# Update issue status
bd update <issue-id> --claim
bd update <issue-id> --status done

# Sync with Dolt remote
bd dolt push
```

### Working with Issues

Issues in Beads are:
- **Dolt-native**: Stored in a versioned database independently of code branches
- **AI-friendly**: CLI-first design works perfectly with AI coding agents
- **Branch-independent**: Checkout changes cannot downgrade task state
- **Explicitly replicated**: `bd dolt push` publishes durable task changes

## Why Beads?

✨ **AI-Native Design**
- Built specifically for AI-assisted development workflows
- CLI-first interface works seamlessly with AI coding agents
- No context switching to web UIs

🚀 **Developer Focused**
- Issues are discoverable from the project through `bd`
- Works offline, syncs when you push
- Fast, lightweight, and stays out of your way

🔧 **Dolt Replication**
- Task history is independent of code commits and pull requests
- Shared task state across linked Git worktrees
- Explicit remote push/pull

## Get Started with Beads

Try Beads in your own projects:

```bash
# Install Beads
curl -sSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash

# Initialize in your repo
bd init

# Create your first issue
bd create "Try out Beads"
```

## Learn More

- **Documentation**: [github.com/steveyegge/beads/docs](https://github.com/steveyegge/beads/tree/main/docs)
- **Quick Start Guide**: Run `bd quickstart`
- **Examples**: [github.com/steveyegge/beads/examples](https://github.com/steveyegge/beads/tree/main/examples)

---

*Beads: Issue tracking that moves at the speed of thought* ⚡
