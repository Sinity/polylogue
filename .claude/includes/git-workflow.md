## Git Workflow

### Branch Model

This project uses a **rewritten linear history** on `history/rewrite`. All work goes through feature branches that are **squash-merged** onto `history/rewrite` via GitHub PRs.

```
history/rewrite          ← squash target (linear history of arc commits)
  └── feature/X/Y       ← your working branch (individual commits preserved)
        ↓ squash-merge
history/rewrite          ← gains one clean commit
        ↓ fast-forward
master                   ← updated to match
```

**`master`** should always equal `history/rewrite` (or be a fast-forward of it). If `master` is ahead, those commits need to be on a feature branch first.

### Branch Naming

All branches use descriptive names: `feature/<category>/<description>`

| Category | Use |
|----------|-----|
| `feat` | New functionality |
| `refactor` | Structural changes |
| `perf` | Performance improvements |
| `bugfix` | Bug fixes |
| `testing` | Test infrastructure |
| `schema` | Schema changes |
| `docs` | Documentation |

Examples: `feature/perf/pipeline-quality-consolidation`, `feature/refactor/final-consolidation`

**Never** use opaque names like `arc65` or `wip-thing`.

### Creating a Feature Branch

```bash
# Always branch from history/rewrite
git fetch origin
git checkout -b feature/category/description origin/history/rewrite
```

### Committing

Use conventional commits: `type: description`

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`

Include co-author trailer for AI-generated commits:
```
Co-Authored-By: Claude <noreply@anthropic.com>
```

Individual commits on the feature branch don't need to be perfect — they'll be squash-merged. But they should still be meaningful (not "WIP" or "fix typo").

### Merging

1. Push feature branch to origin
2. Create PR targeting `history/rewrite` (not `master`)
3. Link the PR to its era issue via `Ref #NNN` in the body
4. **Squash-merge** the PR — GitHub creates one clean commit on `history/rewrite`
5. The squash commit message should be a rich narrative: what changed, why, and the scope

**Never merge directly to master.** Master is updated by fast-forwarding to `history/rewrite` after squash-merge.

### Era Issues

Each major body of work gets a GitHub issue titled `Era NN: Description`. PRs reference their era issue. This creates the chain: Era Issue → PR → Feature Branch (with individual commits) → Squash commit on `history/rewrite`.

### Remotes

- `origin` — GitHub (`git@github.com:Sinity/polylogue.git`)
- No other remotes. The old `source` self-remote was removed.

### History

The project has two histories sharing no common ancestor:
- `history/rewrite` — synthetic linear history (64+ squash commits), canonical
- `master-original` — original 1,110-commit history, kept for reference only

All new work builds on `history/rewrite`.
