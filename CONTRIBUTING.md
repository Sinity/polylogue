# Contributing

## Development Environment

Work inside the project devshell.

```bash
cd /realm/project/polylogue
direnv allow   # one-time setup; afterward the devshell loads automatically on cd
```

If you are not using `direnv`, enter the same environment manually:

```bash
nix develop
```

## Workflow

All code changes land through feature branches and squash-merged pull requests targeting `master`.

1. Create a branch from `origin/master`.
2. Push the branch and open a pull request.
3. Squash-merge the pull request into `master`.

Direct pushes to `master` should not be used for normal development.

Open an issue when it helps track larger work, design decisions, or follow-up items. For many self-contained changes, a pull request alone is enough.

The devshell regenerates `AGENTS.md` from [CLAUDE.md](CLAUDE.md)
automatically. `AGENTS.md` is intentionally gitignored and should be treated as
a generated local artifact.

## Branch Naming

Use:

`feature/<category>/<description>`

Allowed categories:

- `feat`
- `fix`
- `refactor`
- `perf`
- `test`
- `docs`
- `chore`

Examples:

- `feature/feat/mcp-query-exports`
- `feature/fix/parser-null-guard`
- `feature/refactor/storage-product-splits`

## Pull Requests

Pull requests should:

- use a conventional title like `feat: add X` or `fix(cli): correct Y`
- link a related issue with `Ref #NNN` or `Closes #NNN` when one exists
- record the verification commands that were actually run
- call out risks, migrations, or intentional follow-up work

## Repository Settings

The repository should stay aligned with the workflow above:

- protect `master` against direct pushes
- require pull requests for normal changes
- require the `CI`, `Nix`, and `Pull Request Policy` checks before merge
- keep squash merge enabled and leave merge-commit / rebase-merge disabled
- do not require an issue for every pull request; use issues for larger tracked work, decisions, and follow-up chains

## Verification Baseline

Once you are inside the devshell, for most code changes run:

```bash
pytest -q --ignore=tests/integration
```

Add narrower or broader checks as needed for the touched area:

```bash
ruff check polylogue tests devtools
nix build .#polylogue
nix flake check
```

See [TESTING.md](TESTING.md) for the baseline test matrix and protected test surfaces, and [docs/test-quality-workflows.md](docs/test-quality-workflows.md) for the generated validation-lane, mutation-campaign, and benchmark catalog.

The following files are generated from live sources and should be refreshed when their source surface changes:

- `AGENTS.md` from `CLAUDE.md` via `render-agents`
- `docs/cli-reference.md` from the live CLI help via `render-cli-reference`
- `docs/test-quality-workflows.md` from the live quality registries via `render-quality-reference`
