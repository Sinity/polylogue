# Contributing

## Development Environment

Work inside the project devshell.

```bash
cd path/to/polylogue
direnv allow   # one-time setup; afterward the devshell loads automatically on cd
```

If you are not using `direnv`, enter the same environment manually:

```bash
nix develop
```

All commands below assume you are already inside that environment. If not, use
`nix develop -c <command>`.

The devshell regenerates `AGENTS.md` from [CLAUDE.md](CLAUDE.md) on entry.
It is gitignored.

For repository maintenance, use `devtools`:

```bash
devtools --help
devtools status
devtools render-all
```

## Workflow

All code changes land through feature branches and squash-merged pull requests
targeting `master`.

1. Open an issue first when the work is non-trivial, spans multiple PRs,
   or introduces architectural decisions. Skip for self-contained fixes.
2. Create a branch from `origin/master`.
3. Work on the branch. Git hooks enforce format and lint on commit, and
   run `devtools verify --quick` on push.
4. Run `devtools verify` (full, with pytest) before creating the PR.
5. Open a pull request. The template has required sections — fill them
   all in. The PR title becomes the squash-merge subject on `master`.
6. CI must pass. Fix failures on the branch, do not merge with red CI.
7. Squash-merge the pull request into `master`.

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

## Commits

Use conventional commit subjects on branches:

- `feat:`
- `fix:`
- `refactor:`
- `perf:`
- `test:`
- `docs:`
- `chore:`

Branch commits can be iterative while you are working, but the published branch
should still tell one coherent story. Avoid noisy “final final” or context-free
messages that leave reviewers guessing.

The PR title becomes the squash-merge subject on `master` — write it as the
history line you want. Branch-local commits serve review; the PR title and body
serve history.

## Versioning and Releases

`pyproject.toml` records the last tagged release. Development builds are
identified by git metadata, and `polylogue --version` must include the commit
hash plus the dirty marker when applicable.

Routine PRs do not touch `version = "X.Y.Z"`. Change it only when this exact
slice is cutting the matching `vX.Y.Z` tag.

The release procedure is:

1. Update `pyproject.toml` to `X.Y.Z`.
2. Run:

```bash
devtools render-all
devtools render-all --check
ruff check polylogue tests devtools
pytest -q --ignore=tests/integration
nix flake check
```

3. Commit the version bump as its own small change, normally `chore: release X.Y.Z`.
4. Tag that exact commit as `vX.Y.Z`.

If this slice is not producing the matching tag, leave `pyproject.toml`
unchanged.

## Issues

Issues are optional. Use them when they improve planning or future traceability:

- larger features or refactors
- bug reports that need a repro or acceptance record
- architectural or research questions
- follow-up chains that will span more than one PR
- durable unresolved debt discovered during implementation or verification

Skip the issue when the change is self-contained and the PR itself is enough.

When you do open an issue:

- use the provided issue templates
- write in terms of outcome, constraints, and acceptance criteria
- prefer planning issues over retroactive bookkeeping
- convert anonymous debt into tracked debt:
  - expected-failure tests that represent real bugs
  - TODO comments that would otherwise persist beyond the current PR
  - warnings or degraded behavior accepted temporarily for scope reasons
  - follow-up work called out in PR text or scratch notes
- if a test or comment carries durable debt, reference the issue from that location when practical

## Pull Requests

Pull requests should:

- use a conventional title like `feat: add X` or `fix(cli): correct Y`
- treat that title as the final squash-merge commit subject on `master`
- explain the problem, solution, verification, and any remaining risk or follow-up
- link a related issue with `Ref #NNN` or `Closes #NNN` when one exists
- record the verification commands that were actually run
- update docs, config, and governance when behavior or workflow changes

Use `Ref #NNN` when the issue should stay open after merge, and `Closes #NNN`
when the merge should close it.

## Repository Settings

The repository should stay aligned with the workflow above:

- protect `master` against direct pushes
- require pull requests for normal changes
- require the `CI`, `Nix`, and `Pull Request Policy` checks before merge
- keep squash merge enabled and leave merge-commit and rebase-merge disabled
- enable automatic deletion of head branches after merge
- allow Update branch for stale PRs
- do not require an issue for every pull request

## Verification Baseline

The devshell installs git hooks automatically (`core.hooksPath .githooks`):

- **pre-commit**: `ruff format --check` + `ruff check` on staged files.
- **pre-push**: `devtools verify --quick` (format + lint + render-all --check).

Before creating a PR, run the full baseline:

```bash
devtools verify            # format + lint + render-all --check + pytest
```

Or manually:

```bash
ruff format --check polylogue/ tests/ devtools/
ruff check polylogue/ tests/ devtools/
devtools render-all --check
pytest -q --ignore=tests/integration
```

Add `devtools build-package` or `nix flake check` when touching packaging or
Nix expressions.

See [TESTING.md](TESTING.md) and [docs/devtools.md](docs/devtools.md) for
details.

## Widening Type-Check Coverage

CI runs `mypy --strict` over a subset of `polylogue/*`. The legacy debt
is tracked as a list of excluded files in `[tool.mypy] exclude` in
`pyproject.toml`. The list only shrinks.

To clean a module:

1. Pick a file from the exclude list.
2. Remove its entry and run `mypy polylogue/` — fix the reported errors.
3. Run `devtools verify` to confirm the gate still passes.
4. Commit the removed entry together with the fixes.

New files under `polylogue/` are checked by default. Anything added
that fails `mypy --strict` must be cleaned before merge, not added to
the exclude list.
