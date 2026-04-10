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

The devshell regenerates `AGENTS.md` from [CLAUDE.md](CLAUDE.md)
automatically. `AGENTS.md` is gitignored and treated as a generated local
file.

For repo-maintenance tasks, prefer the unified `devtools` entrypoint:

```bash
python -m devtools --help
python -m devtools --list-commands
python -m devtools --list-commands --json
python -m devtools status
python -m devtools status --json
python -m devtools render-all
```

The JSON discovery/status forms are the best fit for agents and other
automation that need to inspect the devtools surface without scraping prose.

## Workflow

All code changes land through feature branches and squash-merged pull requests
targeting `master`.

1. Create a branch from `origin/master`.
2. Push the branch and open a pull request.
3. Squash-merge the pull request into `master`.

Direct pushes to `master` should not be used for normal development.

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

Squash-merge behavior matters here:

- the pull-request title becomes the final commit subject on `master`
- write that title as the history line you want to keep
- keep branch-local commits useful for review, but optimize the PR title and PR
  body for the final preserved narrative

## Versioning and Releases

The package version in `pyproject.toml` is distribution metadata, not a
per-merge ritual.

- do not bump it for every routine PR
- bump it when cutting a real tagged release or publishing a real distributable change
- until a formal release program exists, prefer accurate docs and clean `master`
  history over synthetic version churn

## Issues

Issues are optional. Use them when they improve planning or future traceability:

- larger features or refactors
- bug reports that need a repro or acceptance record
- architectural or research questions
- follow-up chains that will span more than one PR

Skip the issue when the change is self-contained and the PR itself is enough.

When you do open an issue:

- use the provided issue templates
- write in terms of outcome, constraints, and acceptance criteria
- prefer planning issues over retroactive bookkeeping
- keep the language concise and operational

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

The default shape for a good PR here is:

- one coherent implementation slice
- a title that can stand alone in `master` history
- a body that records summary, problem, solution, and verification
- no requirement for a linked issue unless that issue added planning value

## Repository Settings

The repository should stay aligned with the workflow above:

- protect `master` against direct pushes
- require pull requests for normal changes
- require the `CI`, `Nix`, and `Pull Request Policy` checks before merge
- keep squash merge enabled and leave merge-commit and rebase-merge disabled
- enable automatic deletion of head branches after merge
- allow “Update branch” when required checks need a refreshed base
- do not require an issue for every pull request

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

See [TESTING.md](TESTING.md) for the baseline test matrix and protected test
areas, and [docs/test-quality-workflows.md](docs/test-quality-workflows.md)
for the generated validation-lane, mutation-campaign, and benchmark catalog.

The following files are generated from live sources and should be refreshed when
their source surface changes:

- `AGENTS.md` from `CLAUDE.md`
- `docs/cli-reference.md` from the live CLI help
- `docs/test-quality-workflows.md` from the live quality registries
- `docs/README.md` and the generated documentation table in `README.md` from the shared docs registry

Refresh them together with:

```bash
python -m devtools render-all
python -m devtools render-all --check
```

See [docs/devtools.md](docs/devtools.md) for the `devtools` commands and the
local-state layout under [`.cache/README.md`](.cache/README.md) and
[`.local/README.md`](.local/README.md).
