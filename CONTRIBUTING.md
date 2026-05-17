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
4. Run `devtools verify` before creating the PR. The default pytest step uses
   pytest-testmon affected-test selection; run `devtools verify --seed-testmon`
   first if the dependency database is not seeded.
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

User-visible changes (new flags, renamed or removed commands, output changes,
breaking migrations, security fixes) get a one-line entry in the `Unreleased`
section of [CHANGELOG.md](CHANGELOG.md) as part of the PR. Refactors and
test-only changes are exempt.

See [docs/release.md](docs/release.md) for the cut-time checklist
(pre-flight checks, the `Unreleased → [X.Y.Z]` move, tagging, and post-cut
verification). The condensed procedure:

1. Update `pyproject.toml` to `X.Y.Z` and roll the `Unreleased` heading
   in `CHANGELOG.md` to `[X.Y.Z] — YYYY-MM-DD`.
2. Run:

```bash
devtools render-all
devtools render-all --check
ruff check polylogue tests devtools
pytest -q --ignore=tests/integration
nix flake check
```

3. Commit the version bump as its own small change, normally `chore: release X.Y.Z`.
4. Tag that exact commit as `vX.Y.Z` (signed and annotated).

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

## Git Hooks

The devshell installs git hooks automatically (`core.hooksPath .githooks`):

- **pre-commit**: `ruff format --check` + `ruff check` on staged files.
  Also runs a worktree-escape detector (#1211): when committing from a
  linked worktree, the hook resolves the worktree root from its per-
  worktree git dir and aborts if the current working directory has
  drifted outside that root (a common failure mode when an agent
  `cd`s into the main checkout from inside a worktree). Set
  `POLYLOGUE_ALLOW_WORKTREE_ESCAPE=1` for legitimate cross-worktree
  commit flows.
- **pre-push**: `devtools verify --quick` (format, lint, mypy, generated
  surfaces, and fast manifest checks).

The pre-push hook is an early failure gate. The PR baseline is the
`devtools verify` workflow below.

## Type Checking

CI runs `mypy --strict` on all files under `polylogue/`, `tests/`, and
`devtools/`. Configuration is in `pyproject.toml`:

```toml
[tool.mypy]
strict = true
files = ["polylogue", "tests/**/*.py", "devtools/**/*.py"]
```

There is no exclude list. All new files are checked by default. The mypy gate
runs as part of `devtools verify` and in CI.

## Verification Baseline

Before creating a PR, run the local baseline. CI runs the same checks, while
local pytest selection is accelerated by pytest-testmon.

```bash
devtools verify            # static/generated gates + pytest-testmon affected tests
devtools verify --seed-testmon --skip-slow  # seed/update affected-test DB
devtools verify --all      # explicit full non-integration pytest diagnostic
devtools verify --quick    # format + lint + mypy + render-all --check (skip tests)
devtools verify --lab      # explicit lab checks beyond the quick/default loop
```

The quick gate runs on `git push` via `.githooks/pre-push`. It's a fast check,
not a substitute for the default baseline. The default command fails fast when
`.testmondata` and `.cache/testmon/seed.json` are missing; do not rely on
silent full-suite fallback.

`devtools verify` does not replay a prior verify result. It always runs the
static gates and lets pytest-testmon decide affected tests from the current
source, dependency, and Python-version state. If pytest configuration,
dependency locks, or shared test infrastructure changed since the seed, the
default command automatically widens the pytest step to `--testmon-noselect`
and refreshes dependency data.

Add `devtools build-package` or `nix flake check` when touching packaging or
Nix expressions. See [TESTING.md](TESTING.md) and [docs/devtools.md](docs/devtools.md)
for details.

Verification Impact: every PR gets a `Polylogue Verification Impact` comment. It's a verification
impact report showing affected domains, required gates, and known gaps. Use it
to choose focused verification — run the gates that match touched files, state
in the PR why a suggested gate is only optional for that change.

## PR Body Discipline

The PR template requires four sections:

1. **Summary**: one paragraph describing what changed and why
2. **Problem**: evidence or observation that triggered the work. Not "user asked"
3. **Solution**: modules touched, non-obvious decisions, alternatives rejected
4. **Verification**: exact commands run and the output line that matters. Not
   "tests pass"

The PR title becomes the squash-merge subject on `master`. Rules:
- ≤72 characters, conventional prefix, imperative mood
- Describes what changed, not what was worked on
- Accurate — do not claim alignment or unification unless the diff achieves it
