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
devtools render all
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

## Schema-Touching Changes

Polylogue has no in-place schema upgrade chain. A PR that bumps
`SCHEMA_VERSION` or otherwise changes the canonical SQLite shape is
not an in-place storage upgrade — it is a deletes-then-defines edit of `SCHEMA_DDL`.
The PR body must replace any upgrade-path section with a
**re-ingest plan**:

- which user-visible archive operation triggers rebuild/re-acquisition from
  source (e.g. `polylogue ops reset --index && polylogued run` for index-tier
  schema bumps),
- which downstream products (insights, blob store, FTS) are rebuilt
  automatically vs. needing explicit recomputation,
- the expected end-user impact (rebuild time, disk usage, anything
  that requires action beyond the reset).

There is no requirement to provide an in-place upgrade path, and PRs
that try to add one will be rejected on policy grounds (see
`docs/internals.md` § "Schema Versioning Model" and #1212).

## Versioning and Releases

`pyproject.toml` records the last tagged release. Development builds are
identified by git metadata, and `polylogue --version` must include the commit
hash plus the dirty marker when applicable.

Routine PRs do not touch `version = "X.Y.Z"` or `CHANGELOG.md`. Both are
maintained by [release-please](https://github.com/googleapis/release-please)
from conventional commit subjects on `master` — see
[docs/release.md](docs/release.md) for the full flow.

What this means in practice for normal PRs:

- Use conventional commit subjects (`feat:`, `fix:`, `perf:`, `refactor:`,
  `docs:`, `test:`, `chore:`, `build:`, `ci:`, `style:`). The PR title is
  what becomes the squash-merge subject on `master` and is what
  release-please reads.
- Do not edit `CHANGELOG.md` or `pyproject.toml` `version` directly. A
  user-visible change is described by its `feat:` / `fix:` / `perf:` subject
  and PR body; release-please rolls it into the changelog at release time.
- A breaking change uses `feat!:` / `fix!:` or a `BREAKING CHANGE:` footer
  in the commit body, exactly as conventional-commits specifies.

The release itself is one merge: release-please keeps an open
`chore(release): X.Y.Z` PR up to date on every push to `master`. Merging it
bumps the version, rolls `Unreleased → [X.Y.Z]`, and pushes the signed
`vX.Y.Z` tag. The downstream `release.yml` workflow handles PyPI + GHCR
publish from the tag. The manual procedure is retained in
[docs/release.md](docs/release.md#manual-fallback) as a fallback only.

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
- link a related issue with neutral wording such as `Ref #NNN` when one exists
- record the verification commands that were actually run
- update docs, config, and governance when behavior or workflow changes

Use `Ref #NNN` by default. Do not use GitHub resolver keywords in PR bodies
or comments unless the operator explicitly asks for that exact PR to change
that exact issue's GitHub state.

## Documentation Site Previews

The documentation site (`devtools render pages` → `.cache/site/`) is
published to GitHub Pages on every push to `master` via
`.github/workflows/pages.yml`.

PRs that touch docs, render pages helpers, or top-level Markdown files
trigger `.github/workflows/pages-preview.yml`, which rebuilds the site
and uploads it as a workflow artifact named
`docs-site-preview-pr-<NNN>`. Download the artifact from the PR's
Checks → Pages Preview run, extract, and serve locally:

```bash
unzip docs-site-preview-pr-*.zip -d /tmp/polylogue-docs
python -m http.server --directory /tmp/polylogue-docs 8000
```

Per-PR live preview URLs (`/pr/NNN/`) and versioned release trees
(`/vX.Y.Z/`, plus a `/latest/` alias) are deferred follow-ups under
#1307 — both require migrating `pages.yml` from the single-target
`actions/deploy-pages` flow to a branch-based deploy.

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
devtools verify --quick    # format + lint + mypy + render all --check (skip tests)
devtools verify --lab      # explicit lab checks beyond the quick/default loop
```

The quick gate runs on `git push` via `.githooks/pre-push`. It's a fast check,
not a substitute for the default baseline. The default command fails fast when
`.cache/testmon/testmondata` and `.cache/testmon/seed.json` are missing; do not
rely on silent full-suite fallback.

`devtools verify` does not replay a prior verify result. It always runs the
static gates and then invokes pytest-testmon for affected-test selection from
the current source, dependency, and Python-version state. The default pytest
step combines marker filters with `--testmon-forceselect` so scale-tier
deselection does not silently expand the run back to the whole suite; affected
testmon runs are single-process by default to avoid xdist collection skew.
Polylogue does not maintain a parallel changed-file router for helper/config
paths; use `devtools verify --seed-testmon` when you intentionally want to
refresh the dependency database and `devtools verify --all` for an explicit full
diagnostic.

Add `devtools release build-package` or `nix flake check` when touching packaging or
Nix expressions. See [TESTING.md](TESTING.md) and [docs/devtools.md](docs/devtools.md)
for details.

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
