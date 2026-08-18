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

`AGENTS.md` is a symlink to [CLAUDE.md](CLAUDE.md); edit `CLAUDE.md`.

For repository maintenance, use `devtools`:

```bash
devtools --help
devtools status
devtools render all
```

### Audit tooling

The `audit` dependency group supplies `ast-grep` for local structural candidate generation:

```bash
uv sync --extra dev --group audit --frozen
uv run --group audit ast-grep --version
```

The project devshell also provides `scc` for code-size and complexity summaries and `codeql` for periodic security-focused analysis. These tools are advisory until a separately scoped audit policy adopts a specific rule or query.

Before choosing a module, read [Code Navigation](docs/code-navigation.md). It
maps common changes to the package that owns their meaning, the production
route that must be exercised, and the minimum verification expected. The
ordered placement decision remains in
[Architecture § Placement Rules](docs/architecture.md#placement-rules).

## Workflow

All code changes land through feature branches and squash-merged pull requests
targeting `master`.

1. Open an issue first when the work is non-trivial, spans multiple PRs,
   or introduces architectural decisions. Skip for self-contained fixes.
2. Create a branch from `origin/master`.
3. Work on the branch. Git hooks enforce format and lint on commit, and
   run `devtools verify --quick` on push.
4. Run `devtools verify` before creating the PR. It automatically builds or
   repairs native pytest-testmon state, then uses affected-test selection.
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

Polylogue uses two schema-evolution regimes.

Durable tiers (`source.db` and `user.db`) may use explicit additive
migrations. These migrations live under `polylogue/storage/sqlite/migrations/`,
advance `PRAGMA user_version` one version at a time, and require a verified
backup manifest for the affected tier before they run. Destructive durable-tier
changes require a copy-forward design and explicit operator consent; do not
hide them behind a routine migration.

Derived tiers (`index.db`, `embeddings.db`) are rebuildable products. They do
do not get ad hoc in-place migration chains. A PR that bumps their schema edits
the canonical DDL, declares the delta in `storage/sqlite/lifecycle.py`, and
provides either a clone-validated non-semantic fast-forward plan or a
**rebuild/blue-green plan** for semantic changes:

- which user-visible archive operation triggers rebuild/re-acquisition from
  source (e.g. `polylogue ops reset --index && polylogued run` for index-tier
  schema bumps),
- which downstream products (insights, blob store, FTS) are rebuilt
  automatically vs. needing explicit recomputation,
- the expected end-user impact (rebuild time, disk usage, anything
  that requires action beyond the reset).

During development, classify schema changes before editing: metadata-only,
index-only, additive-derived, additive-durable, or semantic-reparse-required.
Batch same-tier schema changes from ready Beads before triggering a live
rebuild. Do not repeatedly reset and re-ingest the active archive for isolated
index additions that can be grouped into one schema bump, and do not call a
full reingest necessary unless the changed semantics actually require replaying
source rows.

The policy lint (`devtools lab policy schema-versioning`) validates derived-tier
lifecycle declarations and clone-safe benign-DDL shapes while allowing numbered
durable-tier SQL migrations. It does not attempt to classify Python helpers by
their names; ad hoc open-path upgrades remain unsupported because the runtime
routes derived changes only through declared fast-forward plans or rebuilds.

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
- require the authoritative CI checks before merge
- keep squash merge enabled and leave merge-commit and rebase-merge disabled
- enable automatic deletion of head branches after merge
- allow Update branch for stale PRs
- do not require an issue for every pull request

## Git Hooks

The devshell installs git hooks automatically. It anchors `core.hooksPath` at
`.beads-hooks` in the absolute checkout selected through Git's shared common
directory, so linked worktrees cannot retain a branch-point relative hook path.
The Beads composite hooks chain the ordinary Polylogue gates below and then run
the matching `bd hooks run ...` integration.

- **pre-commit**: `ruff format --check` + `ruff check` on staged files.
  Also runs a worktree-escape detector (#1211): when committing from a
  linked worktree, the hook resolves the worktree root from its per-
  worktree git dir and aborts if the current working directory has
  drifted outside that root (a common failure mode when an agent
  `cd`s into the main checkout from inside a worktree). Set
  `POLYLOGUE_ALLOW_WORKTREE_ESCAPE=1` for legitimate cross-worktree
  commit flows.
- **prepare-commit-msg / post-checkout / post-merge**: Beads integration hooks
  when `.beads-hooks` is active.
- **pre-push**: `devtools verify --quick` (format, lint, mypy, generated
  surfaces, and fast manifest checks), followed by the Beads pre-push hook when
  `.beads-hooks` is active.

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
devtools verify            # static/generated gates + incremental complete-corpus pytest
devtools verify --quick    # format + lint + mypy + render all --check (skip tests)
devtools verify --lab      # explicit lab checks beyond the quick/default loop
```

The quick gate runs on `git push` via the active pre-push hook. It's a fast
check, not a substitute for the default baseline. The default command safely
repairs missing or invalid native testmon state and automatically builds a new
environment when collection semantics change.

`devtools verify` does not replay a prior verify result. It always runs the
static gates. With a valid native environment, it invokes pytest-testmon to
execute changed-dependency, never-recorded, and previously-failing tests, and
attests the remaining corpus from unchanged recorded greens — earning
release-baseline authority when coverage is complete and green. When it bootstraps a missing or invalid native environment, it runs the
complete correctness corpus. The pytest step covers unit, property, fuzz, and
integration tests while excluding the separately operated `tests/benchmarks`
performance surface. It uses
`--testmon-forceselect` for incremental selection, with one parallel `not
load_sensitive` lane and one serial `load_sensitive` lane over the same native
environment. `tui` is a category marker and remains parallel unless a test is
also explicitly `load_sensitive`. Use
There is no manual seed or repair command, and no separate full-corpus flag:
every plain run is scoped to the complete corpus, executing what changed and
attesting the rest from unchanged recorded greens.

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
