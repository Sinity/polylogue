# Polylogue

AI chat export archiver — ingests Claude, ChatGPT, Codex, Gemini exports into a SQLite archive with full-text search, session analytics, and derived products.

## Working Rules

- New semantics go into the substrate or product layer first, then surfaces
  adapt.
- Archive writes are idempotent by content hash. User metadata (tags, summaries)
  is excluded from the hash — changing it does not trigger re-import.

## Agent Workflow

Rules for AI agents working on this repository. These override default
agent behavior.

### Issue-first for non-trivial work

Open an issue before starting work that is non-trivial, spans multiple
PRs, or introduces architectural decisions. The issue defines scope and
acceptance criteria. Reference it from the PR with `Ref #NNN` or
`Closes #NNN`.

### Verification before push

Run `devtools verify` before creating any PR. The default baseline runs the
static/generated gates plus pytest-testmon affected tests. Seed the testmon
database explicitly on a fresh checkout or after harness/dependency changes.
The git hooks enforce format and lint on commit and `devtools verify --quick`
on push, but the default baseline must pass before the PR is opened.

Do not treat CI as the first verification pass. Anticipate failures
locally.

### Inner-loop verification — use testmon, never blanket-run

The default verification path is `devtools verify`, which uses **pytest-testmon
affected-selection** ("pymon"): it runs only the tests whose dependency graph
touches your changed files, finishing in seconds-to-minutes. This is the normal
way to check a change. For a single target while iterating, use
`devtools test <file>` or `devtools test -k <expr>`.

Anti-pattern (do NOT do this): `devtools test tests/unit/<dir>` over whole
directories, or blanket `pytest tests/unit ...`. Running broad directories is
effectively the full suite — it takes well over an hour and burns the budget to
re-confirm tests your change never touched. A behavior-preserving refactor that
is mypy-green needs only its testmon-affected set, not the whole tree.

- mypy `--strict` is the primary net for type/identifier refactors; trust it.
- `devtools verify` (testmon) is the behavioral net for the affected slice.
- Reserve a full run (`devtools verify --all`) for harness/dependency changes or
  a final pre-PR diagnostic — not the inner loop.

If `devtools verify` reports failures in files your change did not touch and
that testmon did not select for your diff, classify them as pre-existing or
flaky (re-run the exact node) before assuming they are yours.

### PR body discipline

The PR template requires sections: Summary, Problem, Solution,
Verification. Fill all of them. The PR title becomes the squash-merge
commit subject on `master` — write it as the permanent history line.

Rules for the squash-merge subject:

- Under 72 characters.
- Conventional prefix (`feat:`, `fix:`, `refactor:`, `test:`, `chore:`,
  `perf:`, `docs:`).
- Describes what changed, not what was worked on.
- Accurate — do not claim alignment, unification, or convergence unless
  the code actually achieves it.

Rules for the squash-merge body (PR description):

- Problem: why the work was necessary.
- Solution: what was done, key modules and contracts touched.
- Verification: exact commands run, not "tests pass."

### Claim verification

Before writing a commit message or PR body that claims something is
aligned, unified, converged, or complete — verify the claim against the
code. Grep for duplicated logic, check both paths, read the diff. A
claim that doesn't match the code is worse than no claim.

### Issue and PR writing quality

Issues and PR bodies are durable artifacts. Write them for a reader who
has no conversation context — they should stand alone. Include file
paths, acceptance criteria, and design references where applicable.

## Cloud lane (Claude Code Web / Codex Cloud)

Polylogue is well-suited for cloud sandboxes — pure Python, no native deps
beyond pre-built wheels, all paths overridable via `POLYLOGUE_ARCHIVE_ROOT`.

Bootstrap is handled by `.claude/setup.sh` (installs `uv`, runs
`uv sync --extra dev --frozen`, prepares `/tmp/polylogue-archive`). Default
env vars come from `.claude/settings.json`
(`POLYLOGUE_ARCHIVE_ROOT=/tmp/polylogue-archive`, `POLYLOGUE_FORCE_PLAIN=1`,
`HYPOTHESIS_PROFILE=ci`).

Safe to run in cloud:

- `uv run pytest tests/unit -q`
- `uv run pytest tests/property -q`  (`HYPOTHESIS_PROFILE=ci` enforces budgets)
- `uv run ruff check polylogue tests`
- `uv run mypy polylogue`
- `uv run devtools verify` (slow; scope with `--changed-only` if available)
- `polylogued run --no-api` against `/tmp/polylogue-archive` (synthetic fixtures only)

Do NOT in cloud:

- Upload real `~/.claude/projects/` or `~/.codex/sessions/` archives. Fixtures
  only. Real corpus testing happens on the self-hosted runner.
- Run browser-capture flows — they need interactive cookies and are slated to
  move to the ethereal companion host.
- Point at `/realm/data/...` paths; the cloud sandbox has no access to that
  data lake.

Privacy: the data-handling tier is governed by your Anthropic/OpenAI plan;
cloud-agent sandbox content inherits that tier (Pro/Max consumer = training by
default unless opted out; Business/Enterprise = no training). See
`docs/cloud-agents.md` for the full checklist.

@CONTRIBUTING.md
@TESTING.md
@docs/architecture.md
@docs/internals.md
@docs/devtools.md
@docs/cloud-agents.md
