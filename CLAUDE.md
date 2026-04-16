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

Run `devtools verify` (full, with pytest) before creating any PR. The
git hooks enforce format and lint on commit and `devtools verify --quick`
on push, but the full baseline must pass before the PR is opened.

Do not treat CI as the first verification pass. Anticipate failures
locally.

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

@CONTRIBUTING.md
@TESTING.md
@docs/architecture.md
@docs/internals.md
@docs/devtools.md
