# 148. polylogue-th0 — Interactive-surface test harness: pty flows, completions, fuzzy pickers

Priority/type/status: **P2 / task / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

The suite (248k lines) is strong on units/properties/snapshots and blind on exactly the surfaces the UX program is now building: nothing drives a real pty, so fzf select flows, the coming judge TUI (p5g), bare-invocation triage (jnj.13), pager behavior, and terminal-width/color rendering are untested by construction; shell completions (fnm.4) have no correctness harness at all (a broken completion script fails silently forever); interactive-ambiguity moments (jnj.11) can regress without any red test. As the CLI grows TUI-ish flows, the untestable fraction of the product grows with it.

## Existing design note

(1) PTY harness in tests/infra: pexpect (or pty+os primitives — decide by trying pexpect's reliability under pytest-xdist) driving the real CLI binary against the seeded corpus: send keys, assert on screen state with normalized snapshots (strip timing/colors via the existing syrupy terminal-snapshot conventions; explicit width matrix 80/120/200 since fzf layouts shift). Keep the pty lane serial and marked (scale tier) — pty tests are inherently slower; a dozen golden flows, not hundreds. (2) COMPLETION CONTRACTS, no pty needed: invoke the completion entry points directly (Click's shell-complete protocol + the daemon completion endpoint once fnm.4 lands) with a table of (partial-input -> expected candidates) cases generated FROM the grammar registries — the registry is the oracle, so new units/stages get completion tests for free (declare-once payoff). (3) FZF flows: golden scripts per flow (select -> pick -> read; judge accept/reject; ambiguous-ref picker) with deterministic corpus ordering; assert side effects (what got opened/judged) not just screen pixels. (4) Wire as a devtools test lane + CI job (linux runner has pty; macos runner optional).

## Acceptance criteria

PTY harness runs 5+ golden flows green in CI serial lane; completion contract tests are registry-generated and fail when a unit is added without completion metadata; a deliberate fzf-flow regression (reordered candidates) is caught by the harness in a demonstration commit.

## Static mechanism / likely defect

Issue description localizes the mechanism: The suite (248k lines) is strong on units/properties/snapshots and blind on exactly the surfaces the UX program is now building: nothing drives a real pty, so fzf select flows, the coming judge TUI (p5g), bare-invocation triage (jnj.13), pager behavior, and terminal-width/color rendering are untested by construction; shell completions (fnm.4) have no correctness harness at all (a broken completion script fails silently forever); interactive-ambiguity moments (jnj.11) can regress without any red test. As the CLI gr… Design direction: (1) PTY harness in tests/infra: pexpect (or pty+os primitives — decide by trying pexpect's reliability under pytest-xdist) driving the real CLI binary against the seeded corpus: send keys, assert on screen state with normalized snapshots (strip timing/colors via the existing syrupy terminal-snapshot conventions; explicit width matrix 80/120/200 since fzf layouts shift). Keep the pty lane serial and marked (scale tie…

## Source anchors to inspect first

- `CONTRIBUTING.md:102` — Derived-tier schema changes require rebuild/blue-green planning.
- `AGENTS.md:168` — Agent guidance says schema mismatch should rebuild or blue-green-replace derived tiers.
- `polylogue/cli/commands/reset.py` — Current reset/rebuild commands are the operator path to replace derived tiers.
- `polylogue/daemon/convergence_stages.py` — Daemon convergence/readiness state should represent generation progress honestly.
- `polylogue/storage/sqlite/archive_tiers/index.py:296` — messages_fts table DDL lives in archive_tiers/index.py.
- `polylogue/storage/fts/sql.py:11` — messages_fts DDL copy also exists in storage/fts/sql.py.
- `polylogue/storage/fts/fts_lifecycle.py:292` — ensure_fts_triggers_sync owns runtime repair/recreation.
- `polylogue/storage/fts/fts_lifecycle.py:512` — Thread FTS rebuild logic is duplicated in lifecycle repair path.
- `polylogue/daemon/convergence_stages.py:988` — daemon convergence has another repair/readiness path for archive messages_fts.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. (1) PTY harness in tests/infra: pexpect (or pty+os primitives — decide by trying pexpect's reliability under pytest-xdist) driving the real CLI binary against the seeded corpus: send keys, assert on screen state with normalized snapshots (strip timing/colors via the existing syrupy terminal-snapshot conventions
2. explicit width matrix 80/120/200 since fzf layouts shift).
3. Keep the pty lane serial and marked (scale tier) — pty tests are inherently slower
4. a dozen golden flows, not hundreds.
5. (2) COMPLETION CONTRACTS, no pty needed: invoke the completion entry points directly (Click's shell-complete protocol + the daemon completion endpoint once fnm.4 lands) with a table of (partial-input -> expected candidates) cases generated FROM the grammar registries — the registry is the oracle, so new units/stages get completion tests for free (declare-once payoff).
6. (3) FZF flows: golden scripts per flow (select -> pick -> read
7. judge accept/reject

## Tests to add

- Acceptance proof: PTY harness runs 5+ golden flows green in CI serial lane
- Acceptance proof: completion contract tests are registry-generated and fail when a unit is added without completion metadata
- Acceptance proof: a deliberate fzf-flow regression (reordered candidates) is caught by the harness in a demonstration commit.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
