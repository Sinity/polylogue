# 130. polylogue-kwsb — Security & privacy: the archive can forget on purpose and never leaks secrets

Priority/type/status: **P2 / epic / open**. Lane: **00-trust-floor**. Release: **A-trust-floor**. Readiness: **epic-needs-child-closure**.

## What the bead says

WHY: a personal archive of ALL AI work is the most sensitive database on the machine — it must be able to forget on purpose (excision that provably removes bytes, not just rows) and must never leak (localhost daemon reachable from a hostile page, secrets in captured content). Runtime controls exist and are tested (http.py auth/CSRF, MCP role contracts) but backlog ownership was missing. MEMBER BEADS: polylogue-kwsb.1 (Host/Origin gate + receiver token + spool governor — the live DNS-rebinding hole), polylogue-27m (excision), polylogue-jnj.5 (reset-mutation ordering bug: reset.py tombstones before the preview/--yes gate), polylogue-jsy (crawl-source permissions). Epic closes when the covenant doc's claims are each backed by a test or an explicit non-goal.

## Existing design note

No epic owned the security/privacy surface: excision (27m), reset-mutation safety (jnj.5, real bug at reset.py:260-277 where --session/--source tombstone before the preview:327 and --yes gate:331), and crawl-source permissions (jsy) were orphaned. Runtime controls exist and are tested (http.py auth/CSRF, MCP role contracts) but the BACKLOG ownership was missing. Also owns the security-privacy-coverage.yaml manifest gaps. NON-GOAL: do not resurrect the paused sanitize/redaction cluster (chatlog != spec).

## Acceptance criteria

Excision (right-to-forget + secret redaction + blob excision) is execution-grade and shares one mutation-audit/dry-run/--yes contract with reset (jnj.5); the security-privacy-coverage.yaml gaps each have an owning bead or test; the MCP write/admin destructive path shares the same audit-row contract. Verify: devtools verify + the reset/excision dry-run tests.

## Static mechanism / likely defect

Issue description localizes the mechanism: WHY: a personal archive of ALL AI work is the most sensitive database on the machine — it must be able to forget on purpose (excision that provably removes bytes, not just rows) and must never leak (localhost daemon reachable from a hostile page, secrets in captured content). Runtime controls exist and are tested (http.py auth/CSRF, MCP role contracts) but backlog ownership was missing. MEMBER BEADS: polylogue-kwsb.1 (Host/Origin gate + receiver token + spool governor — the live DNS-rebinding hole), polylogue-27m … Design direction: No epic owned the security/privacy surface: excision (27m), reset-mutation safety (jnj.5, real bug at reset.py:260-277 where --session/--source tombstone before the preview:327 and --yes gate:331), and crawl-source permissions (jsy) were orphaned. Runtime controls exist and are tested (http.py auth/CSRF, MCP role contracts) but the BACKLOG ownership was missing. Also owns the security-privacy-coverage.yaml manifest …

## Source anchors to inspect first

- No precise source anchor was localized in this static pass. Start from the bead description and repository search.

## Implementation plan

1. No epic owned the security/privacy surface: excision (27m), reset-mutation safety (jnj.5, real bug at reset.py:260-277 where --session/--source tombstone before the preview:327 and --yes gate:331), and crawl-source permissions (jsy) were orphaned.
2. Runtime controls exist and are tested (http.py auth/CSRF, MCP role contracts) but the BACKLOG ownership was missing.
3. Also owns the security-privacy-coverage.yaml manifest gaps.
4. NON-GOAL: do not resurrect the paused sanitize/redaction cluster (chatlog != spec).

## Tests to add

- Acceptance proof: Excision (right-to-forget + secret redaction + blob excision) is execution-grade and shares one mutation-audit/dry-run/--yes contract with reset (jnj.5)
- Acceptance proof: the security-privacy-coverage.yaml gaps each have an owning bead or test
- Acceptance proof: the MCP write/admin destructive path shares the same audit-row contract.
- Acceptance proof: Verify: devtools verify + the reset/excision dry-run tests.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
