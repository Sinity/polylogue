# 125. polylogue-212.4 — D4 'Behavioral archaeology': six DSL queries, rapid fire

Priority/type/status: **P2 / task / open**. Lane: **12-external-legibility-demos**. Release: **L-legibility-demos**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Each answers a question an engineering lead would ask, each impossible in any chat UI: SEQ thrash-loop hunt; failure-rate by model; which tools break (observed-event outcomes by tool); near:'race condition' semantic probe across providers; abandoned-in-this-repo-this-quarter; then pipe straight into read. Show explain_query_expression once to prove the query means what it says. Nearly free: all reads exist. Doubles as the DSL reference-card content.

## Existing design note

A demo: six DSL queries, each answering a question an engineering lead would ask and each impossible in a chat UI, SEQ thrash-loop hunt; failure-rate by model; which tools break (observed-event outcomes by tool); a `near:'race condition'` semantic probe across providers; abandoned-in-this-repo-this-quarter; then a query piped straight into `read`. Show `explain_query_expression` once to prove a query means what it says. All underlying reads exist; packaging is the work, and the set doubles as the DSL reference-card content.

## Acceptance criteria

1. Six DSL queries are authored and run against the demo/seeded corpus, each producing sensible results: SEQ thrash-loop, failure-rate by model, tool-breakage by observed-event outcome, `near:` semantic probe across providers, abandoned-this-repo-this-quarter, and a query piped into `read`. 2. `explain_query_expression` is shown once demonstrating a query's parsed meaning. 3. The six queries are captured as the DSL reference-card content (committed demo/doc artifact). Verify: each query runs via `polylogue` against the `polylogue demo seed` corpus (recorded output); the demo script is exercised by the docs/visual lane where applicable.

## Static mechanism / likely defect

Issue description localizes the mechanism: Each answers a question an engineering lead would ask, each impossible in any chat UI: SEQ thrash-loop hunt; failure-rate by model; which tools break (observed-event outcomes by tool); near:'race condition' semantic probe across providers; abandoned-in-this-repo-this-quarter; then pipe straight into read. Show explain_query_expression once to prove the query means what it says. Nearly free: all reads exist. Doubles as the DSL reference-card content. Design direction: A demo: six DSL queries, each answering a question an engineering lead would ask and each impossible in a chat UI, SEQ thrash-loop hunt; failure-rate by model; which tools break (observed-event outcomes by tool); a `near:'race condition'` semantic probe across providers; abandoned-in-this-repo-this-quarter; then a query piped straight into `read`. Show `explain_query_expression` once to prove a query means what it s…

## Source anchors to inspect first

- `README.md` — Public claims should be grounded through the claims ledger.
- `docs/agent-forensics.md` — Existing forensics docs are a pattern for proof artifacts.
- `docs/demo.md` — Demo docs should depend on evidence/citation machinery.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.
- `polylogue/archive/actions/followup.py` — Action/followup classification is a real structural analytics input.
- `polylogue/archive/actions/fields.py` — Action fields determine what can be measured without prose heuristics.
- `polylogue/insights/registry.py:294` — Insight registry should become measure/product registry input.
- `scripts/agent_forensics.py` — Existing forensics script is a proof artifact and candidate product surface.

## Implementation plan

1. A demo: six DSL queries, each answering a question an engineering lead would ask and each impossible in a chat UI, SEQ thrash-loop hunt
2. failure-rate by model
3. which tools break (observed-event outcomes by tool)
4. a `near:'race condition'` semantic probe across providers
5. abandoned-in-this-repo-this-quarter
6. then a query piped straight into `read`.
7. Show `explain_query_expression` once to prove a query means what it says.

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: Six DSL queries are authored and run against the demo/seeded corpus, each producing sensible results: SEQ thrash-loop, failure-rate by model, tool-breakage by observed-event outcome, `near:` semantic probe across providers, abandoned-this-repo-this-quarter, and a query piped into `read`.
- Acceptance proof: 2.
- Acceptance proof: `explain_query_expression` is shown once demonstrating a query's parsed meaning.
- Acceptance proof: 3.
- Acceptance proof: The six queries are captured as the DSL reference-card content (committed demo/doc artifact).
- Acceptance proof: Verify: each query runs via `polylogue` against the `polylogue demo seed` corpus (recorded output)
- Acceptance proof: the demo script is exercised by the docs/visual lane where applicable.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --docs --demo`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
