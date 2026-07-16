# 191. polylogue-212 — Demo portfolio: construct-valid demos (D1/D2/D4/D5/D8 + post-hoc forensic Q&A)

Priority/type/status: **P2 / epic / open**. Lane: **12-external-legibility-demos**. Release: **L-legibility-demos**. Readiness: **epic-needs-child-closure**.

## What the bead says

Ground rule for all: every displayed number resolves, on click or --explain, to structural evidence (outcome fields, usage events, provenance refs, raw bytes) — never regex over prose. Each runs on the deterministic demo corpus (seed 1843) for public reproduction + a live-archive operator variant. D3 (resurrect a dead session) is covered by the context-loop preamble bead + uplift campaign; D6 (Wrapped/one-year-four-assistants) is the forensics campaign artifact; D7 (candidates on trial) is the context-loop judgment flow — do not duplicate them here.

COMPOSITIONALITY RULE (operator, 2026-07-03): every demo must decompose into product primitives — DSL queries, saved views, read-package layouts, render profiles, workflow registry entries. Shell/python is allowed only as glue (sequencing, narration). If a demo needs bespoke logic beyond glue, that logic is a missing product primitive: file the primitive as a bead, build it, THEN ship the demo on top. Demos are the forcing function for product algebra, not a parallel scripts directory (the agent_forensics.py -> polylogue analyze fold in tf2.2 is the template).

## Existing design note

Portfolio contract (see 212.7): every demo = executable PROMPT.md emitting the uniform Demo Finding Packet; product primitives only, shell as glue; anti-demo (212.8) ships beside successes. IDEA MENU: a 60-item grounded demo catalog from the 2026-07-06 corpus digestion is preserved at .agent/scratch/corpus-gpt-pro-2026-07-06/D-demos.md — pull from it when extending the portfolio; most items converge on six primitives now tracked elsewhere (query runs rxdo.3, cohorts rxdo.2, annotation batches rxdo.7, artifact edges 1vpm.3, analysis runs rxdo.8, context-compile runs 37t.11/gjg.4). Standouts beyond the current children: Beads swarm autopsy + before/after backlog-quality audit (process story), stale-docs-vs-code reality check, notes-sidecar trap detector, GitHub external-ref reconciliation (operator checklist, never auto-mutation), commit<->session archaeology both directions (7xv), memory-utility analytics (37t.17), flat-dump-vs-compiled-context (gjg.4/37t.11 arm), archive-root pitfall detector (fold into doctor/adoption lane).

## Acceptance criteria

Each demo child (212.1 post-hoc forensic Q&A, 212.2 D1, 212.3 D2, 212.4 D4, 212.5 D5, 212.6 D8) ships in two variants: (a) a public seeded-corpus variant (seed 1843) reproducible with one documented command, and (b) a live-archive operator variant. GROUND RULE: every displayed number resolves, on click or --explain, to structural evidence (outcome fields, usage events, provenance refs, raw bytes) — never regex over prose. COMPOSITIONALITY: every demo decomposes into product primitives (DSL queries, saved views, read-package layouts, render profiles, workflow-registry entries); shell/python is glue only, and any bespoke logic beyond glue is first filed and built as a product primitive. D3/D6/D7 are explicitly out of scope (covered by the context-loop/uplift/forensics campaigns). Epic closeable when all non-deferred children are closed and a cold-reader can drive each public variant to first result unaided. Verify: each child's own acceptance + devtools verify doc-commands over the demo commands.

## Static mechanism / likely defect

Issue description localizes the mechanism: Ground rule for all: every displayed number resolves, on click or --explain, to structural evidence (outcome fields, usage events, provenance refs, raw bytes) — never regex over prose. Each runs on the deterministic demo corpus (seed 1843) for public reproduction + a live-archive operator variant. D3 (resurrect a dead session) is covered by the context-loop preamble bead + uplift campaign; D6 (Wrapped/one-year-four-assistants) is the forensics campaign artifact; D7 (candidates on trial) is the context-loop judgmen… Design direction: Portfolio contract (see 212.7): every demo = executable PROMPT.md emitting the uniform Demo Finding Packet; product primitives only, shell as glue; anti-demo (212.8) ships beside successes. IDEA MENU: a 60-item grounded demo catalog from the 2026-07-06 corpus digestion is preserved at .agent/scratch/corpus-gpt-pro-2026-07-06/D-demos.md — pull from it when extending the portfolio; most items converge on six primitive…

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

1. Portfolio contract (see 212.7): every demo = executable PROMPT.md emitting the uniform Demo Finding Packet
2. product primitives only, shell as glue
3. anti-demo (212.8) ships beside successes.
4. IDEA MENU: a 60-item grounded demo catalog from the 2026-07-06 corpus digestion is preserved at .agent/scratch/corpus-gpt-pro-2026-07-06/D-demos.md — pull from it when extending the portfolio
5. most items converge on six primitives now tracked elsewhere (query runs rxdo.3, cohorts rxdo.2, annotation batches rxdo.7, artifact edges 1vpm.3, analysis runs rxdo.8, context-compile runs 37t.11/gjg.4).
6. Standouts beyond the current children: Beads swarm autopsy + before/after backlog-quality audit (process story), stale-docs-vs-code reality check, notes-sidecar trap detector, GitHub external-ref reconciliation (operator checklist, never auto-mutation), commit<->session archaeology both directions (7xv), memory-utility analytics (37t.17), flat-dump-vs-compiled-context (gjg.4/37t.11 arm), archive-root pitfall detec…

## Tests to add

- Acceptance proof: Each demo child (212.1 post-hoc forensic Q&A, 212.2 D1, 212.3 D2, 212.4 D4, 212.5 D5, 212.6 D8) ships in two variants: (a) a public seeded-corpus variant (seed 1843) reproducible with one documented command, and (b) a live-archive operator variant.
- Acceptance proof: GROUND RULE: every displayed number resolves, on click or --explain, to structural evidence (outcome fields, usage events, provenance refs, raw bytes) — never regex over prose.
- Acceptance proof: COMPOSITIONALITY: every demo decomposes into product primitives (DSL queries, saved views, read-package layouts, render profiles, workflow-registry entries)
- Acceptance proof: shell/python is glue only, and any bespoke logic beyond glue is first filed and built as a product primitive.
- Acceptance proof: D3/D6/D7 are explicitly out of scope (covered by the context-loop/uplift/forensics campaigns).
- Acceptance proof: Epic closeable when all non-deferred children are closed and a cold-reader can drive each public variant to first result unaided.
- Acceptance proof: Verify: each child's own acceptance + devtools verify doc-commands over the demo commands.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --docs --demo`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.
- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
