# 115. polylogue-1vpm.4 — Turn-pair unit with prompt-burst semantics (no double-claimed answers)

Priority/type/status: **P2 / task / open**. Lane: **10-analytics-experiments**. Release: **I-analytics-experiments**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Per-turn latency/cost/correction-rate needs a prompt->answer relation, and the naive pairing law (each prompt -> MIN(next assistant)) is WRONG: two human messages before one answer both claim it. Corrected design: group consecutive human_authored/operator_command prompts into a PROMPT BURST before the next assistant_authored active-path answer; expose prompt_message_ids, burst_size, answer refs, latency (NULL unless both timestamps), token columns, abandoned=true for trailing unanswered bursts. material_origin adjacency is the basis (VIEW per units-B spec); operator_command never silently counted as human prose (prompt_origin filter). Index-tier VIEW + covering index; full query-unit registration ritual (descriptor, payload, schemas, completions, topology regen).

## Acceptance criteria

human->human->assistant yields ONE pair with burst_size=2; tool rows skipped; trailing burst abandoned=true; latency NULL-safe; turn-pairs where answer_model:X works cross-surface. Verify: fixture + unit tests.

## Static mechanism / likely defect

Issue description localizes the mechanism: Per-turn latency/cost/correction-rate needs a prompt->answer relation, and the naive pairing law (each prompt -> MIN(next assistant)) is WRONG: two human messages before one answer both claim it. Corrected design: group consecutive human_authored/operator_command prompts into a PROMPT BURST before the next assistant_authored active-path answer; expose prompt_message_ids, burst_size, answer refs, latency (NULL unless both timestamps), token columns, abandoned=true for trailing unanswered bursts. material_origin adj…

## Source anchors to inspect first

- `polylogue/archive/actions/followup.py` — Action/followup classification is a real structural analytics input.
- `polylogue/archive/actions/fields.py` — Action fields determine what can be measured without prose heuristics.
- `polylogue/insights/registry.py:294` — Insight registry should become measure/product registry input.
- `scripts/agent_forensics.py` — Existing forensics script is a proof artifact and candidate product surface.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.

## Implementation plan

1. Register the measure/outcome with evidence tier, denominator, and uncertainty.
2. Materialize only after source units are stable.
3. Add fixture proving empty/uncovered samples do not become zeros.
4. Render caveats in CLI/report/web outputs.

## Tests to add

- Acceptance proof: human->human->assistant yields ONE pair with burst_size=2
- Acceptance proof: tool rows skipped
- Acceptance proof: trailing burst abandoned=true
- Acceptance proof: latency NULL-safe
- Acceptance proof: turn-pairs where answer_model:X works cross-surface.
- Acceptance proof: Verify: fixture + unit tests.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
