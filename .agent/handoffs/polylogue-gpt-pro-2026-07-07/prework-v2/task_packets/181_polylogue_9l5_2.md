# 181. polylogue-9l5.2 — Cross-provider comparative analytics

Priority/type/status: **P2 / feature / open**. Lane: **10-analytics-experiments**. Release: **I-analytics-experiments**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

The archive is the only place Claude/Codex/ChatGPT/Gemini work traces coexist normalized: same task-shape comparisons — failure rates, retry behavior, cost per completed session, tool-mix, session lengths — by origin/model with explicit coverage tiers per origin so partial provenance cannot masquerade as a finding. This relation is also what the public leaderboard variant reads.

## Existing design note

The killer query shape: 'same repo, same month: Claude Code vs Codex — turns per task, $/session, tool-failure rate, subagent usage' — no lab and no observability vendor can run it; the archive is the only place these providers coexist normalized. Honesty by construction: the coverage matrix (storage/usage.py:51-139) already annotates exact vs estimated accounting per origin — every comparison row carries its coverage tier as a footnote, so partial provenance cannot masquerade as a finding. This relation is also what the public leaderboard variant reads.

THE $0 LANE (fables interop analysis): once local-model sessions exist in the archive (Hermes/Ollama behind the LiteLLM gateway), the same comparison gains a free-lane column — local-model vs API harnesses on the same repo and task class: turns, failure rates, wall-clock, and actual cost $0 vs the API-equivalent counterfactual the api_equivalent cost axis already computes. Answers 'when is the free lane good enough?' with structural outcomes instead of vibes — and it is exactly the evidence-backed comparison shape the open-model community amplifies. Requires fs1.1 keystone outcome extraction so local-agent sessions are outcome-comparable, and per-origin coverage tiers stay mandatory.

## Acceptance criteria

1. On the seeded corpus a cross-origin same-task comparison (turns/task, $/session, tool-failure rate, subagent usage) renders WITH a per-origin coverage-tier footnote on EVERY row, sourced from the storage/usage.py coverage matrix (exact vs estimated per origin). 2. A comparison where one origin lacks priced provenance is REFUSED as a bare number at composition and returns an actionable error (the 9l5.7 composition/honesty guard), not a silent partial. 3. When local-model ($0-lane) sessions are present, a free-lane column shows actual $0 vs the api_equivalent counterfactual. Verify: a DSL `... | compare origin:claude-code-session vs codex-session` query renders on the demo archive; a snapshot test asserts the mandatory coverage footnote AND exercises the refusal path when one origin's provenance is unpriced. Note: the design's 'fs1.1 keystone outcome extraction' phrase is a stale conflation — the outcome-extraction keystone is the closed sru.1; fs1.1 (Hermes importer) is the separate prerequisite only for the $0-lane column's local sessions (orchestrator to rewire the deps accordingly).

## Static mechanism / likely defect

Issue description localizes the mechanism: The archive is the only place Claude/Codex/ChatGPT/Gemini work traces coexist normalized: same task-shape comparisons — failure rates, retry behavior, cost per completed session, tool-mix, session lengths — by origin/model with explicit coverage tiers per origin so partial provenance cannot masquerade as a finding. This relation is also what the public leaderboard variant reads. Design direction: The killer query shape: 'same repo, same month: Claude Code vs Codex — turns per task, $/session, tool-failure rate, subagent usage' — no lab and no observability vendor can run it; the archive is the only place these providers coexist normalized. Honesty by construction: the coverage matrix (storage/usage.py:51-139) already annotates exact vs estimated accounting per origin — every comparison row carries its covera…

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

1. The killer query shape: 'same repo, same month: Claude Code vs Codex — turns per task, $/session, tool-failure rate, subagent usage' — no lab and no observability vendor can run it
2. the archive is the only place these providers coexist normalized.
3. Honesty by construction: the coverage matrix (storage/usage.py:51-139) already annotates exact vs estimated accounting per origin — every comparison row carries its coverage tier as a footnote, so partial provenance cannot masquerade as a finding.
4. This relation is also what the public leaderboard variant reads.
5. THE $0 LANE (fables interop analysis): once local-model sessions exist in the archive (Hermes/Ollama behind the LiteLLM gateway), the same comparison gains a free-lane column — local-model vs API harnesses on the same repo and task class: turns, failure rates, wall-clock, and actual cost $0 vs the API-equivalent counterfactual the api_equivalent cost axis already computes.
6. Answers 'when is the free lane good enough?' with structural outcomes instead of vibes — and it is exactly the evidence-backed comparison shape the open-model community amplifies.
7. Requires fs1.1 keystone outcome extraction so local-agent sessions are outcome-comparable, and per-origin coverage tiers stay mandatory.

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: On the seeded corpus a cross-origin same-task comparison (turns/task, $/session, tool-failure rate, subagent usage) renders WITH a per-origin coverage-tier footnote on EVERY row, sourced from the storage/usage.py coverage matrix (exact vs estimated per origin).
- Acceptance proof: 2.
- Acceptance proof: A comparison where one origin lacks priced provenance is REFUSED as a bare number at composition and returns an actionable error (the 9l5.7 composition/honesty guard), not a silent partial.
- Acceptance proof: 3.
- Acceptance proof: When local-model ($0-lane) sessions are present, a free-lane column shows actual $0 vs the api_equivalent counterfactual.
- Acceptance proof: Verify: a DSL `...
- Acceptance proof: | compare origin:claude-code-session vs codex-session` query renders on the demo archive

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
