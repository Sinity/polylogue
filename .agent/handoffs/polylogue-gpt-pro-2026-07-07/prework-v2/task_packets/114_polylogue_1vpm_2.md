# 114. polylogue-1vpm.2 — Episode unit: tables, 4-signal scorer with false-merge floor, assertion-calibrated

Priority/type/status: **P2 / task / open**. Lane: **10-analytics-experiments**. Release: **I-analytics-experiments**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

episodes / episode_members / episode_edges in index.db. EDGES ARE THE UNIT OF EVIDENCE (member-only storage loses why A attached to B); episode = connected component over eligible edges only. member_set_hash = sha256 of sorted member refs => idempotent re-stitch, scorer version as metadata not identity (same member set = same hypothesis, confidence may change). Members beyond sessions: commit/pr/issue/artifact/raw_event (telemetry can join with no matching AI session). Signals persisted per-edge with contributions: repo/cwd (hard prior; different repo root = strong negative but NOT absolute veto — cross-repo bridges via hard artifacts allowed), repo-conditioned asymmetric time kernel, session-summary embedding (derived from message embeddings weighted over authored material_origin until a session-embedding family exists), shared-hard-artifact (SHA/PR/issue/path-after-normalization/error-fingerprint — dominates). Tiers: linked (topology-proven, quarantined edges excluded) / corroborated (>=2 independent signals, one hard) / candidate (semantic+time only — NEVER default-merged). Anti-stitch signals subtract and can quarantine; quarantined topology cycle-break is an absolute veto sans operator override. Operator confirm/split/reject/quarantine stored as assertions targeting episode/episode-edge refs; accepted/rejected decisions replay as constraints during rebuild AND feed scorer calibration. Rollups honor logical-session dedup (4ts) + material_origin. Verbatim spec: bundles/rnd-bundle-6-of-6.md L466-715.

## Acceptance criteria

Deliberately under-stitches on first corpus (polylogue repo work first — strongest evidence density); zero candidate-only merges in default render; edge evidence auditable; operator decisions survive rebuild; episodes where member.origin:chatgpt and member.origin:claude-code returns cross-tool episodes. Verify: scorer property tests + seeded fixture corpus + precision audit protocol before default-on.

## Static mechanism / likely defect

Issue description localizes the mechanism: episodes / episode_members / episode_edges in index.db. EDGES ARE THE UNIT OF EVIDENCE (member-only storage loses why A attached to B); episode = connected component over eligible edges only. member_set_hash = sha256 of sorted member refs => idempotent re-stitch, scorer version as metadata not identity (same member set = same hypothesis, confidence may change). Members beyond sessions: commit/pr/issue/artifact/raw_event (telemetry can join with no matching AI session). Signals persisted per-edge with contributions…

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

- Acceptance proof: Deliberately under-stitches on first corpus (polylogue repo work first — strongest evidence density)
- Acceptance proof: zero candidate-only merges in default render
- Acceptance proof: edge evidence auditable
- Acceptance proof: operator decisions survive rebuild
- Acceptance proof: episodes where member.origin:chatgpt and member.origin:claude-code returns cross-tool episodes.
- Acceptance proof: Verify: scorer property tests + seeded fixture corpus + precision audit protocol before default-on.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
