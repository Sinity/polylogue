# 119. polylogue-cfk — Re-run two-arm uplift with freshness-fixed packs (n>=3 pairs, then n=12-20)

Priority/type/status: **P1 / task / open**. Lane: **10-analytics-experiments**. Release: **I-analytics-experiments**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Successor to campaign polylogue-jxe, which closed diagnostic-negative: raw-ref arm 8/10 vs handoff-pack 5/10, with the loss attributed to packet staleness (pack generated before later devloop work; raw-ref arm found newer beads/archive evidence). The cause was fixed (polylogue-yps freshness metadata + successor links, polylogue-qt3 fast single-process regeneration) but nothing re-tests the hypothesis. Until a re-run exists, the recorded result of the only uplift experiment in the program is 'packs lose', and the context/memory-loop program (polylogue-37t) is building on an unvalidated premise.

## Existing design note

Protocol: identical paired two-arm design as jxe.2/jxe.3 (same scoring rubric, blinded arms, committed comparison artifact under .agent/demos/uplift-two-arm/, cold-reader gate). The one deliberate change: the pack arm consumes a pack REGENERATED AT CONTINUATION START, not a shelf artifact — qt3 made regeneration seconds-fast and in-process; yps metadata must show generated_at ~= consumption time, freshness state fresh, zero successor warnings. Run n>=3 pairs first to de-noise the n=1 pilot; a publishable uplift claim needs n=12-20 pairs. Record the secondary hypothesis rather than assuming it: the raw-ref arm won partly because it could QUERY live state (bd ready, archive search) — if fresh packs still lose, the product conclusion is 'pack = bootstrap seed + live query affordances, not a substitute for querying', and that conclusion should be fed into polylogue-37t design before more pack-content iteration. Pitfall: do not compare against the stale-pack pilot scores directly; the arms must be re-scored on the same new subjects.

PROTOCOL DECISION (operator, 2026-07-03): pack arm = fresh continuation-time pack AND live query access (bd/archive); raw-ref arm = query access only. This tests 'a pack is a better starting point', not 'a pack substitutes for querying' — matching how agents actually work. The pack-only variant was considered and rejected for round one (repeats the pilot's construct problem in a new form); revisit a three-arm design only if the n>=3 result is ambiguous.

## Acceptance criteria

n>=3 paired runs completed under the recorded protocol (fresh continuation-time pack + live query vs raw-ref + live query); per-pair scores + paired analysis committed under .agent/demos/uplift-two-arm/; cold-reader gate on the comparison artifact; result recorded in the bead (positive, negative, or ambiguous -> three-arm follow-up decision).

## Static mechanism / likely defect

Issue description localizes the mechanism: Successor to campaign polylogue-jxe, which closed diagnostic-negative: raw-ref arm 8/10 vs handoff-pack 5/10, with the loss attributed to packet staleness (pack generated before later devloop work; raw-ref arm found newer beads/archive evidence). The cause was fixed (polylogue-yps freshness metadata + successor links, polylogue-qt3 fast single-process regeneration) but nothing re-tests the hypothesis. Until a re-run exists, the recorded result of the only uplift experiment in the program is 'packs lose', and the c… Design direction: Protocol: identical paired two-arm design as jxe.2/jxe.3 (same scoring rubric, blinded arms, committed comparison artifact under .agent/demos/uplift-two-arm/, cold-reader gate). The one deliberate change: the pack arm consumes a pack REGENERATED AT CONTINUATION START, not a shelf artifact — qt3 made regeneration seconds-fast and in-process; yps metadata must show generated_at ~= consumption time, freshness state fre…

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

1. Protocol: identical paired two-arm design as jxe.2/jxe.3 (same scoring rubric, blinded arms, committed comparison artifact under .agent/demos/uplift-two-arm/, cold-reader gate).
2. The one deliberate change: the pack arm consumes a pack REGENERATED AT CONTINUATION START, not a shelf artifact — qt3 made regeneration seconds-fast and in-process
3. yps metadata must show generated_at ~= consumption time, freshness state fresh, zero successor warnings.
4. Run n>=3 pairs first to de-noise the n=1 pilot
5. a publishable uplift claim needs n=12-20 pairs.
6. Record the secondary hypothesis rather than assuming it: the raw-ref arm won partly because it could QUERY live state (bd ready, archive search) — if fresh packs still lose, the product conclusion is 'pack = bootstrap seed + live query affordances, not a substitute for querying', and that conclusion should be fed into polylogue-37t design before more pack-content iteration.
7. Pitfall: do not compare against the stale-pack pilot scores directly

## Tests to add

- Acceptance proof: n>=3 paired runs completed under the recorded protocol (fresh continuation-time pack + live query vs raw-ref + live query)
- Acceptance proof: per-pair scores + paired analysis committed under .agent/demos/uplift-two-arm/
- Acceptance proof: cold-reader gate on the comparison artifact
- Acceptance proof: result recorded in the bead (positive, negative, or ambiguous -> three-arm follow-up decision).

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
