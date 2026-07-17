# 086. polylogue-gjg.3 — Deterministic loss-forensics: 4-tier structural diff + lost-but-later-needed ranking

Priority/type/status: **P2 / task / open**. Lane: **03-lineage-compaction-truth**. Release: **F-lineage-compaction**. Readiness: **blocked-hard**.

Hard blockers: polylogue-gjg.1, polylogue-gjg.2

## What the bead says

The base retained/lost/transformed classifier is deterministic and structural — NO LLM in the base pass (LLM annotation may layer later as separate judgment rows). Four item tiers with canonical keys: file-path (normalized against repo/cwd), tool-outcome (from the actions view; failed outcomes weighted high — losing a failure record is how agents repeat mistakes), marked-decision (assertions kind decision/lesson/caveat/blocker/handoff + 37t.2 inline markers; ranks highest — losing these is how settled debates reopen), cited-ref (canonicalized commit SHAs / gh refs / file:line / polylogue refs; alias-equivalence = transformed). The harm proxy is LOST-THEN-LATER-NEEDED: later_reference_signal (item key reappears post-compaction in a user request/tool call/failing command/answer) dominates the ranking — measured, not vibes. Decomposed loss_score kept auditable per item.

## Existing design note

Registered as a 9l5.7 measure (compaction-loss) with tier=structural and coverage gates; epidemiology = plain relation algebra over the two tables (rate by provider/trigger/session-length bucket, marked-decision loss rate, failed-tool-outcome loss rate, snapshot-coverage rate). Eager event materialization, lazy loss-item computation on first read, then cached — compaction forensics on a 38GB archive must not run at ingest. Honest degradation: every item carries degraded_reasons; unknown never folded into denominators.

## Acceptance criteria

Classifier is pure + property-tested (same inputs => same items); ranking exposes per-component scores; epidemiology query renders with n + coverage footnotes. Verify: fixtures with known-lost items + measure-registry gate test.

## Static mechanism / likely defect

Issue description localizes the mechanism: The base retained/lost/transformed classifier is deterministic and structural — NO LLM in the base pass (LLM annotation may layer later as separate judgment rows). Four item tiers with canonical keys: file-path (normalized against repo/cwd), tool-outcome (from the actions view; failed outcomes weighted high — losing a failure record is how agents repeat mistakes), marked-decision (assertions kind decision/lesson/caveat/blocker/handoff + 37t.2 inline markers; ranks highest — losing these is how settled debates reop… Design direction: Registered as a 9l5.7 measure (compaction-loss) with tier=structural and coverage gates; epidemiology = plain relation algebra over the two tables (rate by provider/trigger/session-length bucket, marked-decision loss rate, failed-tool-outcome loss rate, snapshot-coverage rate). Eager event materialization, lazy loss-item computation on first read, then cached — compaction forensics on a 38GB archive must not run at …

## Source anchors to inspect first

- `polylogue/archive/session/threads.py` — Session/thread lineage read and composition model.
- `polylogue/insights/topology.py` — Topology/lineage derived insight code.
- `polylogue/daemon/lineage_startup.py` — Daemon lineage startup/convergence path.
- `polylogue/archive/coverage.py` — Completeness/truncation cues live here.
- `polylogue/insights/postmortem.py` — Compaction/continuation postmortem evidence is mined here.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.

## Implementation plan

1. Registered as a 9l5.7 measure (compaction-loss) with tier=structural and coverage gates
2. epidemiology = plain relation algebra over the two tables (rate by provider/trigger/session-length bucket, marked-decision loss rate, failed-tool-outcome loss rate, snapshot-coverage rate).
3. Eager event materialization, lazy loss-item computation on first read, then cached — compaction forensics on a 38GB archive must not run at ingest.
4. Honest degradation: every item carries degraded_reasons
5. unknown never folded into denominators.

## Tests to add

- Acceptance proof: Classifier is pure + property-tested (same inputs => same items)
- Acceptance proof: ranking exposes per-component scores
- Acceptance proof: epidemiology query renders with n + coverage footnotes.
- Acceptance proof: Verify: fixtures with known-lost items + measure-registry gate test.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
