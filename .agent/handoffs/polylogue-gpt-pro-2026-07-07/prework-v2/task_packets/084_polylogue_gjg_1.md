# 084. polylogue-gjg.1 — compaction_events + compaction_loss_items derived tables; identity survives rebuild + re-ingest

Priority/type/status: **P2 / task / open**. Lane: **03-lineage-compaction-truth**. Release: **F-lineage-compaction**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Promote compaction from a session_events count + lineage edge to an archived object. New index.db tables: compaction_events (boundary message pointers, lineage link fields, trigger/pre_tokens/preserved_segment from the harness event, snapshot_ref + snapshot_source + snapshot_confidence, degraded_reasons) and compaction_loss_items (tier, canonical item key, retained/lost/transformed/unknown classification, pre/post/later-reference anchors, decomposed scores). Keep the session_events row as the compat index row. Identity: compaction_id hashes ORIGIN-NATIVE identifiers (origin, native session id, provider boundary uuid/source line, provider boundary message ids) — never SQLite rowids and not normalized message_id alone — so the id survives derived rebuild AND source re-ingest; a separate event_content_hash over the interpreted payload lets rebuilds detect same-event-changed-interpretation loudly.

## Existing design note

Derived tier: edit canonical DDL + bump INDEX_SCHEMA_VERSION, batch with the next index bump window (ma2/4ts.5 rule). Extractor already exists (detect_context_compaction handles legacy summary + modern compact_boundary with trigger/pre_tokens/preserved_segment); this materializes it. Blocked conceptually by 4ts.5 (boundary-range columns) which gjg already depends on — coordinate the two in one index bump.

## Acceptance criteria

Rebuild from source produces identical compaction_ids; a re-ingested session keeps its compaction rows; changed interpretation surfaces as event_content_hash delta not silent overwrite. Verify: fixture tests over legacy+modern Claude compactions + Codex compacted records.

## Static mechanism / likely defect

Issue description localizes the mechanism: Promote compaction from a session_events count + lineage edge to an archived object. New index.db tables: compaction_events (boundary message pointers, lineage link fields, trigger/pre_tokens/preserved_segment from the harness event, snapshot_ref + snapshot_source + snapshot_confidence, degraded_reasons) and compaction_loss_items (tier, canonical item key, retained/lost/transformed/unknown classification, pre/post/later-reference anchors, decomposed scores). Keep the session_events row as the compat index row. Ide… Design direction: Derived tier: edit canonical DDL + bump INDEX_SCHEMA_VERSION, batch with the next index bump window (ma2/4ts.5 rule). Extractor already exists (detect_context_compaction handles legacy summary + modern compact_boundary with trigger/pre_tokens/preserved_segment); this materializes it. Blocked conceptually by 4ts.5 (boundary-range columns) which gjg already depends on — coordinate the two in one index bump.

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

1. Derived tier: edit canonical DDL + bump INDEX_SCHEMA_VERSION, batch with the next index bump window (ma2/4ts.5 rule).
2. Extractor already exists (detect_context_compaction handles legacy summary + modern compact_boundary with trigger/pre_tokens/preserved_segment)
3. this materializes it.
4. Blocked conceptually by 4ts.5 (boundary-range columns) which gjg already depends on — coordinate the two in one index bump.

## Tests to add

- Acceptance proof: Rebuild from source produces identical compaction_ids
- Acceptance proof: a re-ingested session keeps its compaction rows
- Acceptance proof: changed interpretation surfaces as event_content_hash delta not silent overwrite.
- Acceptance proof: Verify: fixture tests over legacy+modern Claude compactions + Codex compacted records.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
