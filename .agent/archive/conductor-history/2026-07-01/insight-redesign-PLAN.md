---
created: 2026-06-28
purpose: The FULL insight-layer redesign (the real task) — not the env-gate stopgap
status: active — design mostly converged; "full missing scope" still open; execution not started
read-with: STATE-OF-UNDERSTANDING-2026-06-28.md (§4,5,8,10), insights-dissection-2026-06-28.md, construct-validity-audit-2026-06-28.md
---

# Insight layer redesign

## Why (established, evidence-backed — see STATE doc)
- Deferred insight pass = ~9× the archive write (57ms/session, ≈15min on real archive). 76% is
  `_events_from_text` regex prose-mining via `compile_recovery_digest()` (`rebuild.py:617`).
- "insights" is an INCOHERENT category: cheap deterministic stats + structural lineage + expensive
  heuristic prose-mining, all under one deferred stage + one materializer_version.
- Construct-invalid: work_events ("should we merge PR#123"→recorded "merged"); material_origin=HUMAN is
  a fall-through default ("no robot marker matched" ≠ human); edge-types over-claim (7 advertised, Codex
  collapses fork+resume→FORK, RESUME/REPAIRED never assigned); has_paste unions marker w/ size proxy.

## STOPGAP IN PLACE (temporary, NOT the solution)
`rebuild.py`: `POLYLOGUE_SKIP_RUN_PROJECTION=1` skips the compile_recovery_digest run-projection trio →
kills the 9× tax for dev reingest. Dev daemon runs with it. Default behavior unchanged. This must be
SUPERSEDED by the real redesign, not left as the answer.

## Target design — substrate-not-interpreter
GUIDING PRINCIPLE: **record structure, name it honestly, default to UNKNOWN, assert a positive class
only on positive evidence. Capabilities emerge from the algebra (views/queries over a clean substrate);
materialize ONLY what is expensive-to-compute AND frequently-read AND stable (FTS, embeddings).** No
confidence/provenance machinery while everything is deterministic (add only if/when a genuinely good
heuristic is introduced).

Three things the current "insights" conflates, separated:
- **A. Cheap deterministic per-session stats** (counts, durations, cost, token lanes, latency). Pure
  functions of one session's rows. → COMPUTE INLINE at write as columns on `sessions` (which already
  holds ~18 such columns) OR as SQL VIEWS. Drop the separate deferred `session_profiles` table + its
  materialization tracking + read join. Sessions become atomic; the per-session insight convergence
  stage + most of live_convergence_debt evaporate. Cost as a VIEW over token cols × price catalog →
  reprice-for-free, never stale (retires reprice-in-place as unnecessary).
- **B. Structural lineage** (threads/thread_sessions, topology). Keep; it's topology, not "insight".
- **C. CUT the regex prose-miner** (work_events, phases dead cols, run-projection-via-digest). Replace
  with outcomes READ from structured tool-calls+results — which requires the KEYSTONE:

## KEYSTONE — structured tool-result capture
`blocks` stores tool_name/tool_input (structured) but the tool RESULT only as text — no is_error/exit
column. Source JSONL (Claude `toolUseResult.is_error`, Codex exit) is dropped at parse. FIX: persist
structured outcome (is_error/exit_status) on tool_result blocks (parser + schema bump + reingest).
Enables reading outcomes ("tests passed", "merge succeeded") instead of regex-guessing.

## Construct-validity fixes (apply the principle)
- material_origin: default UNKNOWN; label HUMAN_AUTHORED only on positive human-input signal.
- TopologyEdgeType: assign only provider-authoritative relations; leave rest UNKNOWN (don't default FORK).
- has_paste: marker = truth (separate column); size/code-fence proxy = a DIFFERENT signal, not unioned.
- run-projection query units (`find run/observed-event/context-snapshot`): rebuild from structured data
  if kept, or drop the units (recovery MCP recomputes on read anyway). Don't materialize the regex digest.

## THE OPEN PIECE — "full missing scope" (operator emphasized: figure out what insights SHOULD be)
This is unfinished design. Define the full target taxonomy of derived analytics polylogue should expose
(all deterministic, from structure; mined from Lynchpin's proven-useful catalog — see STATE §8 + the
Lynchpin division-of-labor: polylogue = intrinsic AI-session substrate; Lynchpin = cross-source fusion):
- Tool mechanics: per-tool counts, success/failure/error rates, retry chains, edit→verify loops.
- Timing: turn gaps, tool-call durations, time-to-first-response, idle/stuck gaps, time-to-completion.
- Outcome/abandonment — STRUCTURALLY defined (ended mid-tool-call / on unresolved error / no commit after
  edits), replacing heuristic terminal_state/workflow_shape.
- **Claim-vs-evidence calibration** (FLAGSHIP, unique to polylogue): did the assistant's claim match the
  structured tool outcome? "said tests pass" vs pytest exit; "claimed fixed X" vs an actual Edit/commit.
  Construct-validity-as-a-feature; only possible here (transcript + outcomes together).
- Conversation mechanics: user/assistant ratio, correction/interrupt/restart, compaction-pressure events.
- Cross-AI-session trends (cheap views/aggregates): cost over time, model evolution, tool-use evolution,
  session-volume rhythm.
TODO: turn this into a concrete table (name / inputs / view-or-materialized / consumer / why) and decide
the materialize-vs-view boundary per item. This is the next design deliverable before building.

## Sequencing (proposed; needs no further sign-off on direction, only on scope detail)
1. Keystone: structured tool-result capture (parser + schema). Unblocks everything.
2. Inline deterministic stats onto sessions / views; drop session_profiles deferral.
3. Cut regex miner; derive outcomes from structured results; handle run-projection units.
4. Construct-validity fixes (material_origin, edge types, has_paste).
5. Add principled analytics per the missing-scope table (claim-calibration flagship).
6. Remove the env-gate stopgap (no longer needed once the digest is gone).
Each step: real artifact on real data, validated against the dev archive.
