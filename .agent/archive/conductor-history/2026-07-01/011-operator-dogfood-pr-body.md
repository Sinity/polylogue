# PR: operator-dogfood hardening — correct cost accounting + session-lineage substrate

> Draft squash-merge body for `feature/operator-dogfood-hardening`. Title candidate:
> `feat: normalize session lineage and correct cross-provider cost accounting`
> (≤72 chars, conventional, describes what changed). NOT yet pushed — scoping
> (one PR vs split into a cost PR and a substrate PR) is an operator decision.

## Summary

Dogfooding polylogue against the live archive surfaced two classes of correctness
bug — mis-billed tokens and mis-modeled session lineage — and one data-loss bug
(attachments). The historical 37 GB / 16.4K-session / 5.7M-message snapshot was
pre-dedup and counted fork/continuation replay; the current 2026-06-30 active
archive baseline is 2,390 indexed sessions and 159,956 indexed messages. This
branch fixes the substrate so the archive's counts, costs, and transcripts are
trustworthy.

## Problem

- **Cost double-billing.** Codex token accounting summed cumulative columns as
  deltas and counted cached input + reasoning twice (7.69× inflation on some
  sessions). Not every provider/model was priced.
- **Session-lineage duplication (#2467).** Forks, `thread_spawn` subagents,
  resumes, and Claude `agent-acompact-*` auto-compaction copies physically replay
  the parent's context; polylogue stored and counted that prefix N times. Live
  evidence: 286 "continuation" sessions hold 1.77M of 5.73M messages (31%); 0
  `fork` edges existed because the parser was blind to `forked_from_id`; 187
  acompact copies were mislabeled subagents.
- **Attachments not preserved (#2468).** A synthetic hash was written to
  `attachments.blob_hash` while 0 bytes were ever stored (8,425 rows, 8.4 GB
  claimed).

## Solution

Cost (already validated against `state_5.sqlite` / `stats-cache.json`):
- Disjoint billing lanes for Codex (`fresh_input = input_with_cached - cache_read`);
  vendored full LiteLLM price catalog so every provider/model is priced; runnable
  cross-verification demo.

Substrate (index schema v11 → v13, deletes-then-defines; rebuild from source):
- Parsers detect Codex `forked_from_id` + `source.subagent.thread_spawn`
  (FORK/SUBAGENT) and reclassify `agent-acompact-*` as compaction.
- `session_links` gains `branch_point_message_id` + `inheritance`. The writer
  aligns a child against its parent's composed transcript by per-message content
  signature (conservative contiguous prefix-alignment) and stores only the
  divergent tail; reads (`get_messages` + `read_archive_session_envelope`) compose
  the parent prefix back. Out-of-order ingest is handled by deferred re-extraction
  on edge resolution.
- Codex compaction summary is materialized as a message (parity with Claude).
- `attachments.blob_hash` is nullable + real SHA-256 when acquired; new
  `acquisition_status`; inline export bytes are written to the blob store; the
  fabricated hash is gone.

## Verification

- `devtools verify --quick`: whole-repo `mypy --strict` (1679 files) + ruff +
  `render all --check` — green.
- Behavioral suites green: lineage normalization incl. out-of-order
  (tests/unit/storage/test_lineage_normalization.py), keyset ordering invariant,
  MCP envelope/server-surface contracts, provider-usage report, attachment
  acquisition, plus topology/insights/convergence/repair (93 passed) and
  archive-tier write/self-verify/DDL.
- Real-corpus proof: 80 Codex fork families → 898,222 naive message rows →
  64,072 stored (92.9% eliminated), 25/25 children compose correctly; the
  019ccbf9 subagent reconstructs its full 180-message transcript exactly.

## Migration / re-ingest

Schema-touching: there is no in-place upgrade. On deploy the operator runs
`polylogue ops reset --database && polylogued run` — `source.db` and `user.db`
are untouched, so this is a derived-index rebuild. Expected effect: message-row
count drops materially as fork replays collapse; per-thread Codex token ratios
move toward 1.00 vs `state_5.sqlite`.

## Follow-ups (tracked, non-blocking)

- Compaction boundary-range columns + effective-context derivation.
- Attachment acquisition for non-inline sources (Drive OAuth, export-zip, local
  path sanitizer redesign); surface `acquisition_status` in read payloads.

Ref #2467, Ref #2468
