# Hash-boundary census: every digest producer/consumer classified

**Date**: 2026-07-09
**Bead**: polylogue-9e5.6
**Method**: static read of every `hashlib.*`/`hash_text`/`hash_payload`/
`hash_file` call site, plus every SQL/Python comparison that reads a
`content_hash`-or-similar column back and gates a decision on it (skip,
dedupe, invalidate, re-embed). No product code was changed to produce this
census. Two genuine sibling bugs were found and filed separately (not fixed
here — this bead is investigation-only, matching 9e5.4/9e5.13).

## Motivating context

A prior bead found and fixed a hash comparison of this exact shape: a
"message content" digest that didn't actually cover the field the comparison
claimed to protect. The surviving evidence of that fix is in the current
source: `_message_content_hash`'s docstring
(`polylogue/storage/sqlite/archive_tiers/write.py:1497`) says *"Digest the
stored message content, not just its identity"* and explicitly cross-refs
that `message_embeddings_meta.content_hash` freshness depends on this digest
covering the text sent to the embedder. `_block_content_hash`'s docstring
(same file, line 1543) similarly documents "excluding identity (svfj)" —
bead polylogue-svfj, which added `blocks.content_hash` deliberately excluding
session/message/position/tool_id so citation anchors survive fork-position
shift and re-ingest renumbering. This census asks: are there siblings
elsewhere — a hash that *looks* like it verifies real content equality but
doesn't, or a comparison whose result nothing actually acts on?

## Method and scope

`rg 'hashlib\.(sha256|sha1|md5|blake2)'` across `polylogue/` returns **42**
direct call sites. `polylogue/core/hashing.py`'s four helpers
(`hash_text`, `hash_text_short`, `hash_payload`, `hash_file`) are called from
**23** additional sites outside their own definitions. That is **65** total
producer call sites, grouped below by what they protect rather than listed
one-by-one (most are ID-generation helpers sharing one contract). Every site
is accounted for in one of the groups; none were excluded.

Classification:

- **meaningful** — the hash's inputs genuinely correspond to what the
  consumer's comparison claims to protect, and the comparison gates a real,
  reachable decision.
- **meaningful-by-construction (identifier)** — the hash is used only to
  derive a stable identifier (a PRIMARY KEY / dedup key), not compared for
  drift; "consumer" is the uniqueness constraint itself (`INSERT OR IGNORE`
  / `ON CONFLICT DO NOTHING`), which is a legitimate but different contract
  than a content-hash drift check.
- **vacuous/suspect** — the hash's inputs are missing something the
  comparison implies is covered, or the comparison is dead / never reached,
  or the hash's own docstring claims a guarantee nothing enforces.

## Table 1 — Content-hash / integrity drift checks (the load-bearing group)

| # | Producer (file:line) | Inclusion contract (exact fields hashed) | Consumer (file:line) | Decision on match/mismatch | Classification |
|---|---|---|---|---|---|
| 1 | `session_content_hash` → `_session_hash_payload`/`_message_hash_payload` (`polylogue/pipeline/ids.py:99-169`) | title, created_at, updated_at, per-message {id, role, NFC-normalized text, timestamp, content_blocks (type/text/tool_name/tool_id/tool_input-hash/media_type)}, sorted attachments, session_events. NFC-normalized; None vs "" disambiguated via sentinels. **Excludes** user metadata (tags/corrections) by design. | `content_unchanged = existing_hash_hex == payload.content_hash` (`polylogue/pipeline/services/ingest_batch/_core.py:400`); also `sessions_writes.session_exists_by_hash` (`SELECT 1 FROM sessions WHERE content_hash = ?`, older async API path) | Match → skip re-parse/re-write (idempotent re-ingest); mismatch → full session replace + downstream FTS/embedding/insight invalidation | **meaningful** — verified the excluded-by-design claim (tagging doesn't trigger re-import) and the included fields (any message/attachment/event change flips it) |
| 2 | `_message_content_hash` (`polylogue/storage/sqlite/archive_tiers/write.py:1497-1533`) | `"message"`, session_id, provider_message_id, position, variant_index, role, message_type, material_origin, text, user_context_text, then per-block {type, text, tool_name, tool_id, tool_input JSON, semantic_type, media_type, language, is_error, exit_code} | `em.content_hash != stale_m.content_hash` (`polylogue/storage/embeddings/materialization.py:493-495`, `_archive_stale_message_clause`); also surfaced read-only via `STALE_MESSAGES_SQL` (`polylogue/storage/embeddings/sql.py:57-71`) | See Table 2 — this is the freshness signal for embeddings, and it **is** correctly computed (this is the fixed bug's own docstring cross-reference) | **meaningful producer** — see Table 2 for the consumer-side finding |
| 3 | `_block_content_hash` (`write.py:1543-1568`, bead svfj) | `"block"`, block_type, text, tool_name, tool_input_json, semantic_type, media_type, language, is_error, exit_code. **Deliberately excludes** session_id/message_id/position/tool_id | `resolve_block_anchor` (`polylogue/storage/block_anchor.py`) looks up `blocks.content_hash` to resolve a stored citation anchor to `ok`/`drifted_position`/`drifted_message`/`ambiguous`/`hash_mismatch`/`missing` | Anchor resolution across re-ingest/fork-position shift; `hash_mismatch` never auto-rewrites | **meaningful** — svfj's own empirical check (4.46M blocks, 0.069% collision rate) backs the "ambiguous is rare" design claim |
| 4 | `_catalog_hash` (`polylogue/storage/sqlite/archive_tiers/pricing_seed.py:29-40`) | sorted `PRICING` dict entries: `model_name:input_rate:output_rate:cache_read_rate:cache_write_rate` | **none** — `price_catalogs.catalog_hash` is written once at seed time and never `SELECT`ed/compared anywhere in `polylogue/` | Docstring claims "for change-detection"; no branch reads it back, so no decision is ever made on it | **vacuous — filed as polylogue-w379** |
| 5 | `deterministic_blob_hash`/`deterministic_history_sidecar_id` (`source_write.py:125,154`) | Raw payload bytes (blob); origin+source_path+content_hash (sidecar id) | PK uniqueness (`raw_sessions`/history-sidecar tables) | Same content → same id → `INSERT OR IGNORE`/upsert no-op; different content → new row | **meaningful-by-construction** |
| 6 | `BlobStore.write_from_bytes`/`write_from_path` (`polylogue/storage/blob_store.py:99,147,193,246,319`) | Raw file/bytes content, streamed 1 MiB chunks | `BlobStore.verify()` (`blob_store.py:241`) re-hashes on-disk content and compares to the expected hash_hex (the filename/`raw_id`) | Match → blob intact; mismatch → corruption detected (used by `blob_integrity.py` restore/repair flows, e.g. `_restore_expected_hash_from_path`/`_restore_expected_hash_from_source_span`, lines 845-1050) | **meaningful** — content-addressed store, hash IS the address, verify re-derives and compares |
| 7 | `hashlib.sha256` restore hash (`polylogue/storage/blob_integrity.py:1529`) | Recomputed hash of a payload recovered from a `source_path` span during blob-repair | Compared implicitly via blob-store addressing (new blob written under the recomputed hash; original row's `blob_hash` column is what integrity reports diff against) | Drives `_raw_backed_recovery_action` classification (repair vs skip) | **meaningful** |
| 8 | `fingerprint_file`/`tail_hash_from_path`/`tail_hash_and_last_complete_newline_from_path` (`polylogue/sources/live/batch_support.py:161-221`) | Whole-file SHA-256 (full fingerprint) or bounded tail-window SHA-256 (last N bytes) | `CursorStore` compares stored fingerprint/tail-hash against current file state to detect truncation/rotation vs. append-only growth (`sources/live/cursor.py`) | Match → treat as pure append (incremental parse from last offset); mismatch → full re-parse | **meaningful** |
| 9 | `_current_parser_fingerprint` (`polylogue/sources/live/batch.py:814`) | Parser module version/config identity | Compared against a stored fingerprint to decide whether cached parse state is still valid | Mismatch → invalidate cached batch parse state | **meaningful** (not deep-audited beyond signature; low risk — narrow blast radius, config/version string only) |
| 10 | `fingerprint_hash` (`polylogue/schemas/observation_identity.py:29-32`) | `repr()` of a structural fingerprint tuple, truncated to 16 hex chars | `fingerprint_hash(...) == request.cluster_id` (`polylogue/schemas/operator/inference.py:297`) | Selects samples belonging to a schema cluster for operator review | **meaningful** — deterministic grouping key, real equality gate |
| 11 | Content-hash citation anchor format (`format_block_anchor`, `block_anchor.py:86-112`) | Reuses block content_hash (row 3) as the anchor's hex suffix; validates 64-char hex | Parsed back by the anchor resolver | See row 3 | duplicate of row 3, listed for completeness of the anchor format itself |

## Table 2 — Embedding freshness: does `message_embeddings_meta.content_hash` guard anything real?

This was the bead's own named concern. Answer: **partially — the check is
real and correctly computed, but 3 of 4 real selection call sites bypass it.**

`select_pending_archive_session_window()`
(`polylogue/storage/embeddings/materialization.py:263`) takes
`include_stale_checks: bool = True`. When `True`, it folds
`_archive_stale_message_clause` (an `EXISTS` comparing
`em.content_hash != stale_m.content_hash` per message, materialization.py:
482-497) into the "does this session still need embedding work" decision.
This is the correct freshness signal — `messages.content_hash` (Table 1 row
2) changes when the text sent to the embedder changes, and nothing else sets
`embedding_status.needs_reindex=1` for a content-only edit (that flag is only
set by `mark_all_archive_sessions_needs_reindex`, the model/dimension-change
reconciler `_reconcile_embedding_config_change`, or an embedding error path).

| Caller | File:line | `include_stale_checks` | Role |
|---|---|---|---|
| `_archive_pending_embedding_session_ids` (used by both `check` and `execute` of the daemon's per-source-path "embed" `ConvergenceStage`) | `daemon/convergence_stages.py:1219`, execute at `:1246` | relies on default `True` | **The only path that actually detects and re-embeds a content-changed message** — fires per ingested source path |
| `_drain_archive_embedding_backlog_once` (daemon's bulk backlog-catchup sweep) | `daemon/embedding_backlog.py:127` | explicit `False` | Never detects content drift; only picks up missing/`needs_reindex=1` rows |
| `polylogue embed` backfill (operator CLI) | `cli/commands/embed.py:619` | explicit `False` | Same — an operator manually running the main backfill command will not catch edited-message drift without `--rebuild` |
| Embedding preflight/cost estimate | `storage/embeddings/preflight.py:224` | explicit `False` | Estimate window intentionally matches the backfill's own (stale-blind) selection, per its own comment — consistent with the backfill, but propagates the same blind spot |

None of the three `False` sites carry a comment explaining the tradeoff.
Net effect: content-edit drift is corrected **only if** the foreground
per-source-path daemon convergence hook fires for that exact path before the
session would otherwise be touched by the bulk backlog drain or a manual
`polylogue embed` run. If embedding is enabled after the fact, the daemon
isn't running at ingest time, or the per-path probe errors and the session
falls through to backlog-style handling, the drift is silently never
detected by any of the other three paths (short of an operator-invoked
`--rebuild`, which re-embeds everything unconditionally rather than
detecting the specific change).

**Filed as polylogue-wmsc** (not fixed — investigation only).

## Table 3 — ID-generation-only hash sites (no drift comparison; consumer = PK uniqueness)

These 23 sites all call `hash_text`/`hash_text_short`/`hash_payload` purely
to derive a short, stable identifier — the "consumer" is a uniqueness
constraint or dict key, not a later equality check for change detection.
Verified each has a real PK/uniqueness consumer (no dead identifiers):

| Producer | Consumer |
|---|---|
| `polylogue/browser_capture/receiver.py:190` (`hash_text_short` session suffix) | session id disambiguation on capture |
| `polylogue/archive/actions/fields.py:129` (`act-` id) | `actions` view row identity |
| `polylogue/storage/runtime/archive/records.py:126` (`blk-` id) | synthetic block id fallback |
| `polylogue/sources/parsers/drive_support_attachments.py:128-165` (inline-file / youtube-video ids) | attachment dedup key |
| `polylogue/sources/parsers/base_support.py:131` (`att-` id) | attachment id fallback |
| `polylogue/storage/insights/session/timeline_rows.py:54,217` (`wev-`/`sph-` ids) | work-event / session-phase row identity |
| `polylogue/context/compiler.py:190,360` (`query-unit:`/`context-snapshot:` fingerprints) | context segment/snapshot ref, used as a cache/lookup key elsewhere in the context pipeline |
| `polylogue/daemon/user_state_http.py:131,136` (`_default_saved_view_id`/`_default_annotation_id`) | default id when the operator's HTTP request omits one; PK on `saved_views`/`annotations` |
| `polylogue/insights/transforms.py:2099` (evidence digest) | stable evidence-ref id for citation |
| `polylogue/storage/artifacts/inspection.py:28` (`obs-` id) | observation-sample identity |
| `polylogue/publication/__init__.py:53` (whole-file sha256 in an output manifest) | published-artifact manifest entry (informational checksum, not compared against a prior manifest — one-shot scan) |
| `polylogue/core/security.py:35,47` (`original_hash`, truncated 12 chars) | masked-value display in logs (shows a stable short hash instead of the raw secret) — not a comparison, a redaction aid |
| `polylogue/daemon/notification_backends/webhook.py:93` (`hmac.new(..., hashlib.sha256)`) | HMAC signature for outbound webhook payloads — verified by the receiver, out of this repo's scope |
| `polylogue/schemas/sampling_db.py:37`, `schemas/validation/corpus.py:79` (`_blob_hash_hex`) | display/lookup helpers over an existing stored blob_hash column (not new hash computation) |

All of these were spot-checked for a real consumer (uniqueness constraint or
lookup key reachable from a real code path); none showed the "computed and
discarded" shape of row 4 in Table 1.

## Findings summary

- **65 producer call sites** censused (42 direct `hashlib.*` + 23
  `core.hashing` helper call sites), zero left unclassified.
- **1 vacuous producer/consumer pair**: `price_catalogs.catalog_hash` is
  computed with a docstring claiming "change-detection" but is never read
  back anywhere — filed **polylogue-w379**.
- **1 partially-vacuous consumer pattern**: the embedding freshness check
  (`message_embeddings_meta.content_hash` vs `messages.content_hash`) is
  correctly computed and wired into exactly one of four real selection call
  sites; the other three (bulk backlog drain, manual CLI backfill, preflight
  estimate) explicitly disable it with no documented rationale — filed
  **polylogue-wmsc**.
- All other content-hash-shaped comparisons audited (session identity hash,
  message content hash's own producer correctness, block content-hash
  citation anchors, blob-store integrity verify, cursor file-fingerprint
  change detection, schema-cluster fingerprint grouping) are **meaningful**:
  inputs match what the comparison claims, and the branch they gate is
  reachable in production.
- ID-generation-only hash sites (23 of the 65) are a structurally different
  contract (uniqueness key, not drift detector) and were spot-checked rather
  than deep-audited per site; none showed a dead/unreachable consumer.

## Register-or-fail lint

Not built in this pass — documented as follow-up **polylogue-okpn**. The
mechanical check ("every new hashlib/content_hash call site must be
registered with a producer/consumer/classification entry") is real and
worth having, but has non-trivial design surface of its own (registry format
— generated-from-YAML vs. regexing this markdown table; whether pure
ID-generation sites need the same schema as content-hash drift checks; where
it plugs into `devtools verify --quick` without adding latency). Scoping it
properly is more valuable than a rushed version bolted onto this audit pass.

## Not fully verified

- `polylogue/sources/parsers/codex.py:459` and a handful of the smaller
  `hashlib.sha256` sites in `polylogue/storage/sqlite/archive_tiers/
  user_write.py:168` and `async_sqlite_raw.py:110`/`queries/raw_writes.py:32`
  (the latter two are `hashlib.sha256(blob_hash).digest()` — re-hashing an
  *already-hashed* value, likely a sharding/bucketing transform rather than
  a content hash) were confirmed to have a real consumer by grep but not
  traced line-by-line to the same depth as Tables 1-2; nothing in their
  signatures or call sites suggested the vacuous-hash shape, but a deeper
  pass could still find something here.
- `_current_parser_fingerprint` (batch.py:814, Table 1 row 9) was verified
  by signature and one caller, not fully traced end-to-end.
