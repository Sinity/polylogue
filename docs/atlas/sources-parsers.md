# Sources and Parsers

## Area boundary

Sources acquire bytes and identify their material source. Detection chooses a
provider parser by input shape; the pipeline normalizes provider records into
parsed sessions before the storage writer lowers them
(`polylogue/sources/dispatch.py:1-80`; `CompiledDetectorRegistry.detect` in `polylogue/sources/detection.py:88-105`;
`ingest_record` in
`polylogue/pipeline/services/ingest_worker.py:1137-1226`; `_run_parse_plan`
in the same file at `1037-1089`;
`_materialize_parsed_sessions` in the same file at `896-959`).

## Detection and parse route

1. Acquisition records raw bytes and source metadata in `source.db`.
2. Detector bindings are declared by each `OriginSpec`, not by `dispatch.py`.
   `compile_detector_registry` validates them and sorts them per
   `DetectionMode` by `(mode_rank or detector_tightness, local_rank,
   binding_id)`; `CompiledDetectorRegistry.detect` returns the first predicate
   that claims the payload (`polylogue/sources/detection.py:88-105`;
   `polylogue/sources/detection.py:196-227`).
3. The selected provider parser emits normalized sessions, messages, blocks,
   tool uses, tool results, and lineage hints.
4. `write_parsed_session_to_archive` computes public origin and identities,
   writes the parsed tree, and resolves asserted parent links
   (`write_parsed_session_to_archive` in
   `polylogue/storage/sqlite/archive_tiers/write.py:2119`).
5. The daemon converger materializes FTS, embeddings, and insight read models.

## Detector tightness order

Lower number runs first. Tightness must be unique among executable
`OriginSpec`s, which is enforced at spec validation
(`OriginSpecRegistry.diagnostics` in
`polylogue/sources/origin_specs.py:1452-1462`). Current executable order:

| Tightness | Origin |
| --- | --- |
| 10 | `gemini-cli-session` |
| 20 | `hermes-session` |
| 30 | `antigravity-session` |
| 50 | `codex-session` |
| 60 | `claude-code-session` |
| 70 | `chatgpt-export` |
| 80 | `claude-ai-export` |
| 82 | `claude-design-session` |
| 85 | `grok-export` |
| 90 | `aistudio-drive` |
| 95 | `otel-genai` |

`beads-issue` (reserved) and `unknown-export` (compatibility-only) carry no
tightness and are not executable detectors. A new detector inserted looser
than it deserves lets an earlier parser claim its records; a binding may also
declare `mode_rank` to take a different precedence in one payload mode than
its origin-wide tightness (`polylogue/sources/detection.py:36-42`).

## Identity vocabulary

`Provider` names the older provider-wire family at acquisition/parser/schema
boundaries. `Origin` is the public source-origin token and the only coordinate
public filters accept. `Source` carries richer acquisition identity. The
mapping is non-injective -- `Provider.GEMINI` and `Provider.DRIVE` both map to
`Origin.AISTUDIO_DRIVE` -- so an origin must never be reversed into a guessed
provider (`docs/provider-origin-identity.md:15-30`;
`docs/provider-origin-identity.md:108-115`).

## Invariants

- Native Codex browser-capture envelopes retain the original record array and
  delegate to the ordinary Codex parser before merging envelope attachments;
  the extension does not provide a Codex page adapter.
  Native attachment turns lacking provider IDs use explicit retained ordinals
  with matching native role/text to produce private owner coordinates. The
  parser refuses absent or conflicting evidence instead of inventing an ID.

- Detection is shape-based and ordered by declared tightness, per payload mode.
- Parsing preserves structured tool-result outcome and exit-code fields;
  prose is not an outcome oracle.
- Parser inference cannot overwrite a hook-authoritative lineage edge: when
  `_authoritative_parent_claim` returns a hook-asserted parent, the write
  uses that parent for lineage resolution. A hook parent with no parser parent
  promotes the session to `SessionKind.SUBAGENT`
  (`_prepared_message_context` in
  `polylogue/storage/sqlite/archive_tiers/write.py:1655-1698`).
- Replaying identical normalized content is idempotent by content hash;
  user metadata does not alter import identity.
- All ordinary ingest, replay, and reindex paths share the parsed-session
  write choke point (`write_parsed_session_to_archive` in
  `polylogue/storage/sqlite/archive_tiers/write.py:2119`).
- Batch ingest keeps source membership and precedence checks read-only:
  `_core.py` opens one read-only `source.db` handle per batch, and
  `revision_authority_refuses_write` reads `raw_session_memberships` through
  it, while index publication and later blob-publication receipt consumption
  each open their own archive-root-bound write connection
  (`_process_ingest_batch_sync` in
  `polylogue/pipeline/services/ingest_batch/_core.py:3528-3543`;
  `revision_authority_refuses_write` in
  `polylogue/storage/sqlite/archive_tiers/ingest_precedence.py:182-277`;
  `_open_sync_connection` in
  `polylogue/pipeline/services/ingest_batch/_core.py:215-245`). After index
  commit, `_process_ingest_batch_sync` opens the source-tier transaction with
  `archive_root=archive_root` and calls `consume_blob_publication_receipt`
  for each pending attachment receipt
  (`polylogue/pipeline/services/ingest_batch/_core.py:3656-3673`;
  `polylogue/storage/blob_publication.py:553-564`).

## Gotchas

Provider fixtures are not interchangeable with live captures: acquisition
identity and parser shape can differ. Check the provider completeness matrix
before claiming a mode is supported (`polylogue/sources/provider_completeness.py:1-100`).
When adding a detector, add a real-shaped fixture, precedence coverage, and
replay-parity verification. When changing normalized fields, inspect every
lowering reader and the corresponding schema checks.

## Navigation

Start with `polylogue/sources/origin_specs.py` for the declared detector
bindings and tightness, then `polylogue/sources/dispatch.py` for payload
lowering, then the provider parser and its fixture. Follow the parsed object
into `polylogue/storage/sqlite/archive_tiers/write.py`; do not infer the
durable contract from a surface serializer. The provider guides under
`docs/providers/` explain format-specific caveats.

## Settled decode and partial admission

Live intake and retained census share `terminal_decode_evidence`: a known-provider
JSON document or complete JSONL record that cannot decode settles as
`terminal_corrupt_input`. Unknown-provider decode failures retain their distinct
terminal token. Wrapped source I/O failures and parser defects remain retryable.
Changed source bytes form a new observation rather than retrying the same refusal.

A stable JSONL capture with an unfinished final record admits its complete prefix.
`PartialAdmission` records `truncated_tail`, the complete-record count, the prefix
byte offset and the full acquired size. Acquisition seals the count before writer
entry. Intake, batch byte accounting and dispatcher events expose the partial;
the attempt carries `batch:partial_admission` and its event is degraded. No-session
and settled corrupt observations remain excluded. The literal raw retains the
unfinished tail so a later completed observation can advance normally.
