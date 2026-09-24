# Sources and Parsers

## Area boundary

Sources acquire bytes and identify their material source. Detection chooses a
provider parser by input shape; the pipeline normalizes provider records into
parsed sessions before the storage writer lowers them
(`polylogue/sources/dispatch.py:1-80`; `polylogue/sources/detection.py:76-104`;
`polylogue/pipeline/services/ingest_batch/_core.py:1179-1243`).

## Detection and parse route

1. Acquisition records raw bytes and source metadata in `source.db`.
2. Detector bindings are declared by each `OriginSpec`, not by `dispatch.py`.
   `compile_detector_registry` validates them and sorts them per
   `DetectionMode` by `(mode_rank or detector_tightness, local_rank,
   binding_id)`; `CompiledDetectorRegistry.detect` returns the first predicate
   that claims the payload (`polylogue/sources/detection.py:82-104`;
   `polylogue/sources/detection.py:196-227`).
3. The selected provider parser emits normalized sessions, messages, blocks,
   tool uses, tool results, and lineage hints.
4. `write_parsed_session_to_archive` computes public origin and identities,
   writes the parsed tree, and resolves asserted parent links
   (`polylogue/storage/sqlite/archive_tiers/write.py:1108-1124`).
5. The daemon converger materializes FTS, embeddings, and insight read models.

## Detector tightness order

Lower number runs first. Tightness must be unique among executable
`OriginSpec`s, which is enforced at spec validation
(`polylogue/sources/origin_specs.py:1456-1466`). Current executable order:

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

- Detection is shape-based and ordered by declared tightness, per payload mode.
- Parsing preserves structured tool-result outcome and exit-code fields;
  prose is not an outcome oracle.
- Parser inference cannot overwrite a hook-authoritative lineage edge: when
  `_authoritative_parent_claim` returns a hook-asserted parent, the write
  replaces the parser's `parent_session_provider_id` with it and promotes the
  session to `SessionKind.SUBAGENT`
  (`polylogue/storage/sqlite/archive_tiers/write.py:846-867`).
- Replaying identical normalized content is idempotent by content hash;
  user metadata does not alter import identity.
- All ordinary ingest, replay, and reindex paths share the parsed-session
  write choke point (`polylogue/storage/sqlite/archive_tiers/write.py:1108`).
- Batch ingest keeps source membership and precedence checks read-only: the
  batch opens one read-only `source.db` handle for `raw_session_memberships`
  reads, while index publication and the later blob-publication receipt
  consumption each open their own archive-root-bound write connection
  (`polylogue/pipeline/services/ingest_batch/_core.py:2837-2852`;
  `polylogue/pipeline/services/ingest_batch/_core.py:202-212`;
  `polylogue/pipeline/services/ingest_batch/_core.py:2960-2977`).

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
