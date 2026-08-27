# Sources and Parsers

## Area boundary

Sources acquire bytes and identify their material source. Dispatch chooses a
parser by input shape; the pipeline normalizes provider records into parsed
sessions before the storage writer lowers them (`polylogue/sources/dispatch.py:1-120`; `polylogue/pipeline/services/ingest_batch/_core.py:1077-1131`).

## Detection and parse route

1. Acquisition records raw bytes and source metadata in `source.db`.
2. `sources.dispatch` tests detectors in tightness order. A new detector must
   be specific enough that an earlier parser cannot claim its records.
3. The selected provider parser emits normalized sessions, messages, blocks,
   tool uses, tool results, and lineage hints.
4. `write_parsed_session_to_archive` computes public origin and identities,
   writes the parsed tree, and resolves asserted parent links.
5. The daemon converger materializes FTS, embeddings, and insight read models.

## Identity vocabulary

`Provider` names the older provider-wire family at acquisition/parser/schema
boundaries. `Origin` is the public source-origin token. `Source` carries
richer acquisition identity. The mapping can be non-injective, so a parser
must not reverse a public origin into a guessed provider
(`docs/provider-origin-identity.md:1-100`).

## Invariants

- Detection is shape-based and ordered by tightness.
- Parsing preserves structured tool-result outcome and exit-code fields;
  prose is not an outcome oracle.
- Parser inference cannot overwrite a hook-authoritative lineage edge
  (`polylogue/storage/sqlite/archive_tiers/write.py:3890-3949`).
- Replaying identical normalized content is idempotent by content hash;
  user metadata does not alter import identity.
- All ordinary ingest, replay, and reindex paths share the parsed-session
  write choke point.

## Gotchas

Provider fixtures are not interchangeable with live captures: acquisition
identity and parser shape can differ. Check the provider completeness matrix
before claiming a mode is supported (`polylogue/sources/provider_completeness.py:1-100`).
When adding a detector, add a real-shaped fixture, precedence coverage, and
replay-parity verification. When changing normalized fields, inspect every
lowering reader and the corresponding schema checks.

## Navigation

Start with `polylogue/sources/dispatch.py`, then the provider parser and its
fixture. Follow the parsed object into
`polylogue/storage/sqlite/archive_tiers/write.py`; do not infer the durable
contract from a surface serializer. The provider guides under
`docs/providers/` explain format-specific caveats.

verified: 24be873c0 2026-08-27
