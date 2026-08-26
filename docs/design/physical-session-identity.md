# Physical session identity beneath public origin projection

Status: proposed; no migration is included in this bead.

## Finding

The index currently generates `sessions.session_id` as `origin || ':' ||
native_id`. `Origin.AISTUDIO_DRIVE` is intentionally lossy: both Gemini and
Drive provider families project to it. Therefore two raw acquisitions with the
same native id can become one physical index row before lineage or aggregation
can distinguish them. A reparse cannot split a row after the losing write.

The companion census command reads only `source.raw_sessions`, where
`capture_mode`/`detected_provider` are still available. It emits SHA-256
identifiers, not native ids, paths, or raw ids, and labels evidence high,
medium, or low. High means distinct family hints are present; it does not mean
the bytes are independently recoverable. The raw blob must still be reviewed
before any repair.

## Proposed durable model

Introduce a `PhysicalSessionKey` composed of the stable source-family (and a
source-instance namespace when an OriginSpec requires it) plus native id. Keep
`Origin` as the public projection. A compatibility alias relation maps legacy
`origin:native_id` references to zero, one, or many physical keys and exposes
`unique`, `ambiguous`, and `unresolved` outcomes. Public reads must never choose
one member of an ambiguous alias silently.

Reparse from durable raw evidence can create separate physical rows when the
family evidence is present. Historical rows with no distinguishing evidence
remain explicitly ambiguous and are not guessed or destructively rewritten.
Durable assertions and lineage links require an alias-resolution audit before
copy-forward. A backup manifest and review gate are prerequisites for any
durable migration; this bead intentionally lands neither.

## Synthetic proof

`tests/fixtures/physical_identity/two-families.json` models Gemini and Drive
with the same native id. The fixture proves two physical keys, one public
`aistudio-drive:native` reference, and an ambiguous compatibility resolution.

## Consequences

- Index identity and lineage use physical keys; public filters remain Origin.
- Aggregates can state physical versus logical grain without conflating rows.
- Legacy references remain resolvable when unique and safely stop when
  ambiguous.
- The next implementation slice must define source-instance namespacing,
  alias persistence, copy-forward of user/lineage references, and a verified
  backup manifest before changing the durable schema.
