# Material protocol v1

Status: v1 defined and implemented in this repository (`polylogue-303r.1`).
Cross-repository (Sinex) implementation tracked separately (`sinex-4j2.1`,
`sinex-4j2.1.1`) — not yet landed as of this writing, so "checked-in bytes
match Sinex" is proven on the Polylogue side only until that lands.

Implementation: `polylogue/material_protocol/v1/`. Checked-in fixture:
`tests/fixtures/material_protocol/v1/small-session/`. Tests:
`tests/unit/material_protocol/v1/`.

## What this is

Polylogue's public, Sinex-independent wire format for a normalized session
revision: **bounded immutable NDJSON segments plus a revision manifest**.
Given only the manifest and the segment files it names, a reader can
reconstruct a session's messages, blocks (including tool calls/results),
lineage edges, compactions and other session events, attachment references,
usage, and known fidelity gaps — **without an archive DB**. Exact
provider-native artifacts and attachment bytes stay in separate Sinex
materials; this protocol carries references to them, not the bytes.

Encode/decode/verify/segmentation/anchor-resolution helpers only. Transport,
durable publication, and Sinex settlement belong to `polylogue-303r.2`.

## File layout

A revision directory contains:

```
manifest.json
segments/
  seg-00000.ndjson
  seg-00001.ndjson
  ...
```

`manifest.json` is one canonical JSON object (see `RevisionManifest` in
`polylogue/material_protocol/v1/manifest.py`). Each `seg-NNNNN.ndjson` is a
sequence of canonical JSON objects, one per line (`\n`-terminated), UTF-8.

## Canonical framing (byte-stability)

Every JSON value in this protocol — each record and the manifest — is
serialized with exactly one rule: recursively NFC-normalize every string,
then serialize with **sorted object keys** and no incidental whitespace
(`polylogue/material_protocol/v1/canonical.py`, backed by
`orjson.OPT_SORT_KEYS`). Key order therefore never carries meaning. This is
what makes "decode, then re-encode" byte-identical to the original, and what
lets two independent encoders (Polylogue, Sinex) produce identical bytes for
identical input.

## Records

Every record is a JSON object with at least `kind`, `record_id`, and `seq`.
`record_id` formulas mirror `index.db`'s generated columns exactly:

- `session`: `record_id = session_id = "{origin}:{native_id}"`
- `message`: `record_id = message_id = "{session_id}:{COALESCE(native_id, position||'.'||variant_index)}"`
- `block`: `record_id = block_id = "{message_id}:{position}"`
- `attachment`: `record_id = "{message_id}:attachment:{position}"`
- `lineage`: `record_id = "{session_id}:lineage:{dst_origin}:{dst_native_id}:{link_type}"`
- `usage`: `record_id = "{session_id}:usage:{model_name}"`
- `session_event` (covers compaction, via `event_type="compaction"`, and any
  other typed fact the archive records): `record_id = "{session_id}:{position}"`

`seq` is a single global, strictly increasing counter across the whole
revision (starting at 0), in this fixed walk order — this **is** the
manifest's `sequence_rule`:

1. the `session` record
2. `lineage` records, sorted by `(dst_origin, dst_native_id, link_type)`
3. `usage` records, sorted by `model_name`
4. for each message, in transcript order (`position`, `variant_index` —
   **not** re-sorted by timestamp, because equal or missing timestamps can't
   total-order a transcript on their own):
   - the `message` record
   - its `block` records, in block position order
   - its `attachment` records, in attachment position order
   - `session_event` records whose source message is this message, in event
     position order
5. finally, any `session_event` records with no matching source message, in
   event position order

This order is deliberately append-friendly: growing a session by appending
trailing messages (the common live-watcher case) only ever adds records
after every previously emitted one.

Tool call/result correlation is carried structurally: a `tool_use` block and
its `tool_result` block share `tool_id`; `tool_result_is_error` /
`tool_result_exit_code` on the result block are provider-reported outcomes
(`NULL`/`None` = unknown), matching `blocks.tool_result_is_error` /
`tool_result_exit_code` in `index.db`. Pairing multiple same-`tool_id` uses
(the `actions` view's rank-based pairing) is a downstream read-model concern,
not part of this protocol.

## Revision manifest

Key fields (`RevisionManifest` in `manifest.py`):

- `protocol_version`, `semantics_version` — this document's version and the
  Polylogue domain-semantics version the encoder implements.
- `origin_vocabulary_version` / `origin_vocabulary_digest` — pins the public
  `Origin` enum (`polylogue.core.enums.Origin`) this revision was encoded
  against. See [Origin vocabulary pinning](#origin-vocabulary-pinning).
- `session_id`, `origin`, `native_id` — the session identity.
- `revision_id` — `content_digest.polylogue_sha256`: the SHA-256 of every
  sealed segment's bytes concatenated in segment-index order. Two encoders
  that produce the same canonical bytes get the same `revision_id`.
- `superseded_revision_id` — the prior revision this one replaces, or
  `null` for a first revision.
- `content_digest` — a multi-digest descriptor (`ContentDigest`): Polylogue's
  own SHA-256, canonicalizer version, size, media type, and optional
  `sinex_cas_digest` / `provider_digest` slots for cross-system digests.
  **None of these is the domain object id** — `session_id` is.
- `segments` — one `SegmentDescriptor` per sealed segment: index, filename,
  SHA-256, size, record count, and the `[first_seq, last_seq]` range it
  covers.
- `expected_record_counts` — per-`kind` record counts across the whole
  revision.
- `anchors` — `record_id -> AnchorEntry {segment_index, line_index, seq,
  kind, sha256}`. This is the citation mechanism: given only a `record_id`,
  `resolve_anchor()` reads exactly the one named segment, extracts one line,
  and verifies it hashes to the declared `sha256` before returning it — no
  full scan required.
- `sequence_rule` — the fixed ordering rule described above, recorded as
  prose so a reader that hasn't read this document can still see the
  contract.
- `completeness` — currently always `"complete"` for an encoded revision
  (partial/degraded states are 303r.2's transport concern, not this
  protocol's).
- `fidelity_gaps` — a list of `FidelityGap {scope, record_id, gap_kind,
  detail}` entries: known incompleteness the encoder is aware of (e.g. an
  attachment whose bytes were never fetched, a message with no timestamp).
  This is metadata *about* the revision, not bulk transcript text.

## Segmentation and immutability

Segments are bounded (`max_records_per_segment`, default 500) and, once
sealed, immutable: their bytes never change. A session that grows (the
common case — a live watcher appending messages) gets a **new revision**
that:

- reuses every prior segment's exact bytes/descriptor unchanged, and
- appends only the new trailing records into new segment(s) after the prior
  ones.

`encode_appended_revision()` implements this: it verifies the new material's
ordered record list reproduces the prior revision's records as an exact
prefix (raising `NotAnAppendError` otherwise), then only encodes the tail.
Every anchor that existed in the prior revision is byte-identical in the new
one — same `segment_index`, `line_index`, `sha256`.

A **regenerated** revision (the provider file was reparsed from scratch, not
just appended to — a resegmentation/edit, not a pure append) is encoded fresh
via `encode_session_revision(..., superseded_revision_id=prior.revision_id)`.
Its segmentation and anchors are entirely its own; the prior revision's
segment files are not touched, read, or referenced by content, only linked
via `superseded_revision_id`. Stable cross-revision reference bridging for
this case (so a citation into the old revision keeps resolving after a
resegmentation) is `polylogue-303r.4`'s scope, not this protocol's.

## Verification (fail closed)

`verify_revision(manifest, segment_bytes)` runs a full compatibility pass and
raises a `MaterialProtocolError` subclass (never returns a bare boolean) on
any of:

| Failure | Exception |
| --- | --- |
| a required segment file wasn't supplied | `SegmentMissingError` |
| a segment's bytes don't hash to its declared `sha256` (covers any byte change, including a removed/reordered/injected record) | `DigestMismatchError` |
| a segment's declared `record_count`/`size_bytes` don't match reality | `RecordCountMismatchError` / `DigestMismatchError` |
| the revision's overall content digest doesn't match `content_digest.polylogue_sha256` | `DigestMismatchError` |
| records aren't observed in strict `seq` order starting at 0 | `SequenceOrderError` |
| per-`kind` counts don't match `expected_record_counts` | `RecordCountMismatchError` |
| the manifest's `anchors` don't exactly match what a full scan finds (missing, extra, or any field different) | `AnchorMismatchError` |
| `origin_vocabulary_version`/`origin_vocabulary_digest` isn't a known, matching pair | `UnknownOriginVocabularyError` |

`resolve_anchor(manifest, segment_bytes, record_id)` is the narrow,
single-record version of the same integrity check (segment/line lookup +
hash verification), for citing one record without a full-manifest pass.

## Origin vocabulary pinning

`origin_vocab.py` vendors the public `Origin` enum's value set as a
frozen `(version -> sha256 digest)` registry
(`KNOWN_ORIGIN_VOCABULARIES`). `resolve_current_origin_vocabulary()` (used by
the encoder) asserts the live-computed digest of the current `Origin` enum
still matches the frozen entry for `CURRENT_ORIGIN_VOCABULARY_VERSION` — if
someone adds/renames/removes an `Origin` member without bumping the version
and freezing a new digest,
`tests/unit/material_protocol/v1/test_origin_vocab_pinning.py` fails, by
design. `check_origin_vocabulary()` (used by the decoder/verifier) rejects
any `(version, digest)` pair that isn't in the registry —
"unknown or stale vocabulary version quarantines admission", per design.

## Cross-repository fixture parity

`tests/fixtures/material_protocol/v1/small-session/` is a full-featured
synthetic session (multiple messages; a successful and a failed tool call;
two messages sharing a timestamp and one with none, ordered only by
position; a resume lineage edge; a compaction session event; an attachment
ref with unavailable bytes; a usage row; two fidelity gaps; nontrivial
Unicode) encoded once and checked in.
`tests/unit/material_protocol/v1/test_checked_in_fixture.py` proves a fresh
encode of the same input reproduces those bytes exactly. This is the
artifact the design calls "check the same synthetic fixture and digest into
both repositories" — the Sinex side (`sinex-4j2.1.1`) is expected to encode
the equivalent input and match these same bytes/digests; that cross-repo
comparison itself is not runnable from this repository alone until the Sinex
implementation lands.

## What's out of scope here (see polylogue-303r.2+)

- Publishing/transport (JetStream, durable receipts, raw-envelope
  settlement) — `polylogue-303r.2`.
- Stable cross-revision reference bridging through resegmentation/replay —
  `polylogue-303r.4`.
- Durable user-state outbox — `polylogue-303r.5`.
- Lifecycle/retention/deletion — `polylogue-303r.6`.
