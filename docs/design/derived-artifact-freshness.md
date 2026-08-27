# Derived-artifact freshness law

Status: **accepted decision** (2026-08-26)
Tracking: polylogue-ntwtk
Evidence: corpus mine 2026-08-25, covering stale and drift defects in caches,
counters, receipts, and seeded fixtures.

## Decision

Every derived artifact must carry a digest of the source evidence and
derivation semantics that produced it. A reader may use the artifact only
after recomputing the applicable digest and comparing it with the recorded
value. Missing, malformed, or mismatched provenance is **stale** and the
reader must refuse the artifact or expose an explicit degraded result. It
must never be interpreted as fresh because a counter is zero, a timestamp is
recent, or a row exists.

The digest is a binding, not a replacement for domain checks. A reader still
performs the artifact's structural, cardinality, identity, and authorization
checks. Those checks may reject an artifact whose source digest matches.

## Contract

The source digest covers the complete source projection needed to reproduce
the artifact, in deterministic order, with typed encoding and explicit
lengths or delimiters. It also covers every parser, materializer, schema, or
recipe fingerprint that changes the meaning of that projection. A digest
must not include mutable output state, wall-clock time, or an unrelated
durable tier.

Each derived artifact records:

- the digest algorithm and source-projection identifier;
- the source digest;
- the derivation or recipe fingerprint;
- the artifact schema or materializer version;
- the verification state and, where useful, the reason for refusal.

Per-subject artifacts may bind to a subject content hash. Archive-wide
artifacts may bind to a source-evidence snapshot. Both are instances of the
same law. Timestamps, row counts, generation numbers, and file metadata are
diagnostic or supporting evidence. They are not freshness authority by
themselves.

## Reindex boundary

The source-to-index reindex path is the canonical enforcement boundary. It
must capture the source digest before replay, bind every candidate and pass
receipt to that value, recompute it after replay, and refuse promotion or
receipt consumption when the values differ. A resumed pass must use the
original source binding and recheck it before applying or promoting output.

The existing `rebuild_source_evidence_snapshot` primitive and the
`source_evidence_after` rebuild-receipt field satisfy this boundary for the
source-to-index replay route. New derived artifacts produced at that
boundary must use the same source binding instead of inventing a parallel
snapshot. The reindex route remains the place where a source change is
turned into derived-state invalidation; individual readers may additionally
perform cheaper subject-level checks.

## Reader behavior

Readers use these dispositions:

| Evidence | Disposition |
| --- | --- |
| Digest and derivation binding match, structural checks pass | `valid` |
| Digest differs or required provenance is absent | `stale` and refuse |
| Binding is malformed, unverifiable, or the source cannot be read | `unknown` and refuse |
| Digest matches but domain checks fail | `invalid` and refuse |

Refusal is typed and observable. It may schedule ordinary convergence debt
when the artifact is retryable, but it must not silently repair, substitute a
zero, or label the old artifact ready. A fresh rebuild is the only authority
for replacing a stale derived artifact.

## Scope and rollout

This decision applies to rebuildable and disposable derived state, including
indexes, caches, counters, receipts, seeded fixtures, and materialized
insights. Durable source evidence and user-authored assertions are inputs to
the contract, not derived artifacts to be invalidated by it. Audit receipts
remain append-only evidence; any derived projection of them carries its own
binding.

Existing provenance mechanisms are retained and aligned rather than
duplicated: session-insight content-hash bindings, FTS freshness records,
embedding generation authentication, and reindex source snapshots are
current adopters with different source projections. Each future migration
must name its projection, digest owner, reader refusal path, and lifecycle
classification. A metadata-only addition is insufficient when old rows
cannot be proven fresh; those rows remain stale until the owning rebuild
route regenerates them.

This decision does not authorize a broad schema migration or a manual cache
repair command. Follow-up implementation slices own each artifact family,
with the reindex boundary providing the shared source-binding contract.
