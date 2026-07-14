# Polylogue on Sinex

## Status

The full Sinex-backed architecture described here is a target, not the current implementation.

Polylogue is, and permanently remains, SQLite-native: standalone SQLite is a first-class, supported product mode, not a deprecated migration path (operator directive, 2026-07-13). `polylogue.toml`'s `[sinex] mode` selects the Sinex-backed authority profile and defaults to `off`, which performs zero Sinex transport work and creates no durable publication obligations.

`polylogue.sinex` (polylogue-303r.2) is the first real Polylogue-side publication producer: a durable `source.db` obligation ledger (`sinex_publication_obligations`, mode `mirror`/`primary`, idempotent by protocol version + revision id + manifest digest), a transport contract modeled on Sinex's documented `DurableEmissionReceipt` (sinex-r6d.11) and `RawEnvelopeSettlement` (sinex-r6d.12) primitives, and a `PublicationService` that stages obligations in the same transaction as the evidence they cover and only reports a revision confirmed when a receipt actually unlocks progress. Its tests exercise the service against an in-process, contract-faithful reference transport (`LocalReferenceTransport`) rather than live Sinex JetStream: as of this package landing, Sinex's own consumer implementation for this exact contract (sinex-4j2.1.1) has not merged, and sinex-r6d.11 itself — the receipt primitive this contract targets — is still open upstream. **No production code path is wired to either transport yet**: setting `[sinex] mode = "mirror"` (or `"primary"`) in `polylogue.toml` today has zero effect on real archive writes — no ingest, daemon, or CLI call site constructs a `PublicationService` from that setting — and `polylogue config --format json` surfaces this explicitly as a `sinex_mode_not_yet_wired` diagnostic. Wiring a real deployment to either the reference transport or live Sinex transport is follow-up work, not something this package's landing completes unilaterally. See `polylogue/sinex/__init__.py` for the full scope note and `tests/unit/sinex/` for the durability/idempotency/receipt-barrier proof this package carries today.

Before `polylogue.sinex` landed, Polylogue could at most emit a low-volume bridge event to Sinex containing session metadata such as identity, origin, content hash, message count, model, and optional cost — the current package supersedes that as the intended producer, though it does not yet make Sinex the authority for complete transcript content or Polylogue user state end-to-end (that requires the remaining 303r.1/.4/.5/.6 phases plus the live Sinex counterpart above).

The maximal target is deliberately stronger:

> **In Sinex-backed mode, Sinex stores the durable provider-native and normalized transcript evidence, Polylogue-domain history, judgments, lifecycle, context deliveries, and model effects. Polylogue remains the AI-work ontology, normalization/composition kernel, query layer, and product. SQLite remains the standalone store or a local/offline projection and outbox.**

This direction supersedes a metadata-only bridge as the ultimate architecture. It does not imply that the implementation or rebuild proof already exists.

## Division of responsibility

| Concern | Polylogue | Sinex |
|---|---|---|
| Provider formats | Parsing and AI-session normalization semantics | Durable provider-native material |
| Sessions, messages, blocks, tools | Domain authority and product behavior | Durable event/material backend and scalable projections |
| Forks, continuations, subagents, compaction | Logical composition semantics | Persistence of typed relationships and replay history |
| Physical versus logical accounting | Domain calculations and user-facing views | Durable usage observations and projection substrate |
| Assertions, lessons, judgments, context policy | AI-work lifecycle and UX | Durable authority, retention, and cross-domain judgment substrate |
| Context compilation | Selection, trust, budget, omissions, delivery manifest | Ambient evidence, coverage state, and model-effect ledger |
| Terminal, Git, browser, filesystem, desktop | Correlation and rendering when relevant | Source and evidence authority |
| Standalone operation | SQLite authority | Not required |
| Backed operation | SQLite edge projection, cache, UI state, offline outbox | Durable authority |

The rule is:

> **Sinex owns durable persistence and evidentiary lifecycle. Polylogue owns AI-work semantics and product behavior.**

## Why store both raw and normalized transcripts

Storing only provider-native files would preserve bytes but force every consumer to understand unstable vendor formats. Storing only normalized messages would prevent auditing and reinterpretation when parsers change.

The target therefore retains:

1. provider-native archives, rollout files, browser payloads, hook records, attachments, and usage exports;
2. immutable, bounded Polylogue-normalized segments;
3. durable observations that point to exact material records;
4. revision manifests that expose complete versus partial imports;
5. rebuildable PostgreSQL and SQLite projections.

## Identity contract

Sinex event IDs identify a particular admitted interpretation and may change on replay. Polylogue refs must remain stable across replay and resegmentation.

The combined model needs separate identities for:

- stable Polylogue domain object;
- domain content/semantic revision;
- Sinex interpretation event;
- exact source-material occurrence.

A content hash, byte offset, or Sinex event UUID alone is not a stable message identity.

## Lineage is not derivation provenance

A Polylogue continuation, fork, copied prefix, subagent, or compaction boundary is a domain relationship. Sinex derivation provenance means that one interpretation was computed from upstream events.

The integration must not encode session topology as derivation ancestry. This is essential for replay, logical accounting, and safe deletion of shared-prefix evidence.

## Complete revision visibility

A transcript revision can contain thousands of related records. The target ingress protocol needs:

- a revision or batch ID;
- a manifest and content digests;
- expected material/event counts;
- settlement state;
- a final commit frontier;
- retry and partial-state semantics.

A reader should see either the previous complete revision or the new complete revision, not a half-imported session presented as complete.

## PostgreSQL and SQLite are complementary

A backed deployment can maintain Polylogue-domain PostgreSQL projections for shared/server reads while retaining SQLite for:

- standalone operation;
- offline use;
- local FTS and vector acceleration;
- downloaded material cache;
- UI state;
- projection watermarks;
- a durable offline-write outbox;
- deterministic tests and demos.

No irreplaceable transcript, judgment, handoff, or context-delivery state should remain silently SQLite-only in backed mode.

## Decisive proof

The architecture becomes real when the following succeeds:

```text
1. Import provider-native and normalized material into Sinex.
2. Use Polylogue normally and record stable refs.
3. Delete every rebuildable Polylogue SQLite tier.
4. Run a rebuild from Sinex.
5. Compare sessions, messages, blocks, topology, usage, assertions, context deliveries, and refs.
6. Explain every difference.
7. Rerun the flagship demos.
```

Until that proof exists, public wording must say “target,” “program,” or “Sinex-backed direction,” not “current backend.”

## Roadmap authority

The owning Sinex program is `sinex-4j2`. Related Polylogue decisions that make a metadata-only bridge permanent should be superseded by a new Beads decision under the maximal direction. GitHub Issues are not roadmap authority for either project.
