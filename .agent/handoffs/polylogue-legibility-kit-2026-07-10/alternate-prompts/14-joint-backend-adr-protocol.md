# Fork 14 — Maximal Sinex-backed Polylogue ADR and wire protocol

Work across the supplied Polylogue and Sinex repositories. Use Beads as roadmap authority, especially `sinex-4j2` and the corresponding Polylogue integration decisions. Explicitly supersede any doctrine that says Sinex must ultimately remain metadata-only.

## Target authority

- Sinex is the canonical durable backend for provider-native transcript material, immutable normalized transcript material, durable transcript-domain history, judgments, lifecycle, and model effects.
- Polylogue owns provider normalization semantics, AI-work ontology, lineage and compaction, physical/logical accounting, reviewed memory, context compilation, query behavior, and product UX.
- SQLite remains Polylogue’s standalone backend and local/offline edge projection.
- Beads remains work-intent authority.

## Mission

Write an implementation-grade cross-repository ADR and versioned protocol skeleton. Do not implement the full data plane, but remove ambiguity sufficient for multiple implementation agents to proceed safely.

## Required design

1. Authority matrix by data class.
2. Stable object identity, domain revision identity, Sinex interpretation event identity, and material occurrence refs.
3. Provider-native and normalized material descriptors with multiple digests and canonicalization versions.
4. Immutable segment and transcript-revision manifest format.
5. Compact event vocabulary for session/message/tool/subagent/compaction/usage/assertion/judgment/context-delivery history.
6. Domain-local ordering fields.
7. Bundle settlement, expected counts/digests, and complete-revision frontier.
8. PostgreSQL projection ownership and SQLite replica/outbox behavior.
9. Replay and stable-ref semantics.
10. Privacy classes, raw-text capability, generic MCP redaction, and model-input policy.
11. Deletion and derived-cascade behavior, carefully distinguishing Polylogue topology from Sinex derivation provenance.
12. Shared model-effect/embedding recipe identity.
13. Standalone versus Sinex-backed mode and no-dual-master rule.
14. Full `polylogue rebuild --from-sinex` parity proof.

## Owned scope

Own new ADR/schema/example packet files and narrow comments in the current thin bridge. Avoid changing production runtime behavior unless needed to make the existing contract identify itself honestly as transitional.

## Validation

Validate schemas against examples, run Markdown/link checks, and produce at least three worked traces:

- initial transcript revision;
- append/resume plus replay under new parser semantics;
- offline user assertion reconciled into Sinex and back to SQLite.

## Deliverables

Produce patches for both repositories, protocol schemas/examples, a sequence diagram, migration phases, explicit superseded decisions, validation output, and a list of Beads that should be created or rewritten.
