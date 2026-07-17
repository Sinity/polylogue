# Fork prompt 13 — Prototype stable identity and Sinex-backed rebuild

Use both uploaded repositories and prior analysis. Focus narrowly on the hardest impedance mismatch: Sinex event IDs identify replay-specific interpretations, while Polylogue refs require stable session/message/block/assertion/context identities across replay and resegmentation.

Design and implement a reference prototype for:

- stable Polylogue domain object identity;
- domain revision identity;
- alias/reconciliation records;
- mapping to one or more Sinex interpretation event IDs;
- mapping to exact provider-native and normalized material anchors;
- stable refs that survive a parser v1→v2 replay;
- a projection watermark and local SQLite rebuild cursor;
- a minimal `rebuild from Sinex` path for a deterministic subset: sessions, messages, blocks, one topology edge, one assertion, and one context delivery.

Do not use content hash, mutable byte offset, or Sinex event UUID alone as stable identity. Use immutable normalized segments or record identities so reserialization does not invalidate refs.

The prototype can be schema-first if full runtime integration is too broad, but it must include executable tests or a small harness proving:

1. identical occurrence import resolves to the same stable object;
2. changed parser semantics creates a new interpretation/revision as appropriate;
3. stable public ref still resolves;
4. deleting rebuildable SQLite state and replaying the deterministic Sinex fixture reconstructs equivalent domain objects;
5. intentionally local UI state is excluded from parity;
6. ambiguity routes to reconciliation rather than silent merge.

Produce patches or prototype code in both repos, schema/DDL, tests, parity report format, and a Beads-ready handoff under `/mnt/data/interop-identity-rebuild/`.
