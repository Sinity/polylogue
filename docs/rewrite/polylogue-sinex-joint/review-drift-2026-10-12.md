# Drift review of the joint decisions, 2026-10-12

Each row of `decisions.md` (1-25) was compared with the current text of both specs and the joint prototypes, by grepping both specs for the row's terms and reading the owning sections. Sinex-side drift was fixed in place. Polylogue's spec was not edited.

Counts: 21 consistent, 3 drifted on the Sinex side (all fixed), 0 drifted on the Polylogue side, 1 superseded but unmarked (annotated).

| Row | Status | Evidence | Action |
|---|---|---|---|
| 1 Division of work | consistent | `sinex-spec/spec/model.md` §10; `polylogue-spec/decisions/sinex.md` "What each side keeps" | none |
| 2 Excision is a true purge | superseded, unmarked | #6 maps Polylogue excision onto tombstones and crypto-shredding with the byte purge as fallback (`sinex-spec/spec/storage.md:609`, `polylogue-spec/decisions/sinex.md` Deletion) | annotated in `decisions.md` |
| 3 What Polylogue sends | drifted, Sinex | `sinex-spec/catalog/taxonomy.toml` had one kind `polylogue.record` with a `record_kind` field for everything; `model.md` §10, Polylogue's `decisions/sinex.md` and the contract test use one kind per object | catalog: `polylogue.input`, `.session`, `.message`, `.revision`, `.turn`, and `.record` for archive.db rows only; binding kinds updated; 100 kinds (counts fixed in `catalog/README.md`, `DIGEST.md`) |
| 4 Identity | drifted, Sinex | same cause as row 3: citations are `sinex:eq/polylogue/<kind>/<id>` per kind, which a single catalog kind could not express | fixed with row 3 |
| 5 Revisions, lineage | consistent | `model.md` §10; contract test admits revisions whole | none |
| 6 Purge (Revision 2) | consistent | `storage.md` §9; contract test's purge orders | none |
| 7 Projections and proof | consistent | `sinex-spec/plan/slices.md` slice 6; Polylogue slice 13 | none |
| 8 Transport | consistent | O1 keeps NATS as ingress (`decisions/event-log.md`); `polylogue-spec/decisions/sinex.md` Transport | none |
| 9 Shared crates | consistent | `sinex-spec/plan/workspace.md:15`; the `hashed-blobs` prototype has no tokio, rusqlite, sqlx or async-nats | none |
| 10 Sequencing | consistent | `slices.md` slice 6 "right after Polylogue is in daily use"; Polylogue slice 13 "right after slice 4" | none |
| 11 Content store | consistent (with #12, #14 annotations) | excision writes new blobs with same-length filler on both sides (`storage.md:484`, `polylogue-spec/spec/storage/archive-db.md:334`) | none |
| 12 Amendment to #11 | consistent | deltas are patch-from on both sides (`storage.md`, `polylogue-spec/spec/storage/blobs.md`); the decision does not name the delta method | none |
| 13 xtask core | consistent | `sinex-spec/plan/workspace.md:178, :232` | none |
| 14 Digest | consistent | `blake3:<hex>` in both specs and in the crate's `Display` | none |
| 15 Logical day | consistent | default 04:00 in `analysis.md:174`, `query.md:81` and Polylogue's `query.md:123`, `glossary.md:126` | none |
| 16 Annotations and claims | consistent | `holds`/`fails`/`unknown`/`unchecked` in both glossaries | none |
| 17 read_as | consistent | both `query.md` and glossaries | none |
| 18 One brief | consistent | 25% share and 100 ms deadline in `polylogue-spec/spec/user-data.md:336-337` and `sinex-spec/spec/surfaces.md` §3.2 | none |
| 19 Sealed blobs | consistent (#24 annotation) | the crate's `put_sealed` takes a level and a `DataKey` that carries `key_ref`, a refinement of `put_sealed(bytes, key_ref)` per #12's level per write | none |
| 20 AI-session fusion | drifted, Sinex | `sinex-spec/spec/derivation.md` §7.3.1 said "cross-provider AI sessions", while the decision fuses one session seen through several routes and never fuses different origins; `fusion.toml` itself matched on the origin-prefixed ID | prose fixed in two places; the fusion rule's inputs are now `polylogue.session` and `polylogue.message` |
| 21 Visual language | consistent | Polylogue's `polylogue-visual-tokens` check compares its copy with `visual-tokens.css` | none |
| 22 Turns as spans | consistent | fields in `taxonomy.toml` and `polylogue-spec/decisions/sinex.md:73` | none |
| 23 Sealed frame format | consistent (#24 annotation) | `hashed-blobs` and PT10 use the keyed name in HKDF info and AAD | none |
| 24 Keyed names | consistent | `sinex-spec/decisions/privacy.md`; `hashed-blobs` | none |
| 25 Shared pacing | consistent | both say 20% IO pressure and 1 s syncs (`performance.md`, `derivation.md` §2.1, `polylogue-spec/spec/behavior.md:170`). The event-path prototype implements the rate cut as a hold of the slow commit's length; its README says so | none |
