---
created: 2026-07-16
purpose: Decide rebuild equivalence and canonical freshness identity for FTS, embeddings, and insights
status: recommended-decision
project: polylogue
---

# Derived freshness

## Decision

A derived value is current if and only if its exact `DerivationKey` matches the
current source identity and recipe identity, and its successful result belongs
to the active generation. Generation, counts, timestamps, and boolean
`needs_reindex` flags are not sufficient currentness proofs.

Define one small typed derivation protocol and receipt vocabulary. Keep
domain-specific storage ledgers because FTS, embeddings, and insight
materializations have different grains and integrity checks. Do not create one
universal derivation table.

## Derivation key

```text
DerivationKey = {
  subject: logical item and grain,
  source_identity: exact input content/version identities,
  recipe_identity: implementation/schema/model/config identities that affect output,
  output_contract: type, dimensions/schema, canonicalization,
}
```

An attempt additionally carries `generation`, `attempt_id`, resource/producer
identity, start/end state, and result integrity hash. `output_hash` validates a
result; it is not the request lookup key.

Privacy, authorization, retention, and deletion eligibility remain lifecycle
metadata, not computational identity. This matches the model-effect distinction
in `polylogue-303r.7`.

## Shared state transitions

- source or recipe identity change creates `pending(new_key)`;
- work captures the exact key and generation before computation;
- success/error may update state only if the current pending key and generation
  still match;
- a superseded worker records its receipt but cannot clear newer debt;
- retryability is orthogonal to freshness;
- rebuild, incremental, targeted, sync, and async routes call the same semantic
  derivation function and differ only in selection/scheduling;
- public compatibility fields such as `needs_reindex` are derived projections.

## Domain applications

### Embeddings

Use embeddable block/message content identity plus canonicalization, selector,
chunking version, provider, model/revision, dimensions, task/input type,
normalization, tool implementation, and input/schema version. This is the
complete recipe key described by `wmsc` and `303r.7`. Success and error writes
are conditional on the captured key/generation.

### FTS

Contentless FTS cannot prove identity from row count. Add a rebuildable
`messages_fts_identity(rowid PRIMARY KEY, block_id UNIQUE, source_hash,
recipe_id)` ledger maintained with the FTS triggers. Exact reconciliation
compares desired `(rowid, block_id, source_hash, recipe_id)` to the identity
ledger and FTS docsize/row presence. Missing, excess, reused-rowid, mismatched,
and corrupt states are distinct. Bounded startup work may repair known debt;
periodic/on-demand exact audit proves the whole set.

### Materialized insights and projections

Use the complete durable/source fact set identity relevant to each insight plus
the registered insight/materializer definition version. A session content hash
alone is insufficient when a projection also depends on user assertions,
pricing catalogs, topology, or global archive state; the owning definition must
declare those inputs. Global derivations use a frontier/snapshot identity, not
the wall-clock materialization time.

## Rebuild equivalence

For a fixed source snapshot and recipe key, incremental, targeted repair, full
rebuild, synchronous, asynchronous, and restarted execution must produce the
same canonical facts and completeness state. Operational metadata may differ
only in explicitly excluded attempt/timing/resource fields.

A fast-forward/clone rebuild is valid only when it proves source snapshot
stability, surviving schema-object and row parity, integrity, and exact recipe
transition before atomic generation promotion. Otherwise replay from durable
evidence.

## Competitive alternatives

| Alternative | Advantage | Why not chosen |
| --- | --- | --- |
| Boolean `needs_reindex` | Simple | Cannot say which source/recipe made it stale and races newer work |
| Row/count parity | Cheap | Rowid reuse and compensating missing/excess rows pass |
| Timestamp freshness | Easy to inspect | Clocks do not identify semantic inputs or recipes |
| Global schema version only | Catches structural changes | Misses model, canonicalization, catalog, selector, and content changes |
| One universal derivation ledger | Uniform queries | Forces incompatible grains/lifecycles together and becomes a framework |
| Always full rebuild | Strong end state | Too expensive for convergence and still needs recipe/currentness identity |

## Migration sequence

1. Land the shared typed `DerivationKey`/attempt predicate through embeddings,
   the strongest existing consumer.
2. Add the FTS identity ledger in the next batched index schema generation.
3. Adopt the protocol for one insight whose dependencies exceed session content.
4. Derive compatibility flags; remove independent stale predicates only after
   parity and mutation proof.
5. Keep derived-tier rollback generations until postflight proves the new key.

## Required proof

- change each computational key field individually and prove reuse is denied;
- change only eligibility metadata and prove computational identity is stable
  while authorization can still deny reuse;
- force old workers to finish after a new generation and prove they cannot mark
  current;
- construct equal-count FTS sets with missing/excess/reused rowids and detect all;
- compare all derivation routes on the same seeded snapshot;
- mutate one rebuild step or stale predicate and retain historical `wmsc`,
  `1xc.12`, `1dk1`, and usage-materialization witnesses.

Primary evidence: `polylogue-wmsc`, `polylogue-1xc.12`, `polylogue-303r.7`,
`polylogue-f2qv.5`; embedding materialization/write modules, FTS DDL/triggers,
and insight materialization registries.
