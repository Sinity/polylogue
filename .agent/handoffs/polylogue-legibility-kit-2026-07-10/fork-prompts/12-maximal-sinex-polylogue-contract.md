# Fork prompt 12 — Specify and patch the maximal Sinex–Polylogue contract

Use both uploaded repositories and the full prior analysis. Treat the following premise as settled for this task:

> In Sinex-backed mode, Sinex ultimately stores provider-native transcript material, normalized transcript material, durable Polylogue-domain history, judgments, lifecycle, context deliveries, and model effects. Polylogue remains the AI-work ontology, parser/domain kernel, query/composition layer, and product. SQLite remains standalone authority or a local/offline projection and outbox.

Do not preserve the metadata-only doctrine as the ultimate boundary. Identify and explicitly supersede contradictory Beads or docs without rewriting history.

Produce a concrete cross-project contract covering:

- authority matrix by data class;
- stable domain object IDs, revision IDs, Sinex interpretation event IDs, and material anchors;
- provider-native and normalized immutable material segments;
- multi-digest content descriptors;
- session revision manifests;
- admitted observation vocabulary;
- batch settlement and complete-revision visibility;
- Polylogue domain relationships versus Sinex derivation provenance;
- physical versus logical transcript accounting;
- PostgreSQL domain projections and SQLite edge replica;
- offline writes and conflict handling;
- assertion/judgment/context lifecycle;
- shared model-effect recipe identity;
- public refs and cross-resolution;
- replay, supersession, deletion, shared-prefix safety, and cache/vector invalidation;
- source coverage and projection frontiers;
- security/capability boundaries for raw transcript text;
- a full `polylogue rebuild --from-sinex` acceptance proof.

Inspect `sinex-4j2` and children, current source contract, integration-authority docs, Polylogue `polylogue-6mv`, `polylogue-fs1.9`, storage tiers, assertions, context, refs, and topology. Produce:

1. versioned JSON Schemas and example payloads;
2. an architecture decision record for each repo;
3. exact Beads amendments/new children with dependencies and acceptance criteria;
4. patch-ready documentation changes;
5. the smallest safe code patch that deepens the current bridge without pretending the full architecture is complete;
6. an end-to-end phased implementation/verification plan.

Store outputs under `/mnt/data/maximal-sinex-polylogue-contract/`, with separate patches for both repos. Return links to schemas, ADRs, patches, and Beads plan.
