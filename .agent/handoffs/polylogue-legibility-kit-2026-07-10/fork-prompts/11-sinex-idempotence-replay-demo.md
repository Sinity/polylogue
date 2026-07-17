# Fork prompt 11 — Implement Sinex “Import It Twice” and replay contrast

Use the uploaded Sinex repository and prior analysis. Implement the strongest bounded slice of `sinex-cem.13`, coordinated conceptually with `sinex-908` and the later honest-revision demos.

Primary claim: importing the same source occurrences twice creates zero duplicate current occurrences, while replay under a changed semantics version intentionally creates new interpretations over the same stable occurrences.

Required scenario:

- first import of a deterministic export admits N occurrences;
- second import of identical bytes admits zero new current occurrences and emits a receipt for suppressed duplicates;
- a grown export adds exactly one new occurrence and supersedes only where declared;
- the same content from another source with ambiguous identity becomes an adjudication candidate rather than automatic fusion;
- replay under semantics v2 creates new interpretation identities while preserving stable occurrence/domain identities;
- a semantic diff reports which projections changed.

The demo must teach the distinction among occurrence identity, interpretation event identity, content hash, and stable domain object identity. Do not use event UUID as the stable public object ref.

Inspect equivalence/occurrence policy, material registry, parser replay, audit archive, domain reducers, operations, and public refs. Create independent manifests and tests. If stable identity infrastructure is insufficient, produce a failing acceptance test and a precise minimal patch/design rather than faking the result.

Produce patch, tests, packet, report, and verification receipt under `/mnt/data/sinex-import-twice/`. Include a follow-on design note showing how the same substrate supports `sinex-cem.14` and `.3` without conflating them.
