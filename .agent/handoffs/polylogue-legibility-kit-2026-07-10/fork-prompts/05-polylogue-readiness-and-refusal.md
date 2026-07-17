# Fork prompt 05 — Make Polylogue degrade loudly

Use the uploaded Polylogue repository and prior analysis. Implement the smallest coherent, reusable slice of `polylogue-avg` and `polylogue-cpf.4` needed for public demos and ordinary reads to distinguish:

- complete;
- degraded;
- not supported;
- timed out;
- truncated;
- stale/not current;
- incomplete projection or missing modality.

Do not attempt a global health-system rewrite. Start from actual result/view envelope contracts and the existing honesty anti-demo. Ensure the demo/read paths used by The Receipts, Count It Once, findings, and context can carry one bounded machine-readable and human-visible signal instead of returning a normal-looking empty or partial answer.

Required adversarial cases:

1. missing paired tool evidence;
2. stale FTS or derived projection;
3. query timeout;
4. result truncation;
5. unsupported cross-source reconstruction;
6. missing provider modality;
7. incomplete import/revision.

The implementation must be fail-closed where a false complete result would be misleading, but avoid turning every warning into a hard error. Define precedence when several states apply and include evidence/coverage details.

Use existing readiness, derived-status, surface-envelope, anti-demo, and daemon/query contracts. Add targeted tests and one deterministic anti-demo packet proving that refusal is a successful product behavior.

Do not close broad Beads unless their full acceptance criteria are met. Produce a patch, a state vocabulary/design record, tests, a packet, and a verification receipt under `/mnt/data/polylogue-readiness-refusal/`.
