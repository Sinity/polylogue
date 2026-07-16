# Fork prompt 10 — Implement Sinex “The Missing Source”

Use the uploaded Sinex repository and prior analysis. Implement or drive as far as statically possible the first flagship Sinex demo, strengthening `sinex-cem.2` with `sinex-jdp` and `sinex-60r`.

Primary claim: Sinex distinguishes a true quiet interval from an interval for which a source could not provide trustworthy evidence.

Build a deterministic fault-injection scenario with:

1. a healthy source and a genuinely empty interval;
2. a source process that is down or checkpoint-stalled;
3. a source process that is alive but emits semantically unusable empty records;
4. material acquired but parser/projection behind;
5. another source producing an event in the same interval, proving the overall query is not empty.

The result must carry source coverage, last confirmed occurrence/frontier, reason, evidence strength, and overall completeness. A normal-looking empty timeline is a failure.

Use the real source contracts, staged material, checkpoint/coverage, API/view-envelope, and demo verification paths. Avoid a shell script that bypasses the product. Create an independent expected source-sequence manifest and inject faults through supported test/sandbox mechanisms.

Produce:

- implementation patch and tests;
- demo packet with oracle, controls, and falsifier;
- human report and deterministic terminal/HTML recording source;
- exact commands and reset behavior;
- public claim wording and scope;
- Beads handoff with any uncovered substrate blockers.

Run targeted Rust/xtask tests. Respect checkout-local Postgres/NATS isolation and report actual test commands. Store outputs under `/mnt/data/sinex-missing-source-demo/`.
