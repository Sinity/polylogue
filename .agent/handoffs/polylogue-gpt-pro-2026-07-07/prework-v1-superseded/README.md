# Polylogue urgent/correctness bead static prework

Generated from the Beads export and the unpacked Polylogue source snapshot. The goal is to turn urgent and correctness-critical beads into coding-agent task packets where the remaining work is mostly implementation plus verification.

This is not a live patch. It is a static diagnosis/handoff package. Each packet contains the bead, the likely failure mechanism, files/functions to inspect, implementation shape, tests, and a verification lane. Line numbers refer to the unpacked source tree in this analysis environment: `/mnt/data/work2/polylogue/tree/polylogue`.

Execution rule: take packets in the listed order unless a lower packet becomes a blocker for the current one. For each packet, first reproduce the current failure or missing invariant with a focused test, then patch the single choke point, then run the focused verification lane. Avoid broad refactors unless the packet explicitly calls for one.

## Packet index

01. `polylogue-s7ae.6` — Classify the aborted full verification run before coordination deploy (P1, release-gate, ready-now / evidence-work) → [tasks/01_polylogue-s7ae.6.md](tasks/01_polylogue-s7ae.6.md)
02. `polylogue-37t.15` — Force non-user assertion writes through candidate/non-injected policy (P1, agent-write-safety, ready-now / code-local) → [tasks/02_polylogue-37t.15.md](tasks/02_polylogue-37t.15.md)
03. `polylogue-kwsb.1` — Daemon/capture Host, Origin, receiver-token, and spool hardening (P1, security, ready-now / code-local with extension smoke) → [tasks/03_polylogue-kwsb.1.md](tasks/03_polylogue-kwsb.1.md)
04. `polylogue-8jg9.4 + polylogue-8jg9.2` — Make ops-doctor orphan cleanup use the same lease/generation invariants as blob GC (P1/P2, blob-integrity, ready-now / code-local) → [tasks/04_polylogue-8jg9.4_plus_polylogue-8jg9.2.md](tasks/04_polylogue-8jg9.4_plus_polylogue-8jg9.2.md)
05. `polylogue-83u.4` — Separate source-tier referenced-blob debt from index attachment-acquisition debt (P1, blob-integrity, ready-now / diagnostic-code) → [tasks/05_polylogue-83u.4.md](tasks/05_polylogue-83u.4.md)
06. `polylogue-9e5.28` — Make the rigor audit iterate the full insight registry, not only existing contracts (P1, evidence-honesty, ready-now / code-local) → [tasks/06_polylogue-9e5.28.md](tasks/06_polylogue-9e5.28.md)
07. `polylogue-9e5.29` — Distinguish absent evidence from true numeric zero at field level (P1, evidence-honesty, needs-small-spec then code) → [tasks/07_polylogue-9e5.29.md](tasks/07_polylogue-9e5.29.md)
08. `polylogue-9e5.30` — Tag prose-mined forensic fields as text-derived (P1, evidence-honesty, ready-now / model-and-renderer) → [tasks/08_polylogue-9e5.30.md](tasks/08_polylogue-9e5.30.md)
09. `polylogue-cpf.5` — Propagate weakest temporal provenance through aggregates (P1, temporal-honesty, ready-now / code-local with schema propagation) → [tasks/09_polylogue-cpf.5.md](tasks/09_polylogue-cpf.5.md)
10. `polylogue-cpf.6` — Inject clock seam for relative-date parsing and audit sort_key_ms epoch pins (P1, temporal-honesty, ready-now / code-local plus audit artifact) → [tasks/10_polylogue-cpf.6.md](tasks/10_polylogue-cpf.6.md)
11. `polylogue-f2qv.2` — Normalize Codex/Claude token lanes into disjoint uncached/cache/reasoning/completion fields (P2, usage-cost-correctness, ready-now / parser+tests) → [tasks/11_polylogue-f2qv.2.md](tasks/11_polylogue-f2qv.2.md)
12. `polylogue-f2qv.1` — Make per-model rollups partition usage events instead of duplicating session totals (P2, usage-cost-correctness, ready-now / storage-rollup) → [tasks/12_polylogue-f2qv.1.md](tasks/12_polylogue-f2qv.1.md)
13. `polylogue-f2qv.4` — Use one LiteLLM-backed pricing resolver and remove tokencost/second maps (P2, usage-cost-correctness, ready-now / grep-and-contract) → [tasks/13_polylogue-f2qv.4.md](tasks/13_polylogue-f2qv.4.md)
14. `polylogue-f2qv.3` — Report API-equivalent dollars and subscription credits as separate fields (P2, usage-cost-correctness, ready-now after lanes/pricing) → [tasks/14_polylogue-f2qv.3.md](tasks/14_polylogue-f2qv.3.md)
15. `polylogue-f2qv.5` — Version-gate provider-usage projection so stale rollups self-heal (P2, usage-cost-correctness, ready-now / convergence-path) → [tasks/15_polylogue-f2qv.5.md](tasks/15_polylogue-f2qv.5.md)
16. `polylogue-20d.4` — Mirror daemon structured-query routing in CLI so non-FTS filters skip the FTS readiness gate (P2, query-correctness, ready-now / code-local) → [tasks/16_polylogue-20d.4.md](tasks/16_polylogue-20d.4.md)
17. `polylogue-1xc.12` — Add FTS drift gauges and metamorphic trigger-coherence tests with rowid-reuse protection (P2, search-integrity, spec-first then code) → [tasks/17_polylogue-1xc.12.md](tasks/17_polylogue-1xc.12.md)
18. `polylogue-83u.3` — Acquire uploaded attachment bytes in live browser capture (P1, attachment-integrity, needs-architecture-note then code) → [tasks/18_polylogue-83u.3.md](tasks/18_polylogue-83u.3.md)
19. `polylogue-83u.2` — Acquire bytes for non-inline sources while live handles are open (P2, attachment-integrity, ready-now after census/classification) → [tasks/19_polylogue-83u.2.md](tasks/19_polylogue-83u.2.md)
20. `polylogue-83u.6` — Run read-only attachment acquisition census by origin/status/bytes (P2, attachment-integrity, ready-now / read-only artifact) → [tasks/20_polylogue-83u.6.md](tasks/20_polylogue-83u.6.md)
21. `polylogue-peo` — Add daemon crash forensics, heartbeat sentinel, and restart evidence (P2, operational-resilience, ready-now / lifecycle module) → [tasks/21_polylogue-peo.md](tasks/21_polylogue-peo.md)
22. `polylogue-4be` — Create a real restore drill for backup proof (P2, backup-integrity, ready-now / devtools+ops artifact) → [tasks/22_polylogue-4be.md](tasks/22_polylogue-4be.md)
23. `polylogue-4ts.3` — Separate subagent auto-compaction from main-session compaction in lineage (P2, lineage-truth, needs-source-confirmation then parser patch) → [tasks/23_polylogue-4ts.3.md](tasks/23_polylogue-4ts.3.md)
24. `polylogue-4ts.4` — Read lineage composition from one transaction/snapshot (P2, lineage-truth, needs-source-confirmation then storage patch) → [tasks/24_polylogue-4ts.4.md](tasks/24_polylogue-4ts.4.md)
25. `polylogue-4ts.6` — Expose transcript completeness instead of silently reading truncated sessions (P2, lineage-truth, needs-source-confirmation then model patch) → [tasks/25_polylogue-4ts.6.md](tasks/25_polylogue-4ts.6.md)
26. `polylogue-b0b` — Replace keyword-only outcome/pathology heuristics with structural evidence where available (P2, evidence-honesty, spec-first then targeted code) → [tasks/26_polylogue-b0b.md](tasks/26_polylogue-b0b.md)
27. `polylogue-9e5.3` — Column-honesty census for nullable/zero/default public fields (P2, evidence-honesty, ready-now / audit-artifact) → [tasks/27_polylogue-9e5.3.md](tasks/27_polylogue-9e5.3.md)
28. `polylogue-9e5.4` — Static get-modify-put race-window audit of shared SQLite writer paths (P2, storage-correctness, ready-now / audit-artifact) → [tasks/28_polylogue-9e5.4.md](tasks/28_polylogue-9e5.4.md)
29. `polylogue-9e5.19` — Storage-layer correctness scenario family in devtools lab (P2, storage-correctness, ready-now after focused bugs) → [tasks/29_polylogue-9e5.19.md](tasks/29_polylogue-9e5.19.md)

## Suggested first implementation batch

1. `polylogue-s7ae.6` release-gate classification.
2. `polylogue-37t.15`, `polylogue-kwsb.1`, `polylogue-8jg9.4`, `polylogue-83u.4` as the trust/data-loss/security floor.
3. `polylogue-9e5.28`, `.29`, `.30`, `cpf.5`, `cpf.6` as evidence-honesty floor.
4. `f2qv.2`, `f2qv.1`, `f2qv.4`, `f2qv.3`, `f2qv.5` as cost/usage truth.
5. Query/search/storage/lineage operational correctness packets.

## Caveats

This package is static prework. Some line numbers will drift after the first patch. Runtime claims still need focused reproduction. Where a packet says `needs-source-confirmation`, the first coding-agent action should be a short `rg`/read pass to confirm the code path before editing.
