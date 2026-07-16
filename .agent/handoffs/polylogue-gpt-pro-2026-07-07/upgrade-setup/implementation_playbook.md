# Implementation playbook

This is the PR-shaped execution view of the upgraded Beads setup. The full order remains in `polylogue_beads_execution_order.csv`; this playbook tells implementers what to batch first.

## Packet 1 — verification classification
Primary bead: `polylogue-s7ae.6`. Output: full verify log, failure classification table, fixes or pre-existing references.

## Packet 2 — blob cleanup safety
Primary beads: `polylogue-8jg9.4`, `polylogue-8jg9.2`. Output: leased-blob race fixture, doctor path safe behavior, no deletion of in-flight blobs.

## Packet 3 — evidence-honest numbers
Primary beads: `polylogue-9e5.28`, `polylogue-9e5.29`, `polylogue-9e5.30`. Output: product-registry audit coverage, field-level evidence contracts, prose-derived caveats.

## Packet 4 — time honesty
Primary beads: `polylogue-cpf.5`, `polylogue-cpf.6`. Output: weakest-source propagation tests, clock seam, `sort_key_ms` audit table.

## Packet 5 — daemon/capture security
Primary bead: `polylogue-kwsb.1`. Output: central Host/Origin/token/spool controls, negative security tests, extension still working.

## Packet 6 — assertion-write safety
Primary bead: `polylogue-37t.15`. Output: all non-user writes candidate/non-injected, rejected candidate cannot be self-revived by agent.

## Packet 7 — missing blob debt
Primary beads: `polylogue-83u.4`, then `polylogue-83u.2`, `polylogue-83u.3`, `polylogue-83u.6`. Output: classification table, restored subset, irrecoverable decision record.

## Packet 8 — read algebra cutover
Primary beads: `polylogue-4p1`, `polylogue-4p1.1`, `polylogue-t46.3`, `polylogue-jnj.1`. Output: CLI/daemon/MCP parity tests.

## Packet 9 — evidence object refs
Primary beads: `polylogue-rxdo.1`, `polylogue-rxdo.2`, `polylogue-rxdo.3`, `polylogue-rxdo.4`, `polylogue-svfj`. Output: query-run/result-set/finding refs and content-hash citation resolver.

## Packet 10 — scheduler and coordination proof
Primary beads: `polylogue-37t.12`, `polylogue-37t.11`, `polylogue-s7ae.3`, `polylogue-s7ae.5`. Output: context ledger, scoped message proof, two-agent worktree demo.

## Packet 11 — variants substrate
Primary beads: `polylogue-0v9p`, `polylogue-arso`, `polylogue-rlsb`, `polylogue-d4zk`. Output: translated/simplified session fixture with alignment.

## Packet 12 — blue-green rebuild
Primary beads: `polylogue-20d.15`, `polylogue-b5l`, `polylogue-1xc.8`. Output: no partial-ready state, generation swap, resource envelope.

## Packet 13 — web evidence basket
Primary beads: `polylogue-bby.11`, `polylogue-bby.15`, `polylogue-bby.8`. Output: basket, report, export, visual smoke.

## Packet 14 — stats registry
Primary beads: `polylogue-9l5.7`, `polylogue-stc`. Output: registered measures with evidence tier, sample frame, uncertainty, confounds.

## Packet 15 — public proof
Primary beads: `polylogue-3tl.16`, `polylogue-3tl.4`, `polylogue-3tl.7`, `polylogue-cfk`. Output: claims ledger, published finding, install matrix, refreshed uplift report.

## Verification matrix

Security: Host/Origin/CSRF/token tests, capture token forgery tests, spool bounds, secret/excision tests, agent-write candidate/non-injected tests.

Data integrity: blob lease race tests, blob resolver tests, missing blob classifier, restore drill, SHA-256 byte verification, content-hash citation drift tests.

Storage: schema migration dry runs, blue-green derived-tier rebuild tests, partial-index readiness tests, WAL/resource envelope tests, large synthetic corpus test.

Query: CLI/daemon/MCP/Python parity, query grammar metamorphic tests, projection/render parity, support matrix, set algebra equivalence.

Lineage: branch matrix, shared-prefix dedup, physical/logical counts, compaction boundary derivation, completeness signal tests.

Analytics: measure registry, empty backing row tests, unknown/null density tests, sample-frame footnotes, uncertainty rendering, cost/token lane partition.

UI: reader smoke, slow/missing route behavior, long-session navigation, live tail with daemon push, evidence basket export.

Public proof: one-command demo, docs link integrity, claims ledger coverage, install matrix, cold-reader pass.
