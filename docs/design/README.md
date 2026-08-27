# Polylogue Design Direction

Design direction lives in the external task authority, not in this directory.
This repository retains only durable design references and pinned historical
evidence; branches and PRs never carry task-state exports. Superseded planning
documents are recovered from Git history when needed.

What remains here are **standing design references** that describe durable
domain models rather than plans:

| Doc | Purpose |
|-----|---------|
| [Session lineage model](session-lineage-model.md) | Fork/resume/compaction storage + composition semantics (polylogue-4ts) |
| [Physical session identity](physical-session-identity.md) | Collision census and durable identity beneath lossy public-origin projection (polylogue-4ts.7) |
| [Hermes archival export contract](hermes-archival-export-contract.md) | Versioned Hermes session export schema + durable lifecycle-event spool + snapshot reconciliation (polylogue-fs1.7) |
| [Analysis rigor](analysis-rigor.md) | Rigor mechanisms for agent claims: population-validity (metric hashes, pre-registration, holdouts) + comparative judgment (Bradley-Terry rankings, agent judges, cascades) (polylogue-rxdo.9) |
| [Query set algebra](query-set-algebra.md) | Set-composition semantics over query results (polylogue-fnm.13) |
| [Agent-first MCP](agent-first-mcp.md) | MCP surface doctrine (polylogue-t46.8, polylogue-rsad) |
| [Project memory](project-memory.md) · [Second brain](second-brain.md) · [Time machine](time-machine.md) · [Archive storytelling](archive-storytelling.md) · [Whole product](whole-product.md) | Vision statements for future planning |
| [Query-action workflows](../product/workflows.md) | Standing selection, cardinality, and executable-evidence guide |
| [Incident 14:32 proof world](incident-1432-proof-world.md) | Shared deterministic adversarial corpus for the still-open proof-world work (polylogue-212.11) |
| [Prefix-blob reclamation](prefix-blob-reclamation.md) | Reference-blob representation for byte-proven superseded revision prefixes; consent-gated durable-tier reclamation (polylogue-vzn6) |
| [Derived-artifact freshness](derived-artifact-freshness.md) | Source-digest and verify-or-refuse law for derived state (polylogue-ntwtk) |
| [Convergence simplification inventory](convergence-simplification-inventory.md) | Deletion/collapse inventory for the daemon convergence redesign — what phases (b)-(d) remove and why (polylogue-m6tp) |

If a doc here stops matching the external task authority, update or purge it.
