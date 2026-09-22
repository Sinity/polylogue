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
| [Hermes archival export contract](hermes-archival-export-contract.md) | Versioned Hermes session export schema + durable lifecycle-event spool + snapshot reconciliation (polylogue-fs1.7) |
| [Analysis rigor](analysis-rigor.md) | Rigor mechanisms for agent claims: population-validity (metric hashes, pre-registration, holdouts) + comparative judgment (Bradley-Terry rankings, agent judges, cascades) (polylogue-rxdo.9) |
| [Query set algebra](query-set-algebra.md) | Set-composition semantics over query results (polylogue-fnm.13) |
| [Agent-first MCP](agent-first-mcp.md) | MCP surface doctrine (polylogue-t46.8, polylogue-rsad) |
| [Transcript-window responsibility](transcript-window-responsibility.md) | Who executes the one-session message window on each public surface, and the two owners that remain (polylogue-vclez) |
| [Project memory](project-memory.md) · [Second brain](second-brain.md) · [Time machine](time-machine.md) · [Archive storytelling](archive-storytelling.md) · [Whole product](whole-product.md) | Vision statements for future planning |
| [Query-action workflows](../product/workflows.md) | Standing selection, cardinality, and executable-evidence guide |
| [Incident 14:32 proof world](incident-1432-proof-world.md) | Shared deterministic adversarial corpus for the still-open proof-world work (polylogue-212.11) |
| [Retained inputs and safe supersession](retained-inputs-and-supersession.md) | What the archive retains per observation and when retained bytes may be retired: material-law retention, scope-bearing identity, value-not-object supersession (polylogue-0qbdh) |
| [Prefix-blob reclamation](prefix-blob-reclamation.md) | Reference-blob representation for byte-proven superseded revision prefixes; consent-gated durable-tier reclamation (polylogue-vzn6). Its production proposal is superseded — see [Retained inputs and safe supersession](retained-inputs-and-supersession.md) |
| [Derived-artifact freshness](derived-artifact-freshness.md) | Source-digest and verify-or-refuse law for derived state (polylogue-ntwtk) |
| [Convergence simplification inventory](convergence-simplification-inventory.md) | Deletion/collapse inventory for the daemon convergence redesign — what phases (b)-(d) remove and why (polylogue-m6tp) |
| [Addressable raw decisions](raw-decision-authority.md) | Durable raw-authority decisions addressed by plan digest instead of per-pass census membership; per-field disposition, transactions, and migration order (polylogue-gen6d) |
| [Interrupted retained source generations](interrupted-retained-source-generation.md) | Evidence for accepted source generations left nonterminal by an interrupted daemon ingest (polylogue-xt5ga). Its abandonment/release policy is superseded by the retention default recorded in the same file (polylogue-fzbzk) |
| [The ops.db diet](ops-tier-diet.md) | Amended ops-tier target: the unknown-representation for rollup-backed facts, the convergence_debt ruling, and the reader-to-replacement map for every table (polylogue-pnxl6) |
| [Daemon core](daemon-core.md) | The resident daemon: ownership, write serialization, ingest shape, convergence, service lifecycle, status cost, with rehearsal measurements (polylogue-bp12n) |
| [Domain derivation adoption ledger](domain-derivation-deletion-ledger.md) | Production ownership, deletion accounting, and remaining convergence predecessors. |

If a doc here stops matching the external task authority, update or purge it.
