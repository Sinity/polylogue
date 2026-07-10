# Polylogue Design Direction

Design direction lives in the Beads backlog, not in this directory: each
surface's owning bead carries its spec in the `design` field (`bd show <id>`),
and the aesthetics/interaction directions live in `.agent/reports/` with
their decisions recorded on beads. Planning documents are mined into beads
and then purged — there is no "historical reference" tier (superseded
`docs/execution-plan.md`, and the MK2/MK3 packs, were removed under this
rule; recover from git history if ever needed).

What remains here are **standing design references** that describe durable
domain models rather than plans:

| Doc | Purpose |
|-----|---------|
| [Session lineage model](session-lineage-model.md) | Fork/resume/compaction storage + composition semantics (polylogue-4ts) |
| [Query set algebra](query-set-algebra.md) | Set-composition semantics over query results (polylogue-fnm.13) |
| [Agent-first MCP](agent-first-mcp.md) | MCP surface doctrine (polylogue-t46.8, polylogue-rsad) |
| [Incident 14:32 proof world](incident-1432-proof-world.md) | Shared deterministic demo corpus model + anti-circularity/anti-vacuity rules (polylogue-212.11, polylogue-212.12) |
| [Project memory](project-memory.md) · [Second brain](second-brain.md) · [Time machine](time-machine.md) · [Archive storytelling](archive-storytelling.md) · [Whole product](whole-product.md) | Vision statements feeding horizon beads |
| [Query-action workflows](query-action-workflows.md) | Moved pointer to the generated `docs/product/workflows.md` |

If a doc here stops matching its owning beads, the beads win — update or
purge the doc.
