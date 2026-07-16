Title: "[testdiet 04] Monotonic derived freshness"

Job ID: `testdiet-04`
Result ZIP: `testdiet-04-derived-freshness-r01.zip`
Dependency: integrate only after `testdiet-02` and `testdiet-03` foundations are
adjudicated.

## Mission

Implement a survivor law that derived Polylogue facts never silently move
backward relative to their content and recipe/generation authority. Cover the
current FTS, insight, and—where the source contract makes it meaningful—
embedding catch-up identities through ingest, reprocess, recipe change,
restart, and rebuild. A reader must see a current generation, an explicit
pending/degraded state, or an honest absence; never a stale value presented as
current.

Use the recommended contract in
`architecture/05-derived-freshness.md`: one small typed `DerivationKey` and
attempt predicate, with domain-specific ledgers for FTS, embeddings, and
insights. Validate the named source/recipe/output fields against current source
and full Bead notes, but do not reopen the rejected timestamp/count/boolean
alternatives without contradictory evidence. Do not invent a Diet-specific
freshness token or a universal derivation table. Reuse the convergence and
rebuild survivor mechanisms produced by the prerequisite lanes. Include varied
arrival orders, superseded workers, and an independent expected freshness
relation.

Name a mutation that ignores recipe identity, lets an old generation mark a
new key current, retains a stale materialization, or marks debt complete early.
If live source proves one tier cannot yet implement the recommended seam,
implement the coherent decided slice and return the exact source conflict and
smallest dependency—not a fresh menu of architecture alternatives.
