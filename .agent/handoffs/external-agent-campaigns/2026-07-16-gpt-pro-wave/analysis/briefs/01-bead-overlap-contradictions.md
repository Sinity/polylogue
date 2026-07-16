Title: "[analysis 01] Bead overlap and contradiction audit"

Job ID: `analysis-01`
Result ZIP: `analysis-01-bead-overlap-contradictions-r01.zip`

Audit all open/in-progress Polylogue Beads by current source area, intended
authority, dependency, and acceptance semantics. Find duplicate owners,
contradictory product decisions, stale descriptions superseded by notes,
parallel registries/protocols, false blockers, and clusters that should share
one branch versus remain separate. Prioritize P0/P1 and active write hotspots.
Return a source-validated graph-change proposal and execution clusters, but do
not mutate Beads. Every proposed merge/supersede/dependency change needs exact
evidence and an explanation of information that must be preserved.
