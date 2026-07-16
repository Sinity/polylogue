Title: "[testdiet 06] Evidence monotonicity and provenance conservation"

Job ID: `testdiet-06`
Result ZIP: `testdiet-06-evidence-provenance-r01.zip`

## Mission

Implement survivor laws for Polylogue evidence monotonicity and quantitative
provenance conservation. Across the current source/index/user evidence models,
prove that adding stronger evidence can refine or supersede a claim without
silently reducing supported information; aggregations conserve totals across
disjoint sources; unknown/absent/unsupported never becomes numeric zero; and
every public value retains the evidence/provenance needed to explain it.

Use `architecture/07-evidence-provenance-and-public-algebra.md` as the
recommended contract. Realize its `EvidenceValue` axes and fact-family
declaration only for the smallest canaries needed by this lane, using current
ObjectRef/EvidenceRef and domain-owned storage. Prove value state, authority,
definition, temporal provenance, enumeration/coverage, and
freshness/degradation independently where the selected family declares them.
Do not create a generic fact store, second evidence lattice, or test-only
registry. Build independent planted facts that include missing,
contradictory, duplicate, and differently authoritative observations and vary
ordering and grouping.

Name mutations such as dropping one source, double-counting a duplicate,
strengthening provenance without evidence, or coercing unknown to zero. Cover
the stable production readers directly; defer surfaces under explicit rewrite
only with a transfer-of-obligation note. Propose dominated arithmetic/local
model tests, but keep unique compatibility and error witnesses.
