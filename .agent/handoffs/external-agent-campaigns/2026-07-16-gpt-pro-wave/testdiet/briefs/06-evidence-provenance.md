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

Use the smallest current source vocabulary and ObjectRef/EvidenceRef mechanisms
already present. Do not create a second evidence lattice or a test-only
registry. Build independent planted facts that include missing, contradictory,
duplicate, and differently authoritative observations and vary ordering and
grouping.

Name mutations such as dropping one source, double-counting a duplicate,
strengthening provenance without evidence, or coercing unknown to zero. Cover
the stable production readers directly; defer surfaces under explicit rewrite
only with a transfer-of-obligation note. Propose dominated arithmetic/local
model tests, but keep unique compatibility and error witnesses.
