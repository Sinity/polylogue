Title: "[testdiet 05] Bounded query work, cancellation, and cleanup"

Job ID: `testdiet-05`
Result ZIP: `testdiet-05-query-bounds-progress-r01.zip`
Dependency: integrate after the query algebra survivor in `testdiet-01`.

## Mission

Implement the strongest currently authorized slice of bounded and
interruptible query work. Reuse the exact-selection facts from the query
survivor and exercise irrelevant-growth equivalence, selected-work scaling,
lossless pagination/addressability, cancellation/deadline propagation, truthful
progress, and cleanup through real SQLite/repository/public routes.

Use an existing work/progress counter if current source provides one. Add a
minimal production observation seam only if it measures actual SQLite/runner
work and has at least the concrete consumer in this survivor; do not duplicate
the query algorithm in a test counter. Prefer deterministic cancellation over
wall-clock flakiness.

Name work-amplification, ignored-cancellation, depth-limit, leaked temporary
state, or double-counted-progress mutations and the expected failure. Read the
current product decision for oversized results and cancellation; implement no
semantics that remain undecided. Propose slow/redundant examples for later
certification without deleting them.
