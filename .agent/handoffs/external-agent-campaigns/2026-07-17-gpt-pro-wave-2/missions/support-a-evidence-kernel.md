# Lane A support — first-party analysis-evidence kernel and synthetic demo proof

You are producing an implementation package for Polylogue, a local archive for
AI sessions. Work from the attached fresh Chisel archive; it is authoritative
over any prior branch/PR memory. Your package will be integrated by a local
lane that separately has access to the private live archive.

## Mission

Implement the reusable first-party representation needed for an analysis to be
an archive object, not a bespoke report: a content-addressed analysis
definition, an immutable analysis-run receipt bound to existing query result
sets/evaluation receipts and archive generations, and materialized findings
which cite that evidence. Use the existing `user.db` assertion and
query-object substrate; do not invent a parallel report registry or make an
unjustified durable schema migration.

Build a privacy-safe synthetic fixture and a real-route test that proves:

1. a definition has stable identity;
2. a run binds the exact inputs, evaluator/world, and privacy/retention stamp;
3. a finding cannot be read as evidence-free prose; and
4. re-running with changed corpus/evaluator inputs creates a distinguishable
   new receipt rather than overwriting history.

Trace the current query-object, evaluation-receipt, assertion, and public
claims code before choosing names. Fit the smallest extension into its native
layer and expose a typed read path only where the product has a real consumer.

## Boundary

Do not regenerate a private claim-vs-evidence report, inspect a live archive,
or alter classifier semantics. The paired local lane owns calibration,
interpretation, and live evidence. Your deliverable must be independently
valuable for any future analysis, not hard-coded to one metric.

## Required package

Return `HANDOFF.md`, `PATCH.diff`, `TESTS.md`, and `EVIDENCE.md`. Include exact
file paths, an acceptance matrix, migration classification, and tests that
exercise production storage/read code—not a toy stand-in. Do not claim tests
were run. If the snapshot proves a necessary prerequisite already exists or a
durable migration is unavoidable, report that precisely rather than fabricating
a second substrate.
