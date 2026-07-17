# Status — Sinex-05 Coverage-Obligation Compiler

**Asked:** "Execute the attached sinex-05-coverage-obligation-compiler.md against the attached
current Sinex Chisel package... Produce exactly the requested evidence-cited report... mark
unavailable evidence unknown."

**Delivered:** "Sinex coverage-obligation compiler review" (~59.1K chars) — a review of what a
proposed test/coverage-obligation compiler would need to guarantee against Sinex's actual build
graph. Found ten orphan tests including five unreferenced source-parser test suites; works
through the history-schema tables (`xtask/src/history/db/schema.rs`) and derivation registry to
determine what "obligation compilation" would need to bind (explicit registry entries vs.
inferred coverage). Embeds rust/sql code snippets showing schema and derivation-registry
excerpts as evidence.

**Recoverable vs LOST:** Fully recovered verbatim (`delivery.md`, turn 23, ~59.1K chars). All
embedded code/sql snippets are inside this same delivery turn — no separate inline-artifacts
content. Nothing LOST — no downloadable sandbox package was referenced.

**Regeneration value:** Low — complete, self-contained, citation-bearing report.
