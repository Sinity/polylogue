Title: "[research 02] Cancellable resumable SQLite reads"

Job ID: `deep-research-02`
Result ZIP: `deep-research-02-sqlite-cancellation-r01.zip`

Research current SQLite and Python sqlite3/aiosqlite mechanisms for bounding,
interrupting, and resuming expensive read-only queries: progress handlers,
interrupt, deadlines, statement lifecycle, transactions/snapshots, keyset
pagination, materialized intermediate state, temp resources, and concurrency
with one writer. Compare semantic guarantees and failure/cleanup behavior.
Produce a decision matrix for Polylogue's query DSL/daemon/API, including which
operations can be losslessly resumed and what receipts/work counters can be
observed without duplicating the query algorithm.

## Research contract

Use Deep Research, not ordinary implementation mode. Prefer current primary and
official sources; record direct URLs, publication/update dates, access date,
and the exact claim each source supports. Distinguish standards, documented
provider policy, measured behavior, informed inference, and proposal. Search
for counterevidence and incompatible constraints rather than writing a survey
that merely confirms the mission premise.

A current Polylogue project-state archive may be attached for product context.
Inspect relevant source and Beads so the research resolves concrete project
decisions, but do not return speculative patches. Map each conclusion to the
named Polylogue decision/Bead, state what should change, what should not change,
and what local experiment would falsify the recommendation.

Create the exact `Result ZIP` named near the top under `/mnt/data/`, containing
`DECISION-MEMO.md`, `SOURCE-LEDGER.md`, `COUNTEREVIDENCE.md`, and
`POLYLOGUE-MAPPING.md`. Do not copy attached project inputs into it. Attach the
finished ZIP through a working user-clickable conversation link; an internal
temporary path alone is not delivery. Reopen and validate the ZIP, report
SHA-256/size/members, and provide a substantive direct answer covering
conclusions, rationale, limitations, missing evidence, and the likely value of
another iteration before the exact package link.

Do not perform an adversarial review unless explicitly requested. On an
ordinary **iterate/continue** request, extend the strongest unresolved research
branch and regenerate the complete package. On an explicit **adversarial
review** request, try to falsify the prior memo with counterevidence, later or
more authoritative sources, incompatible policies/standards, hidden product
assumptions, and experiments that would overturn its recommendations. Repair
legitimate findings, regenerate the cohesive package, and report what changed,
what remains uncertain, and whether another pass is worthwhile.
