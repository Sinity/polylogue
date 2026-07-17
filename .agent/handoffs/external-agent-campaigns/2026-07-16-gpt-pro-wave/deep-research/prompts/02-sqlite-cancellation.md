Title: "[research 02] Cancellable resumable SQLite reads"

Job ID: `deep-research-02`

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

Present a substantive, self-contained research report with conclusions,
rationale, source-by-source support, counterevidence, limitations, missing
evidence, Polylogue decision mappings, and the likely value of another
iteration. It must remain useful to a reader who has not opened the attached
project-state archive.

Do not perform an adversarial review unless explicitly requested. On an
ordinary **iterate/continue** request, extend the strongest unresolved research
branch and return a revised complete report. On an explicit **adversarial
review** request, try to falsify the prior memo with counterevidence, later or
more authoritative sources, incompatible policies/standards, hidden product
assumptions, and experiments that would overturn its recommendations. Repair
legitimate findings and report what changed, what remains uncertain, and
whether another pass is worthwhile.
