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
