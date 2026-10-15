# Decision: how many database files

Status: **decided.** Decided by the spec authors from the operator's
request to reduce the six tiers (2026-08-31, October feedback) and the
audit's count of 8–12k lines of code keeping the six tiers consistent; the
operator may overrule. The split follows one rule: what cannot be recomputed
(`archive.db`), what can (`index.db`), and what costs money to recompute
(`embeddings.db`).

The Python version has six SQLite files (`source`, `index`, `embeddings`,
`user`, `audit`, `ops`) and 156 tables. Every place two durable files had to
agree needed its own ordering-and-replay protocol (audit continuity, the
reference seal, marker streams, excision receipts). The operator questioned
the split on 2026-08-31: "sources.db vs index.db makes sense. I'm not sure
about other tiers."

## Options

- **A. Three files:** `archive.db` (everything that cannot be recomputed:
  input list, notes, saved queries, excisions, settings), `index.db`
  (everything derived, plus daemon progress), `embeddings.db` (vectors, kept
  separate because they are paid for and large).
- **B. Two files:** durable plus derived, with vectors inside the derived
  file. Rejected: rebuilding the index would put purchased vectors at risk.
- **C. One file.** Simplest, but a rebuild of derived data would mean
  rewriting tables inside the durable file, and backups would carry gigabytes
  of rebuildable data.
- **D. Keep six.** Rejected; the protocols between them are most of what
  should disappear.

## Decision

**A.** No two durable files ever need to agree within one write, so no
cross-file protocol exists: a note is one transaction in `archive.db`; an
ingest stores the blob, records the input in `archive.db`, then parses into
`index.db`, and a crash between those steps just leaves parsing to redo.

Daemon progress (file read positions, recent activity for `status`) lives in
`index.db`: if the index is rebuilt, the daemon re-reads files from the
stored blobs, which is the normal path anyway.
