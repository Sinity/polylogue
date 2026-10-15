# Decision: keeping input bytes

Status: **decided.** Keeping every input is the operator's decision; keeping
every revision, and the storage forms below, were decided from
measurements (`evidence/blobstore-v2.md`).

## Decided

Polylogue keeps its own copy of every input's exact bytes in a
content-addressed blob store. Provider files get rotated or auto-deleted
(Claude Code removes old transcripts by default), exports get lost, browser
captures and hook events exist nowhere else. "This is the reason to have a
blob store" (operator, 2026-10-11). The archive must be rebuildable from the
blob store alone, with every original file gone.

## How revisions are stored

**Keep every revision, stored in the cheapest exact form for its kind of
change.** Appends store only the new bytes (a piece). Rewrites of text files
are stored as zstd deltas against the input's newest plain blob when that
halves them (Codex rewrites: 0.94% of a plain copy at the median and 6.7% at most over
1,431 real pairs, `evidence/codex-rewrite-deltas.md`; ChatGPT conversations:
under 3%). Database snapshots, read page by page in one SQLite read transaction
(`sqlite_dbpage`), store only
the pages that changed (7 of 53,122 pages of Codex's state database between
two snapshots). Every form reproduces the exact bytes of every revision
through its extents.

**Why every revision:** providers rewrite their files and drop information.
Codex's rewrite of its rollouts removed reasoning, user messages and tool
output from nearly every file, so an older revision can be the only record of
something. "Keep only the newest copy", the earliest recommendation, would
have lost exactly that.

In July the operator endorsed content-defined chunking ("yes we do want
this"). What it was wanted for, every revision kept at the cost of what
changed, is what this gives: pieces for appends, deltas for rewrites, page
patches for databases. Chunking itself measured worse.

Rejected:

- **Content-defined chunking.** It shared only 5–49% of bytes on Codex
  rewrites, against 90–99% saved by a delta, and adds a chunk index for no
  gain on appends or databases.
- **Keeping only the newest copy.** It loses what the provider dropped, and
  as first specified it rewrote a whole live 1.6 GB file on every growth.
- **Member-level storage of export ZIPs.** It saves 2% on the measured
  exports and cannot reproduce exact ZIP bytes simply; ZIPs stay whole.
- **Logical row dumps of databases.** They lose exact pages and free-page
  content. The Python version chose them (`polylogue-jbcln`, delivered
  2026-09-16; `polylogue-8t81f`) because whole-file snapshots re-stored every
  page and grew the blob store (551 MB of residue), and because page images
  could not be proven against a live database. Both reasons are answered
  now: one read transaction gives a consistent snapshot, and page patches cost
  kilobytes per change. Logical rows are derived from a snapshot by the
  parser. Decided by the spec authors from principle 5 (exact input bytes
  are kept; a logical export is a lossy transform; content that must go is
  removed by excision); the operator may overrule.

## Details

`spec/storage/blobs.md` (Blobs, Inputs, revisions and extents, Observing a
file, Provider databases, What an observation records) owns the
mechanics: where blobs live, frame size, input IDs, change classification,
joining, deltas, page patches, crash safety and excision. This record keeps
only the decision and its reasons. One rule from there matters for the
decision: a blob is never removed by a retention policy, only when no extent
refers to it and no delta uses it as base (after joining, when its bytes live
on in the joined blob) or by excision.
