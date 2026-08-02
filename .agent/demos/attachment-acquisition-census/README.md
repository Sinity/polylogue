# Attachment Acquisition Census

A read-only census methodology, over the deterministic seeded demo archive
(`polylogue demo seed`): how much attachment evidence is actually backed by
bytes, and where is the recoverable gap, broken down by origin and
`acquisition_status`.

This is the sizing methodology behind the acquisition beads in the 83u
program (polylogue-83u.2 Drive/zip/local byte acquisition, polylogue-83u.3
live browser-capture upload interception) and the honesty check on any
"attachments preserved" claim: `unfetched` is the honest, expected floor
(bytes were never fetched — source-deleted, pre-install, provider-expiry —
not a defect), while `missing_blob_ref_count` (an `acquired` row whose blob
file is actually absent from the store) is the one genuinely actionable
debt class.

**This packet is a private-data-free methodology demo, not a live-archive
report.** The actual operator-archive census for the 83u program (real
byte counts, real per-origin totals) is tracked as bead notes on
polylogue-83u, not committed here — this shelf commits only the reusable
census methodology, run against a synthetic fixture.

## Regenerating

```bash
polylogue demo seed --root ./demo-archive --force --with-overlays
POLYLOGUE_ARCHIVE_ROOT="$PWD/demo-archive" \
  bash .agent/demos/attachment-acquisition-census/regenerate.sh
```

`regenerate.sh` requires `POLYLOGUE_ARCHIVE_ROOT` to be set explicitly — it
refuses to run with no archive root configured, so it can never silently
fall back to an operator's live archive (polylogue-0bgr).

Opens `source.db`/`index.db` read-only (`mode=ro`); never mutates the
archive. Cross-checks its totals against
`polylogue ops maintenance attachment-acquisition-debt --output-format json`
(also captured verbatim as `reconcile-attachment-acquisition-debt.json`) and
records `reconciliation.totals_match` in `census.json`.

## Files

- `census.json` — full structured census: totals, per-(origin,status) rows
  with declared/on-disk byte sums and a bounded (~20) attachment-id sample,
  cross-origin fan-out count, and the reconciliation check.
- `ANALYSIS.md` — human-readable summary table.
- `reconcile-attachment-acquisition-debt.json` — the raw output of the
  global (non-origin-broken-down) CLI diagnostic this census reconciles
  against.

## Fixture run

Against the deterministic demo archive, the census finds 1 attachment
total (1 acquired, 0 missing blob refs, reconciled against
`attachment-acquisition-debt`) — see `census.json`/`ANALYSIS.md` for the
full breakdown. This demonstrates the methodology, cross-check, and output
shape; it carries no evidentiary weight about the real archive's
acquisition backlog.
