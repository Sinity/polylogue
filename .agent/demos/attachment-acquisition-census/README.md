# Attachment Acquisition Census

Read-only census over the active archive (polylogue-83u.6): how much
attachment evidence is actually backed by bytes, and where is the
recoverable gap, broken down by origin and `acquisition_status`.

This is the sizing input for the acquisition beads in the 83u program
(polylogue-83u.2 Drive/zip/local byte acquisition, polylogue-83u.3 live
browser-capture upload interception) and the honesty check on any
"attachments preserved" claim: `unfetched` is the honest, expected floor
(bytes were never fetched — source-deleted, pre-install, provider-expiry —
not a defect), while `missing_blob_ref_count` (an `acquired` row whose blob
file is actually absent from the store) is the one genuinely actionable
debt class.

## Regenerating

```bash
POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue \
  bash .agent/demos/attachment-acquisition-census/regenerate.sh
```

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

## Baseline (2026-07-08)

7,390 attachments total; 967 acquired (0 missing blob files — clean);
6,423 unfetched. The unfetched byte volume is heavily concentrated in
`chatgpt-export` (13.4GB declared, 0 acquired) — the largest single
re-acquisition opportunity in the archive, and the natural first target
once polylogue-83u.2/83u.3 land. See `ANALYSIS.md` for the full
per-origin table.

Re-run this census after 83u.2/83u.3 ship to produce the "after" half of
the before/after pair required by this bead's AC; write the delta back
into the parent epic (polylogue-83u) as closing evidence.
