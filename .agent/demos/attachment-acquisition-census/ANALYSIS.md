# Attachment Acquisition Census

Archive root: `/path/to/demo-archive`

Read-only census over the active archive (polylogue-83u.6), grouped by (origin, acquisition_status). `unfetched` is the honest floor (bytes never fetched, e.g. source-deleted / pre-install / provider-expiry) -- not a defect backlog. `missing_blob_ref_count` is the one genuinely actionable class: an `acquired` row whose blob file is absent.

## Totals

- Attachments: 1
- Declared bytes: 53
- Acquired blobs on disk: 1 (53 bytes)
- Missing blob refs (actionable debt): 0
- Acquired rows with a NULL blob_hash (schema anomaly, should be 0): 0
- Cross-origin attachments (referenced from >1 origin): 0
- Reconciles against `polylogue ops maintenance attachment-acquisition-debt`: True

## By origin / acquisition_status

| Origin | Status | Count | Declared bytes | Acquired-on-disk | Missing blob refs |
|---|---|---:|---:|---:|---:|
| aistudio-drive | acquired | 1 | 53 | 1 | 0 |
