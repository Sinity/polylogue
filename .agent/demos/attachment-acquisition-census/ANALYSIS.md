# Attachment Acquisition Census

Archive root: `/home/sinity/.local/share/polylogue`

Read-only census over the active archive (polylogue-83u.6), grouped by (origin, acquisition_status). `unfetched` is the honest floor (bytes never fetched, e.g. source-deleted / pre-install / provider-expiry) -- not a defect backlog. `missing_blob_ref_count` is the one genuinely actionable class: an `acquired` row whose blob file is absent.

## Totals

- Attachments: 7,390
- Declared bytes: 13,714,191,312
- Acquired blobs on disk: 967 (29,722,444 bytes)
- Missing blob refs (actionable debt): 0
- Acquired rows with a NULL blob_hash (schema anomaly, should be 0): 0
- Cross-origin attachments (referenced from >1 origin): 0
- Reconciles against `polylogue ops maintenance attachment-acquisition-debt`: True

## By origin / acquisition_status

| Origin | Status | Count | Declared bytes | Acquired-on-disk | Missing blob refs |
|---|---|---:|---:|---:|---:|
| (unknown) | acquired | 9 | 403,110 | 9 | 0 |
| (unknown) | unfetched | 9 | 0 | 0 | 0 |
| aistudio-drive | acquired | 1 | 41,401 | 1 | 0 |
| aistudio-drive | unfetched | 1,666 | 0 | 0 | 0 |
| chatgpt-export | unfetched | 3,078 | 13,369,453,110 | 0 | 0 |
| claude-ai-export | acquired | 957 | 29,277,933 | 957 | 0 |
| claude-ai-export | unfetched | 1,636 | 315,015,758 | 0 | 0 |
| grok-export | unfetched | 34 | 0 | 0 | 0 |
