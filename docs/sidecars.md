# Sidecars

Every non-primary file the system produces or consumes alongside a primary
artifact. Mishandling a sidecar has caused phantom sessions, invalid
verification graphs, and failed backups; each kind below names its contract.

| Kind | Producer | Consumer | Lifecycle | Contract | Mishandling failure |
| --- | --- | --- | --- | --- | --- |
| Provider metadata (`*.md.metadata.json`, Antigravity) | vendor app | artifact taxonomy and raw evidence archive | copied into durable source-tier raw/artifact records during live acquisition; vendor copy remains | `parse_policy="raw-only"` with no parser (`sources/origin_specs.py`, rule `agent_sidecar_meta`): admitted by `admit_raw_artifact_payload` / `admit_raw_artifact_blob_ref`, never session-parsed | phantom one-message sessions; backups must account for archived copies |
| Claude tool-result payloads (`.json`, `.txt`, `.html`, …) | Claude Code | artifact acquisition joined to owning tool_use | lives with the transcript tree; the bytes are retained as raw evidence | declared opaque payloads (`sources/origin_specs.py`, rule `tool_result_sidecar`); admission completeness owned by polylogue-omsw (only `.json` admitted today) | silent acquisition gaps; independent-session risk |
| Gemini CLI tool outputs (`tool-outputs/session-<id>/*`) | Gemini CLI | artifact acquisition joined to the owning tool call | lives with the chat snapshot tree; the bytes are retained as raw evidence | same `tool_result_sidecar` rule kind, declared on the gemini-cli `OriginSpec`; the masking envelope is a both-ends truncation, so recovered text replaces the inline text rather than extending it | the block keeps `<tool_output_masked>` forever once the tree moves |
| ChatGPT asset names and library catalog | ChatGPT export | provider assembly and retained replay | retained bytes plus an operation-owned scratch index | complete sidecar parsing preserves last duplicate values and original key order; acquired member metadata and rendition joins use the same paged index, whose Native SQLite readers must settle before deletion | incomplete sidecars must not create partial lookup evidence; early artifact deletion can invalidate active readers |
| `history_sidecars` rows (source.db) | acquisition | replay/verification | durable | `content_hash` is a 32-byte blob hash addressing the sidecar bytes, unique with `(origin, source_path)`; the row carries no raw-revision column, so `blob_refs` `ref_type='sidecar'` keyed by `sidecar_id` is its only ownership edge | a sidecar row read as evidence of a primary revision it never names |
| SQLite `-wal`/`-shm`/`-journal` | SQLite runtime | SQLite runtime only | transient beside any live DB (`-journal` in rollback-journal mode) | never copy, hash, or ingest independently; snapshots go through the backup API (`sources/sqlite_snapshot.py::snapshot_sqlite_database`); staging discards them (`_SQLITE_SIDECAR_SUFFIXES`) | `mode=ro` opens fail under read-only mounts — use `immutable=1` only for genuinely frozen files |
| Import staging receipt (`input-*.polylogue-import`) | `stage_source_input` | `read_staging_receipt` and `bind_staged_member` | with the private directory slot | streams original semantic, physical and profile identity plus each staged member's inode and content revision into the manifest spool; outside the payload tree | present unreadable, incomplete or mismatched receipt refuses acquisition; ordinary payload filenames have no metadata meaning |
| Testmon database (`.cache/testmon/testmondata` plus `-wal`, `-shm`, and `-journal`) | testmon/devtools | testmon | mutable per checkout, gitignored | the database and its SQLite sidecars form one mutable testmon state; repository code does not create `.testmondata.bound-*` copies | missing or unusable testmon state can prevent affected verification from constructing a selection |
| Verification graph manifests (`.cache/verify/graph/**`) | devtools | devtools, harvest evidence | immutable JSON manifests per checkout, gitignored | independent manifests describing the verification graph; they are not testmon database copies or SQLite sidecars | invalid or stale graph evidence can make ordinary affected verification refuse before pytest; only explicit `devtools verify --all` runs the complete corpus |

Hook-event carriers (`hooks/carriers/<provider>/<day>/<pid>.ndjson`) are
primary acquisition sources, not sidecars; their envelope contract lives in
`sources/hooks.py`.

Both tool-output families are joined back to their owning block from
**retained bytes, not from the original tree** (polylogue-cq1ql). The join
reads a `RetainedSidecarScope` (`sources/sidecar_evidence.py`); acquisition
resolves that scope from the source tree while derivation resolves it from
`source.db` plus the blob store
(`sources/live/sidecar_resolution.py`). A transcript therefore replays to the
same full text, outcome and ownership after its session directory is gone, and
a preview whose persisted-output pointer names a file the scope does not hold
records an explicit `expected_sidecar_not_retained` outcome instead of silently
staying truncated.

A Claude tool-result payload the transcript points at but the archive does not
retain falls back to the same call's `PostToolUse` hook `tool_response`
(`sources/live/hook_tool_response.py`), so the recovery order is inline
preview, then retained sidecar bytes, then hook. That third fallback is bounded for
`Bash`: Claude Code caps the `tool_response.stdout` it hands a hook at 30,000
characters and repeats the same persisted-output pointer, so a recovered block
records whether the recovery was complete instead of claiming the whole
output. Nothing else is derived from a hook payload into `blocks`.
