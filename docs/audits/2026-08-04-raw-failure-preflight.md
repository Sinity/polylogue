# Raw-failure preflight, 2026-08-04

## Scope and method

This is an immutable, read-only census of the active archive before deploying `polylogue-dyica`. It opened `/realm/db/polylogue/source.db` and `ops.db` with SQLite read-only mode, inspected the stopped user daemon, and compared source metadata with retained raw metadata. It did not start the daemon, run reprocessing, reset a cursor, or remove any source or blob data.

Raw IDs and source paths are intentionally omitted. The captures and exports contain private operator data, and aggregate evidence is sufficient for the implementation decision.

`polylogued.service` was stopped after an exit 143 at 2026-08-04 04:40:38. The one maintenance failure was a session-insights repair blocked by the live source-tier schema version. The deployed runtime expected version 25 while the archive was at version 24. This implementation does not migrate that archive.

## Census

The archive contained 112 failures: 111 raw parse failures and one maintenance failure. There were no raw validation failures.

| Origin or route | Existing artifact kind | Count | Source mutability evidence | Retry eligibility before deployment | Classification |
| --- | --- | ---: | --- | --- | --- |
| `claude-code-session` | `coordinator_session_stream`, `supported_parseable` | 59 | Live Claude source; a later current file observation exists, but historical retained bytes have no recorded byte-prefix proof | Unexplained historical rows | Candidate deferred only after a future route proves the retained bytes are the strict current prefix of a larger source |
| `claude-code-session` | No artifact observation | 4 | Live Claude source, hot-file metadata | Unexplained | Lifecycle revision conflict |
| `codex-session` | No artifact observation | 14 | Live Codex source, hot-file metadata | Unexplained | Lifecycle revision conflict |
| `codex-session` | No artifact observation | 1 | Live Codex source, hot-file metadata | Unexplained | Unconvertible byte-head lifecycle failure |
| `unknown-export` | No artifact observation | 25 | Source missing from archive inbox or legacy inbox | No automatic retry | Terminal unsupported shape: parser produced no sessions |
| `unknown-export` | No artifact observation | 6 | Five legacy inputs remain immutable and available; one source is absent | No automatic retry | Terminal corrupt input: JSON decoding failed |
| `hermes-session` | No artifact observation | 2 | Live Hermes source, hot-file metadata | No automatic retry without a demonstrated parser path | Terminal unsupported shape: artifact produced no materializable sessions |
| maintenance replay | Failure routing record | 1 | Durable source tier is older than the deployed runtime requirement | Blocked pending backup-gated migration and reviewed retry | Explicit maintenance schema mismatch |

The rows sum to 112. The 59 Claude rows are the only existing `raw_artifacts` observations for the 111 raw failures. Their current `supported_parseable` classification is historical parser metadata, not structural proof of a deferred capture. The remaining 52 raw failures have no artifact observation.

The source mutability evidence groups the 111 raw failures as follows: 80 live-source rows with hot-file metadata, 6 immutable legacy inputs still available, and 25 source-missing archive or legacy inputs. Hot metadata alone is not enough to defer a raw. A future ingest must establish both conditions against the same captured bytes: the current source is larger and its prefix hash exactly matches the retained payload.

## Implementation decision

The real full-ingest route now writes a closed `raw_artifacts` outcome after retaining a raw failure:

- `deferred_hot_jsonl_capture` only when the source has grown and its prefix exactly matches the retained incomplete JSONL payload;
- `terminal_corrupt_input` for an incomplete capture without that proof;
- `terminal_unsupported_shape` when parsing yields no positive conversational evidence.

All three states retain the raw payload and its parser diagnostic. Deferred and terminal outcomes acknowledge the source record so the cursor does not retry an unchanged payload indefinitely. Status and health report deferred retryable work, terminal rejections, and unexplained failures separately. Historical rows stay unexplained until an explicitly reviewed route records new evidence.

## Post-deploy boundary

No live reprocessing occurred for this change. A future operator run requires a fresh verified backup, a targeted dry-run receipt, review of each proposed raw state transition, and an apply receipt. It must not reset cursors, bulk reprocess the archive, or delete source/blob data. The stopped daemon must resume convergence for deferred work; remaining unexplained lifecycle failures stay visible for separate diagnosis.
