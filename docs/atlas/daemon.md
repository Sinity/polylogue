# Daemon

## Runtime ownership

The daemon holds writer/rebuild exclusion for its lifetime. `DaemonWriteCoordinator` serializes publication and retains ownership until a cancelled operation actually terminates. `DaemonAPIHTTPServer.execution_kernel` is passed to the UDS server and to daemon derivation owners; their `DaemonWriteThreadBridge` instances use the same coordinator (`polylogue/daemon/cli.py:2653-2658`; `polylogue/daemon/cli.py:2725-2750`; `polylogue/daemon/http.py:5697-5734`; `polylogue/daemon/write_coordinator.py:772-790`).

`run_daemon_services` is the service composition entry point (`polylogue/daemon/cli.py:1953`). Its composition state declares `session_profile_callback` and `embedding_callback` (`polylogue/daemon/cli.py:2595-2596`), and constructs the `FtsConvergenceOwner` for startup work (`polylogue/daemon/cli.py:2823-2840`). FTS runs at startup and periodically; session profiles run after admitted ingest and during the periodic sweep; embeddings use watcher scopes and the periodic backlog owner. `run_daemon_services` hands both callbacks to the live watcher (`polylogue/daemon/cli.py:2988-3003`); after an admitted batch `FileIntakeAdapter.admit_page` calls the watcher's lease-free embedding and profile convergence (`polylogue/operations/intake_adapters.py:1068-1077`; `polylogue/sources/live/watcher.py:1334-1356`); the periodic sweep and backlog services are registered in `periodic_services` (`polylogue/daemon/cli.py:2888-2943`). These are source-route facts, not live deployment evidence.

Convergence emits structured events (`emit`/`span` from `polylogue/logging.py`) rather than free-form log lines: field names pass an allowlist and quarantined names are stripped from both rendered forms when `POLYLOGUE_LOG_REDACT=1` is set, so a rebuild is read from named events such as `daemon.barrier.failed` and their typed fields (`polylogue/logging.py:482-486`; `polylogue/logging.py:770-781`; `polylogue/daemon/convergence.py:1120-1126`).

Each daemon run has one run id, bound into the logging run context before the first event and written as the `daemon_lifecycle` row's `run_id`, so the row, its heartbeats, the status projection of that row and every event of the run join on it (`polylogue/daemon/cli.py:1887`).

The SSE replay ledger `daemon_events` is a resume buffer in the disposable ops tier with no row count or age bound. The daemon process owns it, because it serves every SSE stream and so knows every live cursor; an open stream registers its cursor, and each emit in the owning process removes the rows every live subscriber has read that are either granular topic frames or records superseded by a newer row of their kind, keeping the newest row of each superseded record kind for its in-process readers. Browser capture-health reports are stored atomically in the separate `capture_health_history` ops table and their consumed resume frames are pruned. The CLI streams history in pages; receiver GET accepts positive `page_size` and opaque `cursor` and returns `next_cursor`. The history owner returns at most 100 reports per page, with a continuation for the remainder regardless of requested page size. CLI JSON is staged in a private temporary file and published after the requested walk succeeds; text output streams. Continued pages exclude newly appended reports and refuse `history_cursor_reset` after ops replacement. The events storage owner classifies history reads and transactions: schema skew remains a permanent refusal, transient storage faults return `capture_history_unavailable`, and deterministic faults return `capture_history_storage_failed`; HTTP and CLI adapt those types. The highest removed id is kept as a watermark, and a `Last-Event-ID` below it is answered with a typed `aged_out` resync envelope rather than a short page (`polylogue/daemon/events.py`: `EventSubscriberRegistry`, `prune_daemon_events`, `_cursor_refusal_reason`; `polylogue/daemon/events_http.py`: `_stream_events`).

Each ended run gets one termination receipt, reconciled by the next start as its own `termination_reconciliation` service: the run's lifecycle row records its host identity (pid, boot id, service-manager invocation, cgroup instance and `memory.events` baseline), and the receipt classifies the end from the stop marker, the manager's unit result, kernel and `systemd-oomd` kill records naming that pid or cgroup, the cgroup counters and the boot id, citing each source it used and naming each it could not (`polylogue/operations/daemon_termination.py`: `classify_termination`). `lifecycle_status` carries the newest receipt and the runs still awaiting one.

Correlation crosses the compute boundary explicitly. Neither `threading.Thread`
nor `ThreadPoolExecutor.submit` copies contextvars, so both derivation-kernel
submits wrap their `partial` in `propagate(...)`; without it the work runs on a
pool thread with an empty context and its events lose the run's correlation id
(`polylogue/daemon/convergence.py:176-190`; `polylogue/daemon/convergence.py:280-284`; `polylogue/logging.py:441-445`).

Read this as a statement about the daemon's convergence path, not about the
tree. The ratchet is a `devtools gate patterns` rule, `legacy-stdlib-logger`,
and it matches only the *acquisition* of a stdlib logger
(`devtools/patterns/legacy-stdlib-logger.yml:1`). Green therefore means "no
module acquires a stdlib logger directly" — not "no module logs prose". 100
modules still log through the `get_logger` wrapper, which the rule does not
match; when structlog is unconfigured that wrapper is `_StdlibBoundLogger`.
Its `bind()` keeps validated context, and each call validates its keywords
(only when the record is enabled and at or above the event threshold) and
hands the accepted fields to the stdlib bridge, which emits them on the
`stdlib.record` event (`polylogue/logging.py:213-251`;
`polylogue/logging.py:1065-1098`). The message itself is still prose in
`error_detail`, so a module logging through this wrapper is not converted;
detecting those call sites needs a second rule with its own baseline; see
`docs/structured-logging.md:285`.

## Domain derivations

The typed kernel validates prerequisite names against the supplied ordered domain list. It pages required and excess keys, inspects authoritative output, computes outside the writer lease, and admits each replacement through the writer bridge. Process-local continuation state is disposable. A partially consumed page retains only its bounded unconsumed key suffix and the next-page cursor; resumption reinspects those exact keys rather than offsetting a fresh query whose demand rows may have disappeared. Smaller resumed budgets split that suffix without losing its remaining keys. Reports distinguish pending policy work from failed attempts (`polylogue/daemon/derivation.py:375-428`; `polylogue/daemon/derivation.py:481-498`; `polylogue/daemon/convergence.py:110-123`).

Raw observations use the same owner for admitted raw-to-logical membership. FTS retains canonical triggers, identity membership and the FTS refresh guard; per-session replacement joins exact canonical session membership. Its selected global orphan partition runs at low cadence and streams its binding, but still requires archive-wide scan and transaction work (`polylogue/daemon/raw_observation_owner.py:1`; `polylogue/storage/fts/derivation.py:660-690`; `polylogue/operations/fts_derivation.py:1`).

Embeddings replace one message reference atomically. Validity requires vector presence, full recipe identity and the exact message semantic hash. Provider work occurs outside publication admission. Attempt and cost receipts remain operation evidence (`polylogue/storage/embeddings/derivation.py:400-423`; `polylogue/daemon/embedding_owner.py:1`).

Session counters share one thirteen-measure declaration. Canonical writes recompute from stored messages, and the session-summary adapter inspects and replaces the same partition (`polylogue/storage/derived/session/summary.py:91-105`). Session-profile publication uses its existing domain adapter and shared owner (`polylogue/daemon/session_profile_composition.py:37-66`; `polylogue/storage/derived/session/derivation.py:1`).

## Remaining stage execution

The generic stage engine remains for optional Sinex publication, raw-authority cache warming, attachment acquisition, Claude workflow, delegation evidence and standing queries. It still has path/session callbacks, barriers and stage state. Removing these requires moving each surviving product responsibility to its owner; the domain adoption does not establish complete stage retirement (`polylogue/daemon/convergence_stages.py:424-470`; `polylogue/daemon/convergence.py:733-764`).

The stage walk itself runs off the writer lease. Each stage declares how it reaches the writer: `bridged` means it computes, downloads and drains outside admission and brackets only its short publication with `admit_stage_write`; `whole_execute` is the named residual for a stage that has not split compute from publication yet, and the engine holds the writer across its whole `execute`. Read the field, not the caller's control flow, to know which a stage is (`polylogue/daemon/convergence.py:690-697`; `polylogue/daemon/convergence.py:760-771`; `polylogue/core/stage_admission.py:59-70`). The live route no longer calls the stage pass at all: page admission takes no writer hold of its own, and the daemon's own stage walk owns the generic pass (`polylogue/daemon/convergence.py:704-712`).

`convergence_debt` remains disposable retry state for those surviving stage callers. The generic drain excludes domain-owned stages through `_OWNED_DEBT_STAGES` (`polylogue/daemon/cli.py:143-151`) and filters them before retry (`polylogue/daemon/cli.py:1728-1734`). FTS, embeddings, raw parsing and session profiles therefore do not use the generic stage rows as publication authority. Raw retention has its own live-ingest retry owner (`polylogue/daemon/cli.py:137-142`).

## Cadence loops

Every declared `PERIODIC` service runs through one runner rather than its own `while True`. The runner owns the sleep order (`run_first`), the existence guard (`precondition`, a recorded skip rather than a silent `continue`), the error policy (`record` keeps the cadence, `propagate` lets a schema-recovery signal reach the supervisor), jitter, and the startup gate. Per-loop last-run, next-due, last-error, skip and wakeup counts are the payload the status and metrics surfaces render, so an idle loop and a frozen one are distinguishable from outside the process (`polylogue/daemon/periodic.py:131-146`; `polylogue/daemon/periodic.py:159-171`). A loop given a `wakeup` event shortens its wait when the in-process bus announces a committed write; the declared interval stays as its reconciliation tick, because bus delivery is an optimization and never authority (`polylogue/daemon/event_bus.py:29-46`).

## Readiness and intake

Readiness derives from domain inspection and is reported separately from operation health. FTS does not consult a freshness ledger, and debt cannot certify insight readiness (`polylogue/daemon/fts_status.py:162-168`; `polylogue/readiness/claim_guard.py:1-26`; `polylogue/storage/sqlite/archive_tiers/archive.py:1`).

Hook capture is two ordinary steps, not a route of its own. Producers append one line per event to a per-process NDJSON carrier; `hook_carrier_watch_sources` exposes one carrier directory per harness and the fair-intake dispatcher's ordinary file adapter admits them under a `hook_carrier` class, so the durable cost is paid once per carrier revision rather than once per event (`polylogue/sources/live/watcher.py:270-293`; `polylogue/operations/intake_adapters.py:1900-1917`). `HookEventsDerivation._inspect`, `compute` and `publish` implement the derivation keyed by carrier raw id: it decodes the retained bytes, compares the coordinates they imply against the recorded ones, and publishes the missing events in one source-tier transaction with no blob publication (`polylogue/storage/derived/hook_events.py:250-277`; `polylogue/storage/derived/hook_events.py:292-373`; `polylogue/storage/sqlite/archive_tiers/source_write.py:944-1026`).

`FairIntakeDispatcher._service_class` owns page planning and admission: it discovers a bounded page per class, plans it against the class's byte share, and hands the whole page to one adapter call, which runs one `ingest_files` batch under one writer hold and one embedding/session-profile convergence pass for the page, both off that hold (`polylogue/daemon/intake.py:461-528`; `polylogue/operations/intake_adapters.py:887-892`; `polylogue/operations/intake_adapters.py:1068-1077`; `polylogue/sources/live/watcher.py:1298-1328`). Outcomes stay per item, read back from `LiveBatchMetrics` by path, so the deficit, retry and isolation accounting is unchanged by the batching. `LiveWatcher._note_intake_hint` owns the filesystem hint: it bumps an intake revision and sets the dispatcher's wakeup (`polylogue/sources/live/watcher.py:574-587`).

`FairIntakeDispatcher._service_class` applies a process-local cooldown to repeated retryable failures. `FileIntakeAdapter.admit_page` keeps a stale cursor refusal retryable even when the same batch reports successful files. Terminal refusal isolates only the affected item (`polylogue/daemon/intake.py:607-665`; `polylogue/daemon/intake.py:590-605`; `polylogue/operations/intake_adapters.py:936-947`).

## Status evidence and diagnostic privacy

`overall_status_ok` in `operations/daemon_status.py` owns the verdict for daemon, pinned, and composed status. An acquired stale or unavailable snapshot refutes health even when the required archive operands are healthy. A refresh may replace the previous stale state with refreshing, but preserves explicit unavailable evidence. A pinned read passes `None` for optional runtime evidence it did not acquire.

Raw failure diagnostics use `core/status_error_privacy.py` on both daemon model construction and pinned source projections. Arbitrary exception prose cannot distinguish a relative path from an absolute path concatenated with a preceding word. Slash-bearing prose is concealed through its quote or line boundary, including spaces and Unicode. Drive, rooted and UNC Windows diagnostics are concealed as well. A drive-letter prefix owns repeated forward separators (`C://...`) before inferred URL recognition; one-letter slash schemes in arbitrary prose are therefore treated as Windows paths. A concatenated forward-drive spelling such as `prefixC://...` is indistinguishable from a custom scheme in arbitrary prose and follows whole-scheme URL ownership; drive identity is never inferred from its final letter. Producer-declared relative backslash paths retain their exact span. A raw backslash ends an inferred network URL token. Ordinary unambiguous standalone parsed network URLs retain their scheme, authority, path and query. A URL-shaped substring inside an already-started local path grants no exemption. Only the enclosing quote ends a quoted path tail; an alternate quote inside a filename does not. A diagnostic constructor that owns an exact relative-path span may declare it using `RawFailureSample.relative_path_spans`; declarations are checked and excluded from serialization. Component snapshot errors, cached frame errors, unstructured health diagnostics, convergence-debt errors, live-ingest diagnostics, and serialized periodic-loop errors, nested catchup stage/halt/settlement diagnostics, returned FTS unavailable reasons, cursor-ledger failures and embedding failure/run diagnostics use the same projection at their construction boundaries. Authored health ratios and instructions retain their structural meaning; exception diagnostics and error-detail insertions are sanitized before they enter those messages. Structured source/current paths keep their field semantics. The separate import failure contract currently has no production consumer of its sample builder and is outside this status projection.

Standalone insight freshness acquires its reader in `operations/status_insights.py`, closes it on success or failure, and projects typed SQLite unavailability without inventing counts. The pinned producer in `operations/daemon_status.py` continues to read only its supplied snapshot.

Unstructured prose cannot distinguish a URI path such as `https://api.example.test/a,/opt/leaf` from a URL followed by a comma-delimited local path. The sanitizer conservatively ends an exemption at comma, semicolon, pipe, colon, equals or bracket punctuation followed by an absolute-path-shaped suffix after the scheme delimiter, including an ambiguous separator at the end of an apparent authority. The validated closing bracket of an IP authority remains structural URL syntax; quotes, angle brackets and whitespace already delimit URL tokens. Ordinary URLs, punctuation without that ambiguous suffix, and percent-encoded path/query values retain their text. This is an explicit limit of arbitrary exception prose, not evidence about a structured URL field: declared URL fields retain their existing ownership and semantics. There is currently no diagnostic producer that supplies an independently owned URL span, so no unused URL-declaration API is introduced. Relative declarations also refuse Windows drive/root/UNC anchors. Existing 300-character service reasons and 80-character failure hints project their original display prefix before redaction, retaining those same display limits without processing discarded diagnostic tails.

### Collection and convergence ownership

`operations/daemon_metrics.py` owns archive-generation resolution, tier readers,
collector availability and Prometheus exposition. `daemon/metrics.py` only
adapts that product result to HTTP; process-local collectors retain their daemon
lifecycle. Claude workflow materialization and its current ops receipt are owned
by `operations/claude_workflow_convergence.py`. Source membership and quiet-file
eligibility are owned by `operations/session_source_membership.py`; retained
source paths use the configured durable root while session joins follow the
active index generation. An unreadable active generation raises instead of
serving membership from a conventional shadow index. `operations/sinex_convergence.py` composes publication
and its primary-mode derivation barrier. FTS readiness acquisition and publication
are owned by `operations/fts_derivation.py`; the daemon stage schedules them.

The ordinary pinned status payload includes an `attachments` readiness component
scoped to owed Drive references. Its `unresolved_identity` count comes from
unfetched references with contested native identity, using the same predicate
as attachment convergence. Positive counts degrade status even when no
transport request is executable. A failed inspection is unknown with no
invented count. Resolved identity and terminal attachment absence contribute
zero contested obligations; they do not restart blocked transport work.

Catch-up stage-event history carries `stage_events_available` and a typed
unavailable reason. Empty readable history remains available; failed authority
keeps the mode degraded. Status and workload telemetry read only current ops
tables; unavailable counts never stand in for exact zero.

Normal client read and mutation requests bind the installed client version before their single operation POST. Index-dependent operations also bind the canonical Index schema version. User-only setting get/list and tier backup/restore omit an implicit Index precondition; an explicitly supplied precondition still reaches the resident validator unchanged. The client resolves those expectations after the existing peer-checked socket connects, so an absent daemon does not load the storage or version graph. Explicit caller preconditions remain unchanged. Status discovery and the original operation status, await, cancel and result controls remain available without inferred version preconditions, so a client can inspect an incompatible daemon or recover already accepted work. Session-delete preview cancellation remains a version-bound mutation.

Receipt waits bind the control request execution deadline to its actual `timeout_ms` wait budget. The transport retains its existing response allowance, so the forced final receipt read after an exhausted completion budget does not inherit the normal 30-second control deadline. Accepted mutation identity, cancellation and durable outcome reconciliation remain unchanged.


### Resident mutation selection

Delete previews and combined mark commands lower the query once into the resident selection owner. That owner pins Index and User together, checks the canonical selection outcome, and streams distinct session identities into a private disk relation. First selection requests one row per page, singleton selection at most two; ranked hits are deduplicated at session grain before cardinality is decided. All selection follows the canonical continuation in the same snapshot. Before durable acceptance, the admitted writer compares the selected frame again, including User overlay changes, and refuses a changed frame.

Tags, metadata, tag removal, marks and notes are validated as one command. Every bounded actuator plan and authorization is sealed before the first effect. Delete authorization and execution reference their originating resident request IDs rather than arrays of tokens. Scalar summaries retain the exact total and a display sample; lifecycle responses aggregate the entire batch and page its part details with `parts_offset`, `parts_limit` and `next_parts_offset`. A failed later part preserves actual committed effects and the untouched suffix count. Sampling never determines mutation membership.

### Complete operation response delivery

UDS and HTTP operation responses encode into private temporary scratch before headers are published. The exact Content-Length is then delivered in fixed transfer chunks; the client consumes the complete framed body and incrementally decodes JSON before exposing a result. A permitted individual row or value does not fail because the response exceeds 8 MiB. Row pagination, pinned view authority, strict operation models and mutation recovery remain unchanged. Encoding, malformed framing, disconnect and cancellation retire temporary scratch. Result validation walks the wire values and uses the existing models with their declared JSON tuple, string-enum and ISO-datetime forms; it does not reencode the entire result on either peer. Product Python values and hydrated validation models still contribute memory proportional to the result. This change does not qualify bounded whole-route memory or the separate resident Excision document producer.

Backup and verified restore bind the daemon version and control authority without an implicit Index version precondition. Their declared tier copy and recovery paths remain available when the derived Index is missing or skewed. An explicitly supplied Index precondition is preserved and refused when that control snapshot cannot prove it.

Context compilation submits its disposable scheduler ledger with the compilation start time in milliseconds. Every ledger producer supplies that time explicitly; the resident writer preserves it and refuses a request that omits it rather than inventing an epoch-zero observation.

## Resident insight pages

The eleven registered `analyze insights` list commands call `insights.list`
through the daemon. Its closed discriminated request and result branches use
the registry's existing query and item models. The canonical page reader is
shared with the Python facade and executes on the resident pinned archive
snapshot: origin tag rollups are merged before paging, and cost estimates are
enriched and filtered before paging. Missing daemon or unavailable insight
authority refuses instead of opening a local archive. `ops insights status` also calls the resident `insights.readiness` route on
the same pinned reader, preserving the canonical selected coverage and convergence
verdict. Its named pending-convergence outcome remains visible with zero rows.
The `ops insights audit` command also uses resident `insights.rigor`: every registered product is sampled on that same pinned reader, including explicit uncovered or exempt entries. Per-product read failure remains a named degraded outcome. Export and health commands retain their separate existing routes.
