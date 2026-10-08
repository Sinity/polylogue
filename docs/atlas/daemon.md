# Daemon

## Runtime ownership

HTTP mutations first acquire the existing writer bridge admission. Their bodies
run on the coordinator's writer worker under explicit child-task delegation;
they do not enter the read compute pool or acquire another writer gate. A client
wait deadline reports an indeterminate result and leaves accepted work owned.
If SQL cleanup fails, that original worker remains available for cleanup before
the next mutation can acquire physical custody; HTTP reports retryable
`writer_sql_unsettled` with status 503. The preceding mutation may have committed. Read routes keep their compute
admission and read cancellation behavior. Writer hold duration is telemetry;
elapsed time alone cannot reject a progressing append operation. New work
checks cancellation, while rollback, close and failed-close retry remain
available to the original owner. A supervised service constructs its coroutine
inside the owned task, so cancellation before startup leaves no unawaited
watcher coroutine.

Before a new mutation begins, recovery discovery reads orphaned operations and
unrouted file-replacement plans through the settled Audit continuity view. It
does not acquire writer custody on the preparation creator. Discovery already
inside an admitted Audit command instead borrows that command's original
connection under its same-root writer lease. Both discovery scopes end before
any recovery actuator runs (`AuditRepository.recovery_discovery_read`;
`OperationExecutor._resolve_dead_operations`).

Recovery resolution events name the resolver: authenticated request recovery uses
the request principal, while automated startup uses `daemon:recovery`. The
continuity command retains this identity for crash replay; the original
`operation_runs.actor_ref` remains the actor who began the operation.

The daemon holds writer/rebuild exclusion for its lifetime. `DaemonWriteCoordinator` serializes publication and retains ownership until a cancelled operation actually terminates. `DaemonAPIHTTPServer.execution_kernel` is passed to the UDS server and to daemon derivation owners; their `DaemonWriteThreadBridge` instances use the same coordinator (`polylogue/daemon/cli.py:2653-2658`; `polylogue/daemon/cli.py:2725-2750`; `polylogue/daemon/http.py:5697-5734`; `polylogue/daemon/write_coordinator.py:772-790`).

`run_daemon_services` is the service composition entry point (`polylogue/daemon/cli.py:1953`). Its composition state declares `session_profile_callback` and `embedding_callback` (`polylogue/daemon/cli.py:2595-2596`), and constructs the `FtsConvergenceOwner` for startup work (`polylogue/daemon/cli.py:2823-2840`). FTS runs at startup and periodically; session profiles run after admitted ingest and during the periodic sweep; embeddings use watcher scopes and the periodic backlog owner. `run_daemon_services` hands both callbacks to the live watcher (`polylogue/daemon/cli.py:2988-3003`); after an admitted batch `FileIntakeAdapter.admit_page` calls the watcher's lease-free embedding and profile convergence (`polylogue/operations/intake_adapters.py:1068-1077`; `polylogue/sources/live/watcher.py:1334-1356`); the periodic sweep and backlog services are registered in `periodic_services` (`polylogue/daemon/cli.py:2888-2943`). These are source-route facts, not live deployment evidence.

Convergence emits structured events (`emit`/`span` from `polylogue/logging.py`) rather than free-form log lines: field names pass an allowlist and quarantined names are stripped from both rendered forms when `POLYLOGUE_LOG_REDACT=1` is set, so a rebuild is read from named events such as `daemon.barrier.failed` and their typed fields (`polylogue/logging.py:482-486`; `polylogue/logging.py:770-781`; `polylogue/daemon/convergence.py:1120-1126`).

Each daemon run has one run id, bound into the logging run context before the first event and written as the `daemon_lifecycle` row's `run_id`, so the row, its heartbeats, the status projection of that row and every event of the run join on it (`polylogue/daemon/cli.py:1887`).

The SSE replay ledger `daemon_events` is a resume buffer in the disposable ops tier with no row count or age bound. The daemon process owns it, because it serves every SSE stream and so knows every live cursor; an open stream registers its cursor, and each emit in the owning process removes the rows every live subscriber has read that are either granular topic frames or records superseded by a newer row of their kind, keeping the newest row of each superseded record kind for its in-process readers. Browser capture-health reports are stored atomically in the separate `capture_health_history` ops table and their consumed resume frames are pruned. The CLI streams history in pages; receiver GET accepts positive `page_size` and opaque `cursor` and returns `next_cursor`. The history owner returns at most 100 reports per page, with a continuation for the remainder regardless of requested page size. CLI JSON is staged in a private temporary file and published after the requested walk succeeds; text output streams. Continued pages exclude newly appended reports and refuse `history_cursor_reset` after ops replacement. The events storage owner classifies history reads and transactions: schema skew remains a permanent refusal, transient storage faults return `capture_history_unavailable`, and deterministic faults return `capture_history_storage_failed`; HTTP and CLI adapt those types. The highest removed id is kept as a watermark, and a `Last-Event-ID` below it is answered with a typed `aged_out` resync envelope rather than a short page (`polylogue/daemon/events.py`: `EventSubscriberRegistry`, `prune_daemon_events`, `_cursor_refusal_reason`; `polylogue/daemon/events_http.py`: `_stream_events`).

Each ended run gets one termination receipt, reconciled by the next start as its own `termination_reconciliation` service: the run's lifecycle row records its host identity (pid, boot id, service-manager invocation, cgroup instance and `memory.events` baseline), and the receipt classifies the end from the stop marker, the manager's unit result, kernel and `systemd-oomd` kill records naming that pid or cgroup, the cgroup counters and the boot id, citing each source it used and naming each it could not (`polylogue/operations/daemon_termination.py`: `classify_termination`). `lifecycle_status` carries the newest receipt and the runs still awaiting one.

Correlation crosses the compute boundary explicitly. The shared bounded
compute adapter captures the current submitter's context for each physical
call and restores it on return, failure and cancellation. A reused worker
therefore reports its current operation, regardless of the interpreter's
`thread_inherit_context` default. Nested pure work borrows the same physical
reservation in an isolated copy of that context. Writer capability remains
subject to exact owner task, thread and explicit grant checks; copied logging
correlation does not authorize a writer (`core/compute.py`; `logging.py`;
`storage/sqlite/write_lease.py`).

Ordinary machine exchanges reserve their actual received request-body bytes
in the existing compute admission budget. Direct calls count canonical encoded
request bytes with the shared streaming encoder. The reservation lasts until
physical completion or queued cancellation. This is wire-byte accounting;
decoded request heap, ingress peak memory and read/result memory require their
own measurements and ownership proof (`daemon/uds.py`; `daemon/operation_runtime.py`).

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

The live watcher passes acquired Raw IDs to the supplied resident Raw owner for retained publication. When derived schema authority blocks that owner, the watcher keeps Source acquisition active without opening a replacement parser or publication owner.

The shared compute adapter settles each operation's future even when executor submission races shutdown or the executor cancels work before its worker starts. Cancelling shutdown stops scheduler dispatch. Graceful shutdown closes admission, drains already-admitted work, and closes the executor only after those reservations are released (`polylogue/core/compute.py`).

## Domain derivations

The typed kernel validates prerequisite names against the supplied ordered domain list. It pages required and excess keys, inspects authoritative output, computes outside the writer lease, and admits each replacement through the writer bridge. Publication adopts the coordinator's delegation on the existing compute worker, so preparation observers retain their creator. Its joined native cleanup boundary drains publication handles before the delegation and writer gate retire. Process-local continuation state is disposable. A partially consumed page retains only its bounded unconsumed key suffix and the next-page cursor; resumption reinspects those exact keys rather than offsetting a fresh query whose demand rows may have disappeared. Smaller resumed budgets split that suffix without losing its remaining keys. Reports distinguish pending policy work from failed attempts (`polylogue/daemon/derivation.py:375-428`; `polylogue/daemon/derivation.py:481-498`; `polylogue/daemon/convergence.py:110-123`).

Raw observations use the same owner for admitted raw-to-logical membership. FTS retains canonical triggers, identity membership and the FTS refresh guard; per-session replacement joins exact canonical session membership. Its selected global orphan partition runs at low cadence and streams its binding, but still requires archive-wide scan and transaction work (`polylogue/daemon/raw_observation_owner.py:1`; `polylogue/storage/fts/derivation.py:660-690`; `polylogue/operations/fts_derivation.py:1`).

Embeddings replace one message reference atomically. Validity requires vector presence, full recipe identity and the exact message semantic hash. Provider work occurs outside publication admission. Attempt and cost receipts remain operation evidence (`polylogue/storage/embeddings/derivation.py:400-423`; `polylogue/daemon/embedding_owner.py:1`).

Session counters share one thirteen-measure declaration. Canonical writes recompute from stored messages, and the session-summary adapter inspects and replaces the same partition (`polylogue/storage/derived/session/summary.py:91-105`). Session-profile publication uses its existing domain adapter and shared owner (`polylogue/daemon/session_profile_composition.py:37-66`; `polylogue/storage/derived/session/derivation.py:1`).

## Remaining stage execution

The generic stage engine remains for optional Sinex publication, raw-authority cache warming, attachment acquisition, Claude workflow, delegation evidence and standing queries. It still has path/session callbacks, barriers and stage state. Removing these requires moving each surviving product responsibility to its owner; the domain adoption does not establish complete stage retirement (`polylogue/daemon/convergence_stages.py:424-470`; `polylogue/daemon/convergence.py:733-764`).

The stage walk itself runs off the writer lease. Each stage declares how it reaches the writer: `bridged` means it computes, downloads and drains outside admission and brackets only its short publication with `admit_stage_write`; `whole_execute` is the named residual for a stage that has not split compute from publication yet, and the engine holds the writer across its whole `execute`. Read the field, not the caller's control flow, to know which a stage is (`polylogue/daemon/convergence.py:690-697`; `polylogue/daemon/convergence.py:760-771`; `polylogue/core/stage_admission.py:59-70`). Live append and full ingest still invoke the generic stage pass through `_converge_paths`; the daemon also runs it for owed recovery. Successful debt settlement must name the actual evaluated subject and stage, rather than infer session completion from a source path (`polylogue/sources/live/batch.py`, `_converge_paths`).

`convergence_debt` remains disposable retry state for those surviving stage callers. The generic drain excludes domain-owned stages through `_OWNED_DEBT_STAGES` (`polylogue/daemon/cli.py:143-151`) and filters them before retry (`polylogue/daemon/cli.py:1728-1734`). FTS, embeddings, raw parsing and session profiles therefore do not use the generic stage rows as publication authority. Raw retention has its own live-ingest retry owner (`polylogue/daemon/cli.py:137-142`).

## Cadence loops

Every declared `PERIODIC` service runs through one runner rather than its own `while True`. The runner owns the sleep order (`run_first`), the existence guard (`precondition`, a recorded skip rather than a silent `continue`), the error policy (`record` keeps the cadence, `propagate` lets a schema-recovery signal reach the supervisor), jitter, and the startup gate. Per-loop last-run, next-due, last-error, skip and wakeup counts are the payload the status and metrics surfaces render, so an idle loop and a frozen one are distinguishable from outside the process (`polylogue/daemon/periodic.py:131-146`; `polylogue/daemon/periodic.py:159-171`). A loop given a `wakeup` event shortens its wait when the in-process bus announces a committed write; the declared interval stays as its reconciliation tick, because bus delivery is an optimization and never authority (`polylogue/daemon/event_bus.py:29-46`).

## Readiness and intake

Readiness derives from domain inspection and is reported separately from operation health. FTS does not consult a freshness ledger, and debt cannot certify insight readiness (`polylogue/daemon/fts_status.py:162-168`; `polylogue/readiness/claim_guard.py:1-26`; `polylogue/storage/sqlite/archive_tiers/archive.py:1`).

Hook capture is two ordinary steps, not a route of its own. Producers append one line per event to a per-process NDJSON carrier; `hook_carrier_watch_sources` exposes one carrier directory per harness and the fair-intake dispatcher's ordinary file adapter admits them under a `hook_carrier` class, so the durable cost is paid once per carrier revision rather than once per event (`polylogue/sources/live/watcher.py:270-293`; `polylogue/operations/intake_adapters.py:1900-1917`). `HookEventsDerivation._inspect`, `compute` and `publish` implement the derivation keyed by carrier raw id: it decodes the retained bytes, compares the coordinates they imply against the recorded ones, and publishes the missing events in one source-tier transaction with no blob publication (`polylogue/storage/derived/hook_events.py:250-277`; `polylogue/storage/derived/hook_events.py:292-391`; `polylogue/storage/sqlite/archive_tiers/source_write.py:944-1026`). In daemon intake, the compute submission binds `stage_write_admission`; the publisher checks inputs off-writer and rechecks its compact SQL binding after writer admission, then writes enrichment debt and source events through `DaemonWriteThreadBridge`. A standalone derivation has no bridge and stays inline (`polylogue/daemon/cli.py:3166-3190`; `polylogue/core/stage_admission.py:59-70`).

`FairIntakeDispatcher._service_class` owns page planning and admission: it discovers a bounded page per class, plans it against the class's byte share, and hands the whole page to one adapter call, which runs one `ingest_files` batch under one writer hold and one embedding/session-profile convergence pass for the page, both off that hold (`polylogue/daemon/intake.py:461-528`; `polylogue/operations/intake_adapters.py:887-892`; `polylogue/operations/intake_adapters.py:1068-1077`; `polylogue/sources/live/watcher.py:1298-1328`). Outcomes stay per item, read back from `LiveBatchMetrics` by path, so the deficit, retry and isolation accounting is unchanged by the batching. `LiveWatcher._note_intake_hint` owns the filesystem hint: it bumps an intake revision and sets the dispatcher's wakeup (`polylogue/sources/live/watcher.py:574-587`).

File discovery is bounded by each watch source's declared layout (`SourceLayout` in `polylogue/sources/source_layout.py`), which states the root-relative depth and position of every artifact kind; every `daemon_watch_sources()` entry carries one. `_ordered_children` descends only into a directory `WatchSource.admits_directory` allows and `_source_path_steps` admits only a file `WatchSource.accepts` places in the layout; everything else is reported once as an `outside_declared_layout` exclusion, so a nested copy of a provider tree is never walked (`polylogue/sources/live/discovery.py`). Entry kinds name the provider's `OriginSpec` artifact rule where one exists, and `layout_declaration_defects` checks that each entry's example resolves to that same rule. The live event filter and the cold-build production baseline use the same predicates. The live watch is installed non-recursively on exactly the directories a layout reaches (`LiveWatcher.watched_directories`); `_watch_changes` re-arms it when a reachable directory appears or a watched one disappears, and the creation event's intake hint covers files written before the new watch exists. Every `WatchSource` carries a layout: a canonical name resolves its declaration through `source_layout_for`, wherever its root is, so one-shot ingest (`ingest_sources_archive`) and the one-shot source walk (`sources/source_walk.py`) walk a root named `codex` exactly as `~/.codex/sessions` is walked. Any other name labels explicitly declared input (an export-only origin's directory, an operator drop) and is walked as an export drop, bounded by file format and never entering dot-directories, as the inbox is; schema inference walks default inputs by their watch source's layout and explicit `provider=path` inputs as explicit input.

`FairIntakeDispatcher._service_class` applies a process-local cooldown to repeated retryable failures. `FileIntakeAdapter.admit_page` keeps a stale cursor refusal retryable even when the same batch reports successful files. Terminal refusal isolates only the affected item (`polylogue/daemon/intake.py:607-665`; `polylogue/daemon/intake.py:590-605`; `polylogue/operations/intake_adapters.py:936-947`).

Transient source reads during admission or SQLite capture are deferred input,
not cursor failures. They retain retry debt and schedule another full intake
without consuming the finite failure budget, even when the file observation
stays unchanged across repeated faults. A new unreadable file has no accepted
coordinate for a cursor; its deferred path remains discoverable and owed by
the file adapter. Unsupported bytes and typed permanent refusals retain their
ordinary exclusion contracts (`polylogue/sources/live/batch_support.py`,
`polylogue/sources/live/batch.py`, `polylogue/sources/live/cursor.py`).
Read deferral owns `live_ingest_source_read` debt; retained acquisition or a
definitive current refusal settles that stage. Cancellation propagates through
the batch, file adapter and dispatcher with any grouped cleanup failures intact.

A current retained decode refusal remains a failed derivation outcome. Its exact raw coordinate, parser census, support status and trusted failure carrier are validated by the canonical raw adapter. A later deliberate pass reports that same typed refusal from metadata without parsing the bytes again. Fair intake excludes the exact terminal item and discovery leaves it out of retry backlog; infrastructure failures and unavailable exact-key outcomes remain retryable (`polylogue/storage/derived/raw.py`, `polylogue/daemon/derivation.py`, `polylogue/operations/intake_adapters.py`).

## Status evidence and diagnostic privacy

The WebUI observability monitor retains snapshot frame errors separately from refresh errors in both bootstrap and direct status observations. Catch-up mode `idle` overrides a retained previous phase in its current-phase display.

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

The Unix operation listener keeps accepted sockets until their original handlers physically finish. Slow request bodies and response consumers have no transport deadline. Shutdown stops new handler admission, interrupts retained socket reads and writes, then joins the original handlers while the runtime owner loop remains available to settle admitted work.

Receipt waits bind the control request execution deadline to its actual `timeout_ms` wait budget. The transport retains its existing response allowance, so the forced final receipt read after an exhausted completion budget does not inherit the normal 30-second control deadline. Accepted mutation identity, cancellation and durable outcome reconciliation remain unchanged.

Session Excision prepares and executes under the original request-bound authority,
then retains the complete canonical domain receipt before retiring its source
witness. The bounded mutation result carries scalar counts and a document identity
(request ID, byte length and SHA-256). `operation.result` pages that same retained
product under its original principal and archive identity; retrieval never reruns
the mutation. A failed terminal metadata transfer retains custody for retry.
The CLI stages and verifies the complete length, digest and UTF-8 before emitting
machine JSON with `domain_receipt`; human output uses the scalar counts. Failed
product delivery preserves the committed mutation receipt and reports a delivery
failure rather than a new mutation refusal.

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
The `ops insights audit` command also uses resident `insights.rigor`: every registered product is sampled on that same pinned reader, including explicit uncovered or exempt entries. Per-product read failure remains a named degraded outcome. `ops insights hermes-health` uses resident `insights.hermes_health`, composing the existing read-only probes against the configured Hermes root. Missing derived tiers remain explicit measurement gaps; this diagnostic binds daemon and archive identity without requiring an Index precondition. A supplied Index version is observed through the supported read-only tier reader, then compared; a missing tier or wrong version refuses. The same explicit-only observation applies to User setting get/list. Python callers share the operations composer. Export commands retain their separate existing routes.

`ops insights fable-packet` uses resident `insights.fable_packet` with normal pinned Index/User authority and cancellation. Its exhaustive evidence paging and named unsupported outcomes are documented in [the query/read atlas](query-read-path.md).

`polylogued run --listener-info-path <path>` atomically creates private JSON without replacing an existing destination with the process `pid` and actual bound `listeners.api` and `listeners.browser_capture` host/port pairs (null for disabled listeners). Publication follows all enabled TCP binds and is socket readback, not archive readiness. Port zero requests distinct kernel-assigned listeners; equal positive ports on overlapping hosts remain refused. Component startup events report the assigned port for a zero request. A caller must identify its child and use a unique destination or reject stale process identity. The AgentCTL proof uses this output without closing and reacquiring port reservations.


Ingest diagnostics keep receipt units distinct: `parsed_raw_count` supplies
successful file counts and file rates; `materialized_count` supplies session
throughput. A measured zero file count does not use an older stage-event value.
Stage-event queued/needed file denominators take precedence over parsed counts.
Finished-batch durations and throughput include `completed_with_failures` as
well as `completed`; failed and interrupted attempts remain separate statuses
and do not enter that duration/rate population. Prometheus attempt counts expose
every declared operation-run status. The pinned workload projection applies the
same completion population without reopening its supplied reader.

The workload probe qualifies thread and latency surfaces by row coverage and
readability. Thread views must be readable and their session-profile inputs
complete. Latency coverage uses the canonical quiet-window missing-row and
orphan-row checks, together with profile-input coverage. Read failures leave
the derived readiness unchecked; a readable empty eligible scope is ready.
Planner row estimates are display evidence only. These checks do not certify
value freshness or replace the materializer's partition inspection.

The `polylogued` command registry lives in `daemon/commands.py`. Root help and
version do not initialize service implementations. Selected service commands
retain their original runtime callbacks and options. Status makes one resident
operation request and reports typed absence or refusal instead of recomputing a
local view. The guided path keeps `polylogued run` in another terminal before
issuing its resident transcript read.

Cold command branches defer canonical Source vocabulary and coordination
archive readers until their selected consumer needs them. Selected filters and
archive reads retain their existing owners, validation and connection cleanup.
Each process snapshot row classifies its executable and first Python module
from one complete shell parse. Malformed quoting retains whitespace executable
classification and refuses module inference; classification is local to that
observation.

The CLI's auxiliary archive reads also execute against the resident pinned
reader. Assertion export pages the original User authority and excision
planning remains a declared read. Identity reset prepares one frozen audited
preview on the resident writer; authenticated target pages read its immutable
ordinals and confirmed execution accepts only the preview reference.
Tutorial counts and summary aggregates use
`query.aggregate`, and archive-coverage summaries use `insights.list`.
Composed context images carry their selected read views through
`read.context-image`, so the daemon compiles messages, temporal evidence, and
chronicle excerpts on the same selected archive view.

Cold-build settlement runs after a complete quiescent fair-intake pass with no
pending discovery, local retry deadline, or durable intake retry. It does not
require a successful watched-file admission: operation imports may already
have written the inactive candidate. The settlement owner checks candidate
coverage and promotes a nonempty candidate or discards an empty one. Blocked
and retryable settlement retain their existing evidence and retry conditions.


Embedding orphan maintenance runs without an enabled purchase provider because
paid references can outlive Index replacement. A transaction proving that no
vectors, vector metadata, message references or status rows exist returns an
empty report before requiring active Index generation readiness. Populated
paid state still requires the active source-snapshotted Index before deletion
(`storage/embeddings/reconcile.py`; `daemon/embedding_backlog.py`).

## Configured Drive completion

Drive intake returns `DriveCatchupReport`: `complete`, `pending`, `retryable`,
`blocked`, or `unknown`. Completion requires a full paged listing and a separate
post-acquisition listing, exact native file/revision bindings to retained Raw,
and current materialization in the executing Index snapshot. A private disk
relation holds the full listing and per-file bindings. Its digest, denominator,
selection rule, resolved folder, and observation times travel in the report.
A measured empty folder has zero members. An absent or unfinished witness has
nullable counts and named gaps. Restart reconstructs the relation from a new
listing and retained Source; disposable cursor progress is never completion.

A parse time slice checkpoints retained work. Pending materialization yields
fairly and remains eligible without a failure attempt or an hourly poll.
Only `complete` starts the ordinary hourly poll. Transport failures retry;
blocked and unknown observations recheck evidence on a short cadence.
Cold promotion settles its declared local baseline independently. The
`configured_sources` status component and `claim_guard.converged` separately
withhold full readiness until configured Drive obligations are measured complete,
including when embeddings are disabled.
Resident collection retains an in-flight scan across its response deadline.
Its cache fingerprint binds both archive generations and the configured scope
and listing custody, so replacing a witness invalidates prior readiness even
when no archive file changed. Deferred membership remains pending; conflicting
membership is blocked debt, and typed terminal parser evidence blocks completion
even when it has no free-text diagnostic. Shutdown releases listing scratch
after compute custody drains.

Attachment status reads the supplied Source and Index snapshots and reports
allowed unfetched references, contested identity, unretained suppliers, acquired
objects, and terminal unavailable objects. Terminal provider refusals and
excision remain visible through their typed acquisition events; the stored
`unavailable` disposition does not distinguish their reasons. Stage scheduling
uses executable transport work, so blocked identity cannot keep retrying.


Bulk tag and metadata mutation plans retain original requested IDs and the
missing-ID gap separately from exact authorized session targets. Plan admission
has no session-count ceiling; the daemon pages publication work in groups of
256 without limiting the request. Ordinary
recovery replays only those targets and reports the frozen named gap, even
if a missing session has appeared since authorization. Correction recovery
checks the exact kind, payload, note and normalized author before replay;
an already committed matching effect preserves its creation and update times.

Long source preparation emits `daemon.work.progress` through the structured field registry. `unit_id` identifies an invocation and `productive_id` identifies the retry-stable source recipe; both are registered opaque identifiers. The fresh-build observer counts only counter advances above that recipe’s high-water. Repeated or reset retry counters do not prove progress, while advancing preparation remains observable before durable publication.

Cold-build generation events retain stable lifecycle reason tokens: `explicit_cold_build` for an explicit request, `empty_active_index_generation` for ordinary empty-index admission, and `interrupted_promotion` for promotion recovery. They use the existing registered `reason` field and token validation.

The daemon service composition owns its configured notification adapters for
its lifetime, including supervised health-service restarts. Equal settings,
equivalent backend-selection syntax and other adapters' setting changes
preserve email's hourly allowance. Changing an adapter's own notification
settings replaces it with a fresh allowance; removed destinations are not
cached. Invalid replacements refuse the dispatch
without sending through stale settings. Injected adapters and fanout keep
their existing behavior, and daemon restart starts a fresh in-memory allowance.

Generation promotion logs retain the opaque predecessor generation ID in the registered nullable `predecessor` field. FTS readiness stage terminals retain the boolean publication measurement in the registered `bound` field, alongside their existing ok, degraded or empty outcomes.

The browser receiver owns CaptureJob retirement through the existing
`BrowserCaptureHTTPServer.service_actions` lifecycle hook. Each server turn
drains one bounded leaf-row page and one artifact-directory quantum independently
of client request frequency. The existing eligible retention JSON holds a durable
internal retiring marker; scoped reads and mutations are fenced, discovery hides
the job, and same-intent creation returns retryable 503 until the unique parent
retires. Native assets and other leaves drain before acquisition/job parents,
so their deletion cannot cascade an unbounded membership. Each page commits
under the original receiver registry connection owner. Shutdown closes the
disposable frontier; restart resumes the marker and existing physical roots.
This receiver state is separate from the archive tiers and archive writer.
