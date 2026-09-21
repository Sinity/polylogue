# Daemon

## Runtime ownership

The daemon holds writer/rebuild exclusion for its lifetime. `DaemonWriteCoordinator` serializes publication and retains ownership until a cancelled operation actually terminates. HTTP, UDS and derivation owners share the process compute adapter and writer bridge (`polylogue/daemon/cli.py:1985-1997`; `polylogue/daemon/write_coordinator.py:288-311`; `polylogue/daemon/convergence.py:125-146`).

`run_daemon_services` composes FTS, embedding and session-profile callbacks once. FTS runs at startup and periodically; session profiles run after admitted ingest and during the periodic sweep; embeddings use watcher scopes and the periodic backlog owner (`polylogue/daemon/cli.py:2562-2574`; `polylogue/daemon/cli.py:2629-2643`; `polylogue/daemon/cli.py:2664-2681`; `polylogue/daemon/cli.py:2746-2760`). These are source-route facts, not live deployment evidence.

Convergence emits structured events (`emit`/`span` from `polylogue/logging.py`) rather than free-form log lines: field names pass an allowlist and quarantined names are stripped from both rendered forms when `POLYLOGUE_LOG_REDACT=1` is set, so a rebuild is read from named events such as `daemon.barrier.failed` and their typed fields (`polylogue/logging.py:349-364`; `polylogue/logging.py:400-409`; `polylogue/daemon/convergence.py:959-975`).

Correlation crosses the compute boundary explicitly. Neither `threading.Thread`
nor `ThreadPoolExecutor.submit` copies contextvars, so both derivation-kernel
submits wrap their `partial` in `propagate(...)`; without it the work runs on a
pool thread with an empty context and its events lose the run's correlation id
(`polylogue/daemon/convergence.py:185-195`; `polylogue/daemon/convergence.py:283-294`; `polylogue/logging.py:325-345`).

Read this as a statement about the daemon's convergence path, not about the
tree. The ratchet is a `devtools gate patterns` rule, `legacy-stdlib-logger`,
and it matches only the *acquisition* of a stdlib logger
(`devtools/patterns/legacy-stdlib-logger.yml:1`). Green therefore means "no
module acquires a stdlib logger directly" — not "no module logs prose". 100
modules still log through the `get_logger` wrapper, which the rule does not
match; when structlog is unconfigured that wrapper is `_StdlibBoundLogger`,
whose `bind()` is a no-op and which forwards only `exc_info`/`stack_info`/
`stacklevel`/`extra`, discarding every other structured keyword
(`polylogue/logging.py:165-166`; `polylogue/logging.py:184-186`). Closing that
gap needs a second rule with its own baseline; see
`docs/structured-logging.md:261`.

## Domain derivations

The typed kernel validates prerequisite names against the supplied ordered domain list. It pages required and excess keys, inspects authoritative output, computes outside the writer lease, and admits each replacement through the writer bridge. Process-local continuation state is disposable. Reports distinguish pending policy work from failed attempts (`polylogue/daemon/derivation.py:375-428`; `polylogue/daemon/derivation.py:481-498`; `polylogue/daemon/convergence.py:110-123`).

Raw observations use the same owner for admitted raw-to-logical membership. FTS retains canonical triggers, identity membership and the FTS refresh guard; per-session replacement joins exact canonical session membership. Its selected global orphan partition runs at low cadence and streams its binding, but still requires archive-wide scan and transaction work (`polylogue/daemon/raw_observation_owner.py:1`; `polylogue/storage/fts/derivation.py:660-690`; `polylogue/operations/fts_derivation.py:1`).

Embeddings replace one message reference atomically. Validity requires vector presence, full recipe identity and the exact message semantic hash. Provider work occurs outside publication admission. Attempt and cost receipts remain operation evidence (`polylogue/storage/embeddings/derivation.py:400-423`; `polylogue/daemon/embedding_owner.py:1`).

Session counters share one thirteen-measure declaration. Canonical writes recompute from stored messages, and the session-summary adapter inspects and replaces the same partition (`polylogue/storage/derived/session/summary.py:91-105`). Session-profile publication uses its existing domain adapter and shared owner (`polylogue/daemon/session_profile_composition.py:37-66`; `polylogue/storage/derived/session/derivation.py:1`).

## Remaining stage execution

The generic stage engine remains for optional Sinex publication, raw-authority cache warming, attachment acquisition, Claude workflow, delegation evidence and standing queries. It still has path/session callbacks, barriers and stage state. Removing these requires moving each surviving product responsibility to its owner; the domain adoption does not establish complete stage retirement (`polylogue/daemon/convergence_stages.py:526-563`; `polylogue/daemon/convergence.py:733-764`).

The stage walk itself runs off the writer lease. Each stage declares how it reaches the writer: `bridged` means it computes, downloads and drains outside admission and brackets only its short publication with `admit_stage_write`; `whole_execute` is the named residual for a stage that has not split compute from publication yet, and the engine holds the writer across its whole `execute`. Read the field, not the caller's control flow, to know which a stage is (`polylogue/daemon/convergence.py:704-712`; `polylogue/daemon/convergence.py:778-785`; `polylogue/core/stage_admission.py:59-70`). The live route no longer calls the stage pass at all: page admission takes no writer hold of its own, and the daemon's own stage walk owns the generic pass (`polylogue/daemon/convergence.py:704-712`).

`convergence_debt` remains disposable retry state for those surviving stage callers. The generic drain skips the stages named in `_OWNED_DEBT_STAGES`: FTS, embeddings, raw parsing and session profiles do not use its stage rows as publication authority, and raw retention keeps rows there as its own retry ledger but drains them from the live-ingest pass rather than here (`polylogue/daemon/cli.py:128-128`; `polylogue/daemon/cli.py:1626-1645`; `polylogue/sources/live/convergence_outcome.py:1`).

## Cadence loops

Every declared `PERIODIC` service runs through one runner rather than its own `while True`. The runner owns the sleep order (`run_first`), the existence guard (`precondition`, a recorded skip rather than a silent `continue`), the error policy (`record` keeps the cadence, `propagate` lets a schema-recovery signal reach the supervisor), jitter, and the startup gate. Per-loop last-run, next-due, last-error, skip and wakeup counts are the payload the status and metrics surfaces render, so an idle loop and a frozen one are distinguishable from outside the process (`polylogue/daemon/periodic.py:131-146`; `polylogue/daemon/periodic.py:159-171`). A loop given a `wakeup` event shortens its wait when the in-process bus announces a committed write; the declared interval stays as its reconciliation tick, because bus delivery is an optimization and never authority (`polylogue/daemon/event_bus.py:29-46`).

## Readiness and intake

Readiness derives from domain inspection and is reported separately from operation health. FTS does not consult a freshness ledger, and debt cannot certify insight readiness (`polylogue/daemon/fts_status.py:162-168`; `polylogue/readiness/claim_guard.py:1-26`; `polylogue/storage/sqlite/archive_tiers/archive.py:1`).

Hook capture is two ordinary steps, not a route of its own. Producers append one line per event to a per-process NDJSON carrier; the watcher exposes one carrier directory per harness and the fair-intake dispatcher's ordinary file adapter admits them under a `hook_carrier` class, so the durable cost is paid once per carrier revision rather than once per event (`polylogue/sources/live/watcher.py:299-320`; `polylogue/operations/intake_adapters.py:855-871`). The events themselves are a derivation keyed by carrier raw id: it decodes the retained bytes, compares the coordinates they imply against the recorded ones, and publishes the missing events in one source-tier transaction with no blob publication (`polylogue/storage/derived/hook_events.py:207-232`; `polylogue/storage/derived/hook_events.py:318-345`; `polylogue/storage/sqlite/archive_tiers/source_write.py:977-1010`).

`FairIntakeDispatcher` is the only intake authority: it discovers a bounded page per class, plans it against the class's byte share, and hands the whole page to one adapter call, which runs one `ingest_files` batch under one writer hold and one embedding/session-profile convergence pass for the page, both off that hold (`polylogue/daemon/intake.py:368-393`; `polylogue/operations/intake_adapters.py:278-300`; `polylogue/sources/live/watcher.py:1214-1240`). Outcomes stay per item, read back from `LiveBatchMetrics` by path, so the deficit, retry and isolation accounting is unchanged by the batching. The watcher itself owns no queue: a filesystem event bumps an intake revision and sets the dispatcher's wakeup (`polylogue/sources/live/watcher.py:544-560`).

Fair intake applies a process-local cooldown to repeated retryable failures. A stale cursor refusal remains retryable even when the same batch reports successful files. Terminal refusal isolates only the affected item (`polylogue/daemon/intake.py:301-375`; `polylogue/operations/intake_adapters.py:236-243`).
