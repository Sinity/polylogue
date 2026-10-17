# Configured Codex discovery and continuation

`daemon --config FILE [--until-idle]` accepts exactly one source selection. The existing `claude_main` selection keeps its single-location behavior. A Codex selection explicitly names a home:

```json
{"codex_home":{"home":"/absolute/disposable-codex-home","machine":"synthetic-machine","host":"synthetic-host"}}
```

Optional `rollouts`, `catalog` and `history` paths default to `sessions`, `state_5.sqlite` and `history.jsonl`. Relative paths, including `threads.rollout_path` values, resolve against the configured home. Parent traversal, paths outside that home and source symlinks refuse. No default Codex home is opened. Startup reads sources without provider repair. A running configured daemon advances this same source selection in bounded turns with a one-second pause between turns; each pass uses store-owned discovery and the existing single-owner capture and interpretation queues. `--json` emits a retention-only startup report while the daemon runs. `--until-idle` disables polling, waits for its startup-admitted rollout row effects, keeps prompt references pending and returns degraded exit 5 with their separate status.

Store enumerates `rollout-*.jsonl` independently of thread registration. It separately opens the configured SQLite catalog read-only, pins a read transaction and admits its `threads(id, rollout_path)` rows. A missing or malformed catalog does not prevent independently enumerated rollouts from being retained. Each readable admitted rollout becomes an immutable snapshot through the existing capture/import owner and keyed Codex interpreter. Changed bytes become another retained snapshot; this path does not claim observed revision lineage. Ordinary `origin:codex` discovery, `tool_calls where session:ID`, and returned operation references use the shared CLI and MCP readers. Unknown general Session and source coverage stays unknown.

`status.acquisition` describes this startup only. `accessible_empty` means a completed accessible traversal or empty file; `unavailable` carries a reason and `entries: null`. It never establishes a zero source population. A failed traversal may still have acquired an explicitly reported prefix. `retained_count` and `refused_count` account for successful and failed source capture attempts across rollouts, prompt history and explicitly configured physical history, independently of content deduplication. Catalog discovery availability remains separate. `retained` samples at most 32 input IDs and `refused` samples at most 4 failures, with explicit omitted counts; this presentation never limits completed acquisition. Availability counts name their units: rollout paths, thread rows or retained history inputs. An empty history input still counts as one retained input and reports `accessible_empty` for its bytes. Paths larger than 1024 encoded bytes are omitted with a BLAKE3 `path_key` of the original path bytes; truncated reasons are marked. Catalog rows, filesystem enumeration and source snapshots are separate observations, not one atomic provider snapshot. Removing an original rollout changes current acquisition evidence while retained operation citations remain readable.

`codex-prompt-history` owns retained `history.jsonl` bytes using the existing input/observation/extent schema. It has no conversation parser or canonical row settlement certificate. The sources crate alone projects complete supported JSONL records with nonempty `session_id` and string `text`; timestamps and prompt contents do not become Messages. Store supplies verified retained bytes and original machine/host/path/acquisition-time provenance. The engine exposes exact record byte ranges and BLAKE3 hashes under `pending_references` on status and queries. Reference counts describe supported prompt-record observations in retained snapshots, never distinct conversations or Messages. Public replies carry bounded reference and snapshot samples, observed counts, omitted counts and explicit scan/presentation completeness. A partial scan reports unknown totals and its observed prefix. Exact failed conversation reads filter by the requested Session before presenting reference details. Optional arrays yield to primary query rows at CLI/MCP presentation limits. No conversation Session row is manufactured for a prompt-only reference. A bounded read of that reference returns an unavailable conversation result with its pending source evidence, never a completed empty conversation or invented assistant response. Malformed records and incomplete tails stay explicitly counted.

When independently acquired conversation evidence publishes the same opaque provider Session identity, the reference reports `conversation_evidence`. Its prompt provenance remains separate and conversation coverage remains unknown. Duplicate prompt records are source observations, not additional sessions or conversational Messages. A regular restart without source configuration still exposes retained reference evidence. Failed provider acquisition does not erase earlier snapshots.

Startup admission bounds are 512 visited directory entries, depth 8, 512 catalog rows and 512 attempted combined rollout paths. A larger known union is deferred with `deferred_rollout_paths`, counting known prefix paths not attempted at startup; it is not a total provider population. This keeps startup usable for the running cursor. `--until-idle` reports that finite admission without starting continuation. Individual files and the catalog are at most 16 MiB; one startup captures at most 32 MiB. Prompt snapshots admit 1 MiB and 4096 references; public projection examines a prefix of at most 512 retained prompt snapshots, 4 MiB total prompt bytes and five seconds of work. Its reference sample admits 8 KiB and snapshot sample 4 KiB, with explicit omitted/unknown counts. Startup acquisition metadata samples fit 16 KiB independently of completed capture. The existing rollout parser and operation publication/read bounds also apply. Admission failures refuse or explicitly account for partial acquisition, never certify an empty population.

`scripts/check-codex-discovery.py --binary PATH --output NEW_DIR` exercises an exact packaged binary through ordinary CLI and actual SDK MCP processes. The declared `configured_codex_demo` operation retains command/reply files, daemon logs, binary SHA-256 and a report. Controls cover an empty catalog with an omitted rollout, prompt-only pending evidence, restart, removing only the owned rollout, later conversation identity, changed/malformed prompt snapshots, inaccessible versus empty sources, corrupt catalog catalog-relative paths, 64 prompt references and 301 readable rollout paths that exceed the public detail samples. That demonstration does not qualify physical history acquisition, catalog reconciliation, rewrite lineage, source defaults or activation.

Running continuation exposes `status.continuation` separately from the immutable startup acquisition report. Each last finished turn names catalog and rollout availability, exact admitted union-path prefix units, enumeration cycle and completion, examined/unchanged/acquired/failed file counts, charged source bytes, deferred candidates and history byte availability. `pass_active` marks current work; prior finished details remain historical until that turn ends. `partial` enumeration has an unknown total and keeps its cursor. `accessible_empty` is reserved for a completed empty accessible cycle. Source failures remain unavailable, while other retained reads remain usable. These observations never certify general conversation or Session coverage.

A store-owned cursor retains at most nine directory iterators and advances at most 512 directory entries per turn. It preserves position rather than restarting a capped prefix, starts another cycle after EOF, and resets when the configured root's identity changes. One admitted batch holds the union of at most 512 filesystem rollout paths and 512 catalog paths, plus one prompt path. Independent admitted prefixes remain owed; the batch contains at most 1025 candidates. It drains over as many turns as necessary before discovery advances. Per-turn limits are 512 evaluated candidates and 32 MiB of declared source bytes including one descriptor guard byte each. Unused capacity is released at turn end; the next turn receives its ordinary capacity. There is no process-lifetime snapshot or byte quota. `budget_exhausted` and `deferred_candidates` describe unfinished work, with ordinary polling continuing it automatically.

Fingerprints and stable-descriptor capture run outside the writer. Before reading provider content each turn, the existing owner checks its acquisition/refusal policy. Store's existing retained input and machine/host/path observation metadata recognize previously acquired identical content without creating another request, observation or input. This requires no growing in-memory path cache or new table. Changed candidates use the existing owner import slot, with expected length and content hash checked before source mutation. Required privacy/removal checks remain mandatory. Capture and interpretation retain existing owner fairness; the single polling worker shares cooperative daemon cancellation and joins on shutdown. No extra parser, writer, service or general watcher framework is registered.

The catalog remains a separate read-only finite snapshot of at most 512 rows, with explicit unknown/partial scope when that admission is exceeded. This route does not collect native database pages or repair catalog membership. Independent filesystem continuation can progress beyond a larger catalog or directory prefix. A finite enumeration turn and a complete provider conversation remain separate claims. Removing a provider file preserves retained snapshots and citations. A file waiting while stopped is considered by the next configured startup; `--until-idle` still owns only that startup snapshot.

`scripts/check-codex-continuation.py --binary PATH --output NEW_DIR`, exposed as `configured_codex_continuation_demo`, exercises two later batches through ordinary CLI and official-SDK MCP discovery/read without supplied answer IDs, including an omitted-catalog rollout. Controls also cross reusable turn capacity with many small waves, discover a later rollout beyond one directory prefix, revisit an appended supported rollout, distinguish partial/unavailable/completed-empty acquisition, preserve unchanged source effects and retained citations, catch a waiting restart batch, and interrupt active collection with SIGINT. Private reports record actual acquisition-to-query delay, daemon CPU/RSS/high-water memory, archive bytes, file descriptors and interventions. These finite measurements are not hard RSS or complete source-coverage claims; prior startup controls remain separate evidence.

## Physical history

An optional `codex_home.paginated_history` path explicitly selects a physical
thread-history database, subject to the same home/path rules. There is no default
path. `paginated_interval_seconds` is an integer from 1 through 86400, default
300. Startup reports physical `paginated_history` as pending until owner admission, or
unavailable when the path preflight fails. It opens no physical SQLite view before
the actor/socket exist. Dynamic continuation exposes `paginated_pending`,
`paginated_state` and the most recent exact retained `paginated_input`/length,
independently of startup detail samples. That input names acquired bytes, never a
current-provider frontier. Complete raw retention and supported
row interpretation are different outcomes. A valid unknown schema keeps its raw
input and reports an unavailable `no_parser_yet` interpretation; it cannot certify
an empty conversation. Existing interpretation admission remains 16 MiB/4096 rows.

The existing collector admits physical capture without waiting for its final
reply, so regular configured turns continue. The actor grants one parked history
worker reusable quanta, alternating queued regular capture with physical work and
serving a bounded command round between grants. Ordinary reads remain available.
One read-only SQLite transaction pins a read view before reading page count and every
page through `sqlite_dbpage`, including committed WAL content and exact page one.
An uncommitted provider write is excluded. No second descriptor reads the live
main file, and no temporary complete database copy is written. Read transactions
can delay WAL checkpoints; in rollback-journal mode they also hold a shared lock
that delays the provider writer. The demonstrated concurrent-writer acceptance
is WAL-specific; rollback-mode availability remains unqualified. Cancellation
and a 300-second cooperative active-turn guard refuse incomplete preparation.
A turn admits 8 MiB of conservative work charges or 512 checkpoints. Source and
verification work yield through explicit actor grants; parked time is excluded,
and there is no one-second physical-turn pause or lifetime raw-byte/turn quota.
These are cooperative checkpoints, not guaranteed shutdown latency or hard RSS/IO
bounds: blocking IO, frame compression and existing seek-table finalization are
not preempted. Seek metadata still scales with frame count. Library callbacks
without byte counts conservatively charge 64 KiB each; those charges are not
measured source/decoded bytes. No capped prefix becomes an input.

A first image is one complete plain CAS object. Later capture verifies one retained
complete base and compares against it with a 4 MiB retained-byte window. Every page
still different from that base goes into one current patch. Complete flat extents
reference only the base and that patch, never a chain of previous patches. More
than 8192 extents triggers a streamed full rebase from the same pinned provider
view. This caps publication metadata rather than raw database size. All bytes,
whole digest, complete ranges and CAS custody precede one selection transaction.
Interrupted preparation may leave unselected physical objects, never a selected
partial source. The live SQLite/encoder/checked-base state stays in the one
worker across grants; each turn does not reopen a source generation or reverify
the whole base. Process loss discards that live state. Existing `blobs/tmp/*.part`
files identify non-authoritative preparation and are never resumed as a page
cursor; a new attempt starts one fresh complete view. Such orphan candidates may
remain for file-layer recovery. Cumulative patch-storage amortization is not
implemented here.

Provider-page reads, base-verifier decoded bytes, predecessor range bytes,
canonical-layout rereads and newly retained changed bytes are separate costs;
smaller patch storage does not imply fewer source reads. Ordinary reconstruction
uses two request-capability object-affine expected-object readers for this
base/patch shape, while preserving current source, removal and blob-state fences.
Existing equal-content inputs reuse their verified immutable extent layout.
Returning to older bytes after a different observation still records acquisition
of that older image; it is not mistaken for unchanged current content.

Process-local main/WAL metadata only schedules work. Restart forces verification;
unchanged metadata can skip until an hourly full check. Changes wait for the
configured quiet interval or the hourly check. Failed attempts do not certify a
new stamp or postpone a due full check. Status distinguishes metadata skip,
waiting for quiet and actual byte verification; none certifies conversation
coverage. Polling does not repair the provider database or its thread catalog.
`--until-idle` waits for its one physical startup attempt and then interprets the
exact acquired input, or retains explicit acquisition unavailability. Its finite
120-second caller timeout reports pending work honestly; it does not certify an
idle/empty history or impose an acquisition-format size limit.

`scripts/check-configured-history.py`, declared as `configured_history_demo`, is
the candidate packaged CLI/MCP acceptance driver. It requires explicit
`--sdk-python` and `--sdk-client` paths for the pinned official joint SDK adapter,
freezes its source dependencies into the private result directory, and records
actual SDK version/context-close receipts separately from direct JSON-RPC controls.
Both routes discover operation references before exact fact/origin checks.
It independently checks scoped
rollout/history correspondence, committed WAL-only rows, an uncommitted distractor,
terminal failure/exit-code facts, later same-daemon publication, restart, provider
removal and unsupported-schema retention. Its answer oracle rejects damaged result,
row, verdict, session/turn scope, conflicting origin, uncommitted-presence and
invented-tool controls. The reusable-turn controls additionally test correlated
WAL commits across small quanta, actual regular capture/read service before
physical selection, earlier complete reads during pending work, cancellation and
killed-process fresh-view retry. Qualification requires the exact candidate
package and public journey; an earlier package does not establish changed semantics.
The driver records daemon/MCP reaping and runtime removal before reporting success.
These finite controls do not qualify private corpora, multi-GB resource costs,
rollback-writer availability, cumulative patch-storage amortization or full Sessions.
