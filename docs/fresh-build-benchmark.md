# Fresh-build benchmark

`devtools bench fresh-build` measures the ordinary cold build: an empty archive
root, the production daemon started as `polylogued run --cold-build-index`,
intake of a sealed source corpus through the fair-intake dispatcher, promotion
of the candidate index generation, and derived convergence until the archive
is terminal. It measures the product route, not a harness around its parts.

## Corpora

A corpus is a directory with `home/` (a stand-in `$HOME`: the daemon's typed
default sources resolve `~/.claude/projects`, `~/.codex/sessions` and
`~/.gemini/tmp` inside it), optional `exports/<name>/` directories whose
files each run copies into the scratch archive's inbox before the daemon
starts (the route `polylogue import` stages exports through), and
`manifest.json`, which seals every file's path, size
and SHA-256 under one digest, plus the mtime of each parser sidecar
(`tool-results/`, `tool-outputs/`), which parsing turns into the sidecar
event's timestamp; sampling keeps the source mtimes. Every run and component
re-hashes the tree against the seal and refuses an edited, added, missing or
re-timed file. Receipts compare
only when their corpus digests match.

| Kind | Command | Use |
| --- | --- | --- |
| sample | `corpus sample --out DIR --seed N --fraction F` | A seeded byte-fraction of each (origin, size bucket) stratum of real sources. A unit keeps its parser sidecars: a Claude Code session with its subagents and `tool-results/`, a Gemini CLI project with its `tool-outputs/`. Private. |
| synthetic | `corpus synthetic --out DIR --provider TOKEN (--sessions N | --bytes N) --seed N` | The same lazy session/chunk driver used by synthetic fixtures, from the package's measured workload profile. Missing package profiles refuse; neutral inferred packages qualify mechanics, not an operator workload. |
| files | `corpus files --out DIR [--export ORIGIN=PATH] [--hooks DIR --hooks-fraction F] FILE...` | Exactly the named real transcripts, e.g. one whale; each must be a file its source root's watcher admits. `--export` stages a ChatGPT or Claude.ai export under `exports/`. `--hooks` stages a deterministic fraction of a legacy hook spool for backlog timing. Private. |

Both are private: corpora, manifests and receipts stay outside the checkout
(the command refuses a path inside it), and only aggregate numbers leave the
machine.

## Running

```bash
devtools bench fresh-build run --corpus DIR --work EMPTY_DIR [--profile] \
    [--max-rss-mib N] [--max-promotion-s N] [--max-terminal-s N]
```

Corpus discovery and preparation, full or larger runs, and multiple-run
experiments use the bulk operations `scratch` or `fresh_build_bench`
(`agentctl job start polylogue fresh_build_bench -- run ...`). One explicitly
selected small diagnostic over an already-sealed corpus uses the normal-pool
`fresh_build_probe` operation:

```bash
agentctl job start OPERATION_CHECKOUT fresh_build_probe -- \
    --corpus DIR --work EMPTY_DIR --candidate FROZEN_CANDIDATE
```

`OPERATION_CHECKOUT` is a Polylogue checkout defining this operation;
`FROZEN_CANDIDATE` identifies the product candidate. The command already fixes
the canonical `run` subcommand. It records a
2,048 MiB memory expectation, which reserves no capacity and enforces no
per-job limit. The normal pool's resource policy and host backpressure apply;
normal and bulk have different resource policies, so compare candidate and
control runs through the same operation. The probe has no AgentCTL elapsed-time
deadline: AgentCTL owns cancellation and the canonical driver detects loss
of useful progress. Selection remains explicit; the operation adds no input
truncation or automatic corpus selection.

The driver isolates `HOME`, every XDG root and `POLYLOGUE_ARCHIVE_ROOT` inside the work
directory, writes structured events as JSON lines, samples the daemon's whole
process tree once a second, and polls the archive read-only until the build is
terminal or stops making progress (`--stall-timeout`); a build that keeps
converging runs until it settles. It then stops the daemon with SIGINT and
writes `receipt.json`. `--env POLYLOGUE_NAME=value` tunes the daemon but may
not override a variable the driver sets (archive root, config, logs, sampler).
Corpus, work and scratch directories must lie outside the checkout.

A build is **terminal** when the candidate is promoted, every cursor is
complete or excluded, every raw membership is settled, no convergence debt is
open, and every required readiness domain is ready. Raw rows that were never
parsed are reported but do not gate: a retained non-session artifact is never
parsed as a session, and the `raw_artifacts` readiness domain is the daemon's
own verdict on raw completeness. A receipt is **qualified** when the build is terminal, every check in
`checks` holds (including zero raw parse failures, equal FTS row counts, a clean daemon
shutdown, a lossless event log, a corpus re-verified after the run, and a
candidate whose commit, tracked edits and untracked files did not change
during the run), and every asserted budget passes; the command exits non-zero otherwise. A large single
source (the former 419 MB and 1.6 GB qualifications) is a `files` corpus run
with `--max-rss-mib`.

Every run samples the daemon's per-thread CPU in process (py-spy cannot attach
to the free-threaded interpreter) and reports it as `thread_cpu_s`: each writer
actor, the other threads, and the process total. Threads are sampled while
alive, so per-thread figures are lower bounds and the remainder is reported
as `unattributed`. On a GIL-enabled interpreter, `--profile` adds wall stack
samples and CPU ticks per stack; `profile stacks.json [--thread PREFIX] [--collapsed out]`
summarises them or writes flame-graph input. Free-threaded builds refuse
cross-thread frame capture with `profile_refusal=free_threaded_frame_snapshot_unavailable`;
per-thread CPU and process measurements remain available. A refused stack profile
does not attribute a busy thread to a particular preparation function.

## Receipt

Receipts use `polylogue.fresh-build-receipt.v2`. Refresh, render and comparison
reject other formats before interpreting their measurement fields. `report`,
including JSON output and refresh, returns exit 2 with
`incompatible_receipt_format`; `compare` refuses the same input before evaluating
its measurements. `--allow-unqualified` does not make another format comparable.
The original receipt stays unchanged.

| Section | Source | Contents |
| --- | --- | --- |
| `timing_s` | event log and observer | event preparation/chunks/promotion; observed readiness and derived observation delay |
| `stages` | ops `ingestion_batch` rows | inclusive stage timers summed over batches, grouped by category; parent and child timers may overlap |
| `writer` | `daemon.writer.released` events | busy share to promotion, queue depth, holds per actor |
| `event_log` | complete event file scan | valid events and malformed JSON, non-object or invalid event records; missing files remain unavailable |
| `shutdown_evidence` | sealed event and sampler inputs | exit status, explicit service/asyncio shutdown failures and input completeness; missing evidence stays unknown |
| `progress` | driver observations | accepted material, reduced required work and readiness/publication changes, separately from retry activity and declared next retry times |
| `by_source`, `projection` | `live.ingest.source_group` events, corpus population | seconds per MiB per origin and the intake projection for the sampled population |
| `thread_cpu_s` | in-daemon sampler | CPU seconds per writer actor and per other thread group |
| `process_tree` | driver samples | peak and p95 RSS, CPU seconds, mean cores, block I/O |
| `checks`, `budgets`, `qualified` | archive, samples | terminal checks and asserted budgets |
| `output_fingerprint` | promoted index | per-table digests over the differential harness's comparable relations, and a digest of every `messages_fts` posting by block |

`stage_seconds` preserves every observed inclusive timer. Category totals in
`stage_buckets_inclusive_seconds` may count overlapping parent and child time;
they are not a partition of elapsed time. `stage_timing_coverage` reports each
stage's observed and missing batches. Exclusive time and interval relationships
are unavailable from batch timer names; use the causal trace for interval evidence.

Clean daemon shutdown requires exit zero and complete, hash-bound captured
evidence without shutdown errors. Exit zero alone cannot qualify a historical
receipt. `refresh` rechecks retained evidence into a distinct view and leaves
the original receipt unchanged. Known failed services or asyncio shutdown errors
fail cleanup even when the process exits zero. Missing, malformed or undrained
inputs leave cleanup unknown. Terminal readiness timing retains its separate
qualification and observation uncertainty; daemon exit and event drain do not
prove descendant, native or external resource cleanup.

`compare BEFORE AFTER` prints the deltas and whether the per-table output
digests are identical, and exits non-zero unless the receipts are comparable:
same corpus, same run configuration, same benchmark implementation (a digest of the driver and report code, since `--candidate` may name another checkout), interpreter (version, GIL mode, build string and resolved executable) and host (architecture, CPU model, CPU count, memory, the effective CPU and memory limits the daemon sizes itself from, work filesystem), with bound inputs and terminal checks. A matched optional stack-profile refusal or failed asserted budget does not erase otherwise bound timing evidence. Different actual diagnostic modes still prevent a controlled metric comparison.

This command is an inspection operation. Exit zero establishes comparability;
outputs may still differ or have no fingerprint. Paired `experiment` qualification
also requires the declared output equivalence, but its timing summaries remain
descriptive and leave code improvement unestablished. Neither exit status is a
performance acceptance policy. A speed claim needs a separately declared effect,
confirmation and resource policy.
`--allow-unqualified` admits a run that promoted but did not settle (never one
that stalled before promotion), with a warning; it never admits a run whose candidate or corpus changed during the
build or whose event log lost events. An optimisation claims equivalence only on identical digests from
comparable receipts.

## Evaluation

`compare BEFORE AFTER --json` exposes independent `measurement_validity`, `scope_completion`, `equivalence`, `instrumentation_comparability`, per-metric `metrics`, diagnostic `capabilities` and `performance_decision` axes. New run receipts and experiment reductions also carry these derived evaluations. An absent proof stays `unknown`; the evaluator does not inherit the older aggregate `qualified` flag. Scope completion covers the sealed selected inputs and the declared ordinary daemon route. It makes no whole-install, serving-health, installed or published claim. A partial experiment also records that its planned paired sequence did not complete.

Each metric reports its numeric availability separately from qualification for a controlled comparison. Promotion uses the event milestone; terminal time is a readiness observation with recorded polling, scan and cache uncertainty. CPU is sampled over the daemon process tree and cut at promotion. RSS is a sampled process-tree peak, so child spikes between polls can be missed. Missing CPU evidence leaves promotion evidence available. Output difference or a missing fingerprint leaves timing observations available while refusing an equivalence claim. Exact comparison hashes relation contents and FTS postings; the receipt's separate FTS row-count check proves only equal counts.

The instrumentation axis requires recorded actual sampler coverage and a candidate-bound work-observation handshake whose enabled mode matches the declared mode. Historical absence is unknown, including when the requested mode was off. Different actual modes, including work observation on versus off, prevent qualified speed-metric comparison. Stack capability describes the actual in-daemon sampler coverage: a profile refusal prevents stack attribution, while matched refusals can retain bound timing. Per-thread CPU accounting supplies no native callchains. Refresh validates sealed sampler evidence and never invents historical import or instrumentation proof.

`compare BEFORE AFTER --accept` remains non-zero: a single pair has no launch-frozen confirmation sequence. `--metric` selects the inspected target (`promotion_s`, `terminal_observation_s`, `cpu_to_promotion_s` or `rss_sampled_peak_bytes`); it cannot create a launch policy. Unplanned pairs, A/A calibration and separated descriptive ranges establish no practical decision. Choosing a budget after inspecting results cannot create acceptance. Opt-in experiment acceptance is described below. A clean daemon exit and drained logs leave descendant and external resource cleanup unknown.

## Owner work observations

Add `--observe-work` to `run` or `experiment` to record actual neutral Raw
parse, retained validation, and session profile preparation/publication bodies.
It also retains the Raw frontier inspector's typed mode and counts at the
existing owner boundary. Observation is off by default. The launcher owns
`POLYLOGUE_TRACE_WORK`; the imported candidate reports its actual mode and
module hash. Compare only matching observed modes. This instrumentation has
not yet been qualified for overhead.

Reduce the run's structured event file with the existing trace command:

```bash
python -m devtools.fresh_build_bench.trace_ledger EVENTS_JSONL --out PRIVATE_DIR
```

The ledger's `work_observations` separates body entries, successful and failed
exits, complete input keys, produced artifact references, and the owner's
consumer decisions. Two executions of one key remain two bodies. A cache hit
is a consumer of the original artifact reference, not another body or a
measurement of avoided CPU. Profile publication includes commit; work that
prepared a result before the owner finds it already current remains visible.
These observations change neither reuse guards nor publication authority.

Input keys are conservative: Raw parsing uses the owned artifact and actual
parser recipe; validation additionally binds the accepted prefix, selection,
ordered schema snapshot and implementation; profile preparation binds the
full compute inputs and actual materializer recipe. Direct validation without
owner input evidence has an explicit unknown key. Different keys alone do not
establish necessary work. Instrumentation edits can change existing AST recipe
identities; the benchmark retains its normal recipe certification rather than
waiving that change.

Transport deduplication uses process generation, worker generation and producer
sequence, never a work key. Missing identities, internal sequence gaps,
unmatched bodies and known capture loss retain observed counts but withdraw
execution-amplification ratios. Ratios describe only paired observed bodies;
the stream does not certify the full-run work population or artifact cleanup.
Body CPU uses same-thread intervals united within process and worker lifetimes,
so nested bodies are not added together. It supplies no async CPU attribution
or exclusive per-operation partition.

## Components

`components blob --corpus DIR --scratch DIR [--workers N]` times blob
acquisition of every corpus file, sidecars included. A selection that matches
no file fails. Any worker error, or a corpus file that changed during the
timed work, exits non-zero. It iterates in seconds; the end-to-end run proves
the total.

`components validation --candidate CHECKOUT --input FILE --recipe JSON
--schema-root DIR --scratch DIR` runs complete retained JSONL validation in a
fresh process. `components experiment --before A --after B` with the same
input arguments uses the shared matched experiment runner in declared ABBA
order; `--repetitions N` repeats that order. Each invocation gets fresh scratch
and validation reuse, a fixed schema reader, and isolated configuration and
archive paths. It validates the complete production accepted prefix; no record
or session cap is applied.

The private recipe binds `input_sha256`, `accepted_prefix_size`, `raw_id`,
`evidence_id`, `source_path`, `expected_python`, `config_path`, `config_sha256`,
and `current_schema_snapshot_roots` (ordered `files` entries with `name`, `size`,
`sha256`). `provider` defaults to `codex` for validation. `route` must explicitly
equal `{"domain":"complete-loose-jsonl","validation_mode":"advisory","jsonl":true,
"captured_zip_coordinate":null,"schema_resolution":null,
"schema_resolution_is_explicit":false}`. Other modes, ZIP members and explicit
schema-resolution domains refuse. The declared interpreter and current
schema bytes must match exactly. Source coordinates affect validation, and
remain in private recipes and receipts. Candidates must be clean checkouts.
The launch freezes candidate, input, recipe, runtime and runner identities
before work; each child binds that launch before importing product code and
records actual imported product module hashes. Candidate and input changes
fail rather than producing a timing for different work.

Receipts include the shared process-scoped WorkloadReceipt, raw wall, thread
CPU and process CPU, host load, exact validation result and schema inventory.
ABBA reduction compares all semantic outputs exactly and reports raw order,
median, median absolute deviation and observed ranges. `report` verifies
original launch and constituent hashes and recomputes equivalence. Original
receipts and launch files are never overwritten. These are descriptive
component observations: ranges are not confidence intervals, fresh scratch
is not cold OS cache, and process lifetime RSS high-water is not an invocation
peak. External descendant cleanup, SQL VM steps and hard resource bounds stay
unavailable. Component `report --accept` refuses daemon acceptance.

`components preparation` accepts the same bound arguments, with explicit
`provider: "codex"` and `sidecar_context: "absent"`. It calls production
`prepare_jsonl_blob` with a captured resolver that cannot fall back to ambient
files. Other dependency morphologies refuse. Producer timing excludes independent
validation, complete artifact read consumption and backing retirement. The original
publication attempt keeps custody until readers settle, and its normal `discard()`
retires backing. Output comparison includes complete session content hashes and
canonical lowered shard message/block counts. Events are counted during complete
session reads. Revisions and jobs remain unknown. The same matched runner accepts
`components experiment --stage preparation`.

Prepared counts carry `count_basis: "production-preparation-before-lineage"`.
They describe service demand before archive lineage resolution, deduplication
and materialization; inherited prefixes can amplify these counts relative to
the final Index population.

`components page` uses the same binding arguments for a declared synthetic
acquired Codex page. The input is JSON with positive `count`, `messages`,
`text_size` and a nonempty `namespace`, consumed by the existing independent
Raw fixture owner. Its recipe declares `provider: "codex"`,
`route: {"domain":"synthetic-retained-page","validation_mode":"advisory"}`,
and positive `workers`, `queue_units`, `queue_bytes`. The candidate creates a
closed acquired archive, cloned through the shared immutable-tree owner before
timing. The actual `RawObservationConvergenceOwner.materialize_retained_raw_ids`
call includes selected capture, preparation, validation and retained publication,
with real compute admission and daemon writer owners. No destination callback
or convergence loop is reproduced. Complete output uses the existing canonical
fingerprint reducer, with an archive census and generator source hash.
Owned compute/writer shutdown must physically settle; external cleanup is unknown.
`components experiment --stage page` uses the same ABBA runner. These synthetic
independent transcripts do not qualify inherited, remote or hook morphologies.

`run --route restart|upgrade --seed-receipt RECEIPT` clones closed benchmark
state through the shared immutable-tree workload owner and uses ordinary daemon
startup without `--cold-build-index`. Seed adoption requires complete original
bound evidence, the same sealed corpus, exact current semantic output and clean
product contents. Restart requires identical product contents; upgrade requires
different contents. That distinction alone does not prove a schema rebuild.
Physical contents are frozen before launch and checked on each matched invocation.
The manifest records entry and byte counts, and refuses symlinks. Historical
physical bytes are unclaimed.
A restart may have no new promotion event, so promotion latency stays unavailable.
Practical acceptance currently refuses seeded routes.

`capabilities` probes builtin cProfile and tracemalloc on the actual interpreter
and GIL state. It reports callback/allocation roundtrip availability and provider
limits. Capability availability does not qualify a profiled performance comparison.
Python traced allocations do not account for complete RSS or native allocator
residence.

## Reading the numbers

Missing or null batch metrics stay unknown: their totals are `null`, with
observed and missing batch counts in `metric_coverage`. A measured zero stays
zero. A lossless event verdict requires a present, fully parsed file and known
zero terminal dropped, failure and undrained counters. Scheduled recovery
waiting remains visible without advancing the useful-progress timestamp.
Embeddings are disabled in this isolated benchmark; it does not qualify paid
embedding-backup reuse or complete configured-operation delivery.

The RSS budget uses the larger of the 4 Hz process-tree peak and the daemon
process's own high-water mark (`VmHWM`); a worker child's spike shorter than
the sampling interval is not observed. Corpora are created owner-only.
All times in a receipt are seconds from the driver's launch of the daemon;
event milestones, observations and process samples share that clock. Process
CPU and I/O totals keep the counters of worker processes that exited.
Receipts record the host's load average at start. Wall time on a shared host
moves with load; CPU seconds, writer holds and per-stage timers move less.
The intake projection scales per-origin rates to the population and says so;
every export shares the inbox watch source, so a corpus with exports of more
than one origin leaves those origins unmeasured in the projection. Promotion and derived convergence are reported as measured, not scaled.

## Small paired loop

Select a small sealed corpus once, then reuse it while changing the product candidate:

```bash
devtools bench fresh-build corpus select --out /realm/tmp/work/fb/small \
    --seed 1 --max-bytes 8388608 --max-unit-bytes 1048576 --max-units 12
agentctl job start polylogue fresh_build_bench --workspace /realm/worktree/driver -- \
    experiment --corpus /realm/tmp/work/fb/small --work /realm/tmp/work/fb/pair \
    --before /realm/worktree/before --after /realm/worktree/after --order ABBA --repetitions 1
```

`select` copies real whole session units, including subagents and tool sidecars, plus existing static assembly dependencies at the production retained-assembly coordinates. All three private builders (`select`, `sample`, and `files`) add the same complete static assembly closure. Unit members keep their lexical native-layout positions, including admitted directory aliases. Each dependency is placed by applying the production coordinate resolver to the copied member as well as its original acquired coordinate, so an install-root link cannot separate copied mains from their metadata. Optional absence is allowed; links, unreadable dependencies, membership changes and changed bytes refuse the selected closure. Shared history and index files count once against `--max-bytes`; `--max-unit-bytes` limits each unit's own files. A dependency closure that exceeds the total budget is excluded whole. The manifest records private source provenance, each unit's private and shared bytes, sampled and missing providers, and covered and missing provider/size strata. Selection normalizes source and unit enumeration before consuming its seed, shuffles provider order and each provider's eligible size-bucket order, then takes one complete member from one bucket per provider turn before cycling the remaining buckets. Member order is shuffled within each bucket. This gives tiny feasible selections a provider visit before more buckets from the same provider, subject to the exact private and cumulative closure budgets. The sealed parameters record the strategy, initial provider/bucket orders, and cumulative total-budget skip counts in aggregate and per stratum; missing providers and strata remain visible. This is diagnostic coverage, not a probability sample or population estimator. Its byte budgets select an experiment workload; they do not limit valid product work. Gemini units are complete projects. The separate mutable Codex database source, exports, hooks, embeddings and browser capture are excluded from this initial selected-root route. Synthetic corpora must use `kind="synthetic"`; their timings are repeatability controls. Budgeted selections do not project whole-archive time.

`experiment` runs the existing ordinary-daemon terminal route with a new archive, XDG tree and bytecode directory for every sample. Both arms share the driver, sealed corpus, interpreter and configuration. The host filesystem cache is uncontrolled. Balanced `--order AB` is a single sample per arm; default `ABBA` supplies two per arm. Repetitions repeat that sequence. `experiment.json` references immutable constituent receipts by path and SHA-256, exposes raw timings, medians, median absolute deviations and ranges, and requires bound terminal runs with exact or narrowly recipe-qualified output equivalence. Overlapping observed ranges are labelled `within-observed-noise`; separated ranges are `observed-arm-difference`. Neither observed range separation nor two samples proves speed improvement. All metrics are descriptive, with sample insufficiency stated. Identical recorded candidate identity and actual imported module hashes label the experiment `A/A-calibration`, with no code-improvement claim. A run whose input validity or declared scope completion fails or remains unknown stops the sequence and preserves its receipt. Optional stack refusal alone does not stop it.

Each new run binds a read-only diagnostic policy observer by hash in its launcher. The observer reads the existing published compute owner's capacities, per-class reserves and ceilings, and actual GIL mode after product and profiling imports. It records observed policy changes in `policy.json`, sealed by the run receipt. Sampling is periodic at 50 ms and at normal process exit, so transient changes between samples can be missed. The observer never constructs or changes an adapter. Missing binding prevents qualification and comparison. Different effective policies or launcher settings cannot silently compare as one configuration. Python launcher wrappers are refused because their injected behavior cannot be bound by the interpreter identity; pass the real interpreter executable.

`report RECEIPT --refresh` (also accepting an `experiment.json`) writes a new content-addressed `receipt.view-<digest>.json` beside the original. The view binds the source receipt digest and reducer version, rechecks the evidence seal, and recomputes derived fields. It leaves the original receipt unchanged, so prior experiment references remain valid. Current-format receipts without effective policy or actual candidate readiness proof remain readable, but do not qualify a controlled comparison. Refresh validates recorded proof and never reconstructs historical imports.

The selected population is a timed, non-atomic census. Unselected live sessions may grow while selection copies a stable complete chosen closure; new dependencies or edits in that chosen closure are refused. The launcher inserts the protected candidate path before any sampler or product import, and the observer records actual imported product module paths and content hashes. Candidate import mismatch prevents qualification. The driver hash must also remain unchanged during the measured run.

This first paired loop holds compute policy fixed. Policy sweeps need an explicit treatment contract before their receipts can compare. The strict fingerprint includes parser/lowering identities. A second, explicitly recipe-qualified digest excludes only `sessions.parser_fingerprint` and `sessions.lowering_fingerprint`, after every session row matches its own daemon's actually imported `parser_fingerprint_for_origin` and `lowering_fingerprint` results. The function paths and source hashes bind them to that candidate. All other columns, recipe fields, rows and FTS postings remain exact. The secondary digest rehashes only sessions and reuses every other strict relation and FTS digest. Missing or mismatched proof refuses recipe equivalence. Historical runs lack that observed function proof and cannot acquire it retrospectively.

Promotion is a product event milestone. Terminal timing is the scan finish of the fresh ready observation that certifies the stable-poll decision. The terminal measurement records this as `terminal_observed_s`. Earlier cached or tentative observations remain recorded but do not set this timing. Refresh preserves historical `first_observed_s` recordings and their original cache/certification semantics. Receipts record each poll's start and scan finish, readiness-check window and cache reuse (the readiness cache lasts up to 60 seconds), plus the observation window ending at terminal certification. Scans combine multiple non-atomic reads; cached readiness widens their uncertainty. Ordinary polls reuse readiness, but the final stable-poll decision refreshes any cached all-ready census through the same candidate-bound helper before stopping the daemon. A fresh blocked or refused census prevents terminal certification and resets the tentative stable-poll sequence. These times do not identify the exact ready transition. Terminal throughput is left unknown; promotion throughput retains its event basis. Historical views retain the original poll-start time and explicitly mark missing scan/cache/certification times unknown.

Readiness is checked by a read-only subprocess using the measured arm's interpreter, isolated environment and explicit candidate import path. Its imported function path and source hash, interpreter, GIL state after imports, check window and helper CPU/wall cost are recorded in sealed `readiness.json`; the receipt binds the helper's hash. The helper starts no daemon, compute owner or profiler. Its cost is outside daemon resource samples and promotion timing, and it may interfere with product work. The readiness cache includes candidate path, source hash, interpreter, environment and helper hash. It also binds the promoted Index target. A cached negative census is reused for up to 60 seconds only while advisory metadata for the resolved Source and active Index database, WAL and journal files remains unchanged. A relevant write, file replacement, promotion or unavailable stamp triggers a new candidate-bound census; a change during that census prevents negative reuse on the next poll. These stamps schedule reads and never prove readiness. Unchanged archives keep the existing reuse interval; unrelated Ops writes and SHM reader marks do not invalidate it. Mandatory terminal certification and the post-shutdown census still read fresh readiness. Checks start after promotion with zero debt and can expose ready domains while intake cursors remain incomplete. Invalid imports, process failures or an invalid census produce an observation refusal, never readiness. Known transient SQLite read failures retain retryable observation errors and are never cached as ready; binding requires an eventual successful census. Terminal success still requires complete cursors and every required domain.

The receipt's `fts_row_count_equal` check compares indexable row counts. Exact output comparison separately hashes all FTS postings and relation contents. Derived refresh views rename the older count-only `fts_exact` field; historical receipt bytes remain unchanged.

## Opt-in practical repeatability

`experiment --acceptance-policy POLICY.json --accept` freezes an explicit policy, candidate identities, sealed corpus, interpreter/host configuration and schedule in an exclusively created `launch.json` before the first daemon launch. The policy owns the fixed schedule `ABBAABBA`, with fresh state for every run. Each actual daemon binds that launch before its main entry point, each receipt seals its launch copy, and refresh preserves the recordings. Policy-free experiments retain their descriptive behavior.

This method makes an operational repeatability decision on the sealed selected inputs. It supplies **no statistical confidence or attribution error bound**. Host filesystem caches remain uncontrolled. The four predetermined adjacent pairs are ordinals 1/2, 3/4, 5/6 and 7/8. Every pair must meet both the absolute and relative promotion-time improvement thresholds, all sampled CPU/RSS guardrails, equivalent outputs, actual instrumentation and candidate binding, ordinary-route completion, and bound daemon shutdown. A winning median cannot rescue a losing pair.

An example policy uses neutral illustrative thresholds. Choose the actual values before measuring confirmation:

```json
{
  "format": "polylogue.fresh-build-practical-policy.v1",
  "target_metric": "promotion_s",
  "effect": {"minimum_absolute": 0.5, "minimum_relative": 0.1},
  "resources": {
    "scope": "sampled-daemon-tree",
    "cpu_to_promotion_s": {"absolute_slack": 0.2, "relative_slack": 0.05, "ceiling": 30},
    "rss_sampled_peak_bytes": {"absolute_slack": 1048576, "relative_slack": 0.05, "ceiling": 536870912}
  },
  "cleanup": "bound-daemon-shutdown; descendant-external-unclaimed",
  "confirmation": {
    "method": "two-abba-all-pairs-v1",
    "purpose": "held-out-confirmation",
    "held_out_basis": {
      "attested_untuned": true,
      "statement": "Reserved selection before tuning this candidate or these thresholds",
      "discovery": null
    }
  }
}
```

Effect thresholds must be finite and positive. For each pair the improvement must be at least `max(minimum_absolute, minimum_relative * before)`. Resource slack must be finite and nonnegative; each ceiling is mandatory, finite and positive. Each candidate resource value must be at most both its ceiling and `before + max(absolute_slack, relative_slack * before)`. Missing qualified resource evidence leaves the decision unavailable. CPU-to-promotion before the first process sample is unavailable, never a measured zero. RSS and CPU describe sampled daemon-tree observations; they cannot certify hard physical bounds or short-lived child spikes.

The held-out basis is an explicit operator assumption about absence of tuning. When discovery evidence exists, set `discovery` to its private experiment `path` and `sha256`. Launch checks that reference and its immutable constituents, refuses an identical confirmation corpus, and checks session-unit nonoverlap when both selections recorded unit identities. Shared assembly dependencies may overlap. Missing historical unit evidence stays unavailable in the nonoverlap field; the attestation does not become verified proof. Private provenance stays in private artifacts, not published examples.

Actual candidate import binding remains required. The observer also records content hashes of product modules imported in the measured daemon at shutdown. Different Git commits with identical observed imported content cannot create acceptance. An incomplete or unchanged observed treatment inventory leaves treatment difference unproven; it does not classify all unobserved product code as identical. An imported module difference itself does not attribute the measured effect to a particular executed body.

`report EXPERIMENT.json --accept --json` rechecks immutable constituent hashes and evidence through the existing refresh reducer, recomputes the frozen decision, and exits zero only for acceptance. It writes a distinct view and preserves originals. It cannot attach a policy retrospectively or trust a stored accepted boolean. Unsupported policies are refused. Initial acceptance supports only promotion event time. Exact readiness transitions, terminal observation speed, hard resource ceilings, global descendant/external cleanup, installation, serving health and whole-corpus performance remain outside this policy. Terminal polling/scan/cache uncertainty remains visible in inspection.


## Route coverage and remaining qualification

The existing run and component owners provide the routes below. A route declaration does not prove a candidate completed it; original receipts carry that evidence.

| Route | Existing owner and declared morphology | Qualification limits |
| --- | --- | --- |
| Normal installation | `run`, complete sealed native source units, fresh archive | Ordinary daemon checks and original receipts required |
| Capture | `components blob`, complete sealed payloads | Acquisition copy/hash/publication only |
| Retained validation | `components validation`, complete advisory loose JSONL | Small actual invocation qualified; explicit ZIP or schema resolution refuses |
| Retained preparation | `components preparation`, Codex loose JSONL with declared absent context | Small actual invocation qualified; native-context calibration remains unavailable |
| Retained publication | `components page`, declared independent synthetic Codex inputs | Actual selected owner; current smoke refused before measurement and operational qualification remains open |
| Append | No dedicated matched adapter | Unsupported benchmark morphology |
| Rewrite | No dedicated matched adapter | Unsupported benchmark morphology |
| Retained replay | No dedicated matched adapter | Materialization is not replay evidence |
| Restart | `run --route restart --seed-receipt`, identical product contents | Seed closure and output bound; actual lifecycle qualification remains open |
| Upgrade | `run --route upgrade --seed-receipt`, different product contents | Code difference alone does not prove schema rebuild; actual lifecycle qualification remains open |
| Remote completion | No dedicated matched adapter | Remote input/completion authority remains unsupported |
| Exceptional hooks | Explicit corpus hook spool, separate workload population | Separate backlog evidence; no normal-population extrapolation |

`denominator inventory` and `denominator project` forward to the [bounded denominator owner](fresh-build-denominator.md). Partial native metadata inventory is not a population estimate. Calibration requires complete-unit closure, production preparation receipts and their explicit count basis. Observed native sidecar context outside the supported preparation domain remains unavailable.

`ledger INPUT --out NEWDIR` forwards to the [offline work-ledger owner](fresh-build-work-ledger.md). Recorded spans and duration-only writer facts retain their different coverage; absent body observations cannot produce body CPU attribution.

Direct `components validation --diagnostic cprofile|tracemalloc` uses the [diagnostic capture owner](fresh-build-diagnostics.md) around the same production validation body. It binds candidate, runtime, schema inputs and loaded source authority, and places the immutable manifest reference outside semantic output. Preparation, page and matched diagnostic modes refuse. cProfile reports runtime call elapsed without TID attribution; concurrent nesting and elapsed remain unqualified. Independent caller-thread CPU boundaries are separate. Tracemalloc covers Python traced allocations, with point RSS observations rather than interval peaks. These captures establish no unprofiled speed comparison.


Named Claude units use one project-parent listing and only the selected session subtree; Gemini lists its project root and keeps every project with the existing unit key (including same-stem names), and named Codex walks only its containing directory for the exact rollout. Every selected entry still passes the original native layout, source ownership and link/fault rules. Unrelated sibling names are inspected, but their types, links and subtrees are not visited. Named input is diagnostic selection, not probability sampling.

`sample` keeps its original seeded layout-unit draws and watched-population denominator. Its manifest separates drawn unit bytes from full shared assembly metadata added once per corpus coordinate. Already drawn metadata is not counted twice. When extra shared metadata has no sampling denominator, whole-population intake projection stays unavailable for that origin; actual intake timings and resources remain reported. Previously sealed corpora and receipts retain their original meaning.

## Interactive measurement scope

The interaction harness correlates read-frame and compute intervals by request
ID, including overlapping requests that finish out of order. Mixed-load tags
are preseeded for the two writer sessions, then read independently from the
archive to confirm the same exact populations survived the window. RSS is a
window high-water mark only when the Linux counter reset succeeds; otherwise
it is unavailable. A workload without byte reservations reports queue bytes
as unmeasured rather than publishing zero as a capacity observation.

Cold command RSS comes from that child's wait4 usage record, including its
pre-exec overhead, rather than subtracting cumulative children high-water
marks. Timing samples do not enable import profiling. Retained discovery
receipts carry the original observing host, and milestone trace writes happen
after the measured route finishes.

The perf-floor runner separates quick and representative floor files. It
refuses cross-mode comparison and requires the recorded workload version to
match. Workload version 3 measures canonical retained replay and current query
surfaces; older artifacts require new measurements through that route.
