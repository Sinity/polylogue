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
| files | `corpus files --out DIR [--export ORIGIN=PATH] [--hooks DIR --hooks-fraction F] FILE...` | Exactly the named real transcripts, e.g. one whale; each must be a file its source root's watcher admits. `--export` stages a ChatGPT or Claude.ai export under `exports/`. `--hooks` stages a deterministic fraction of a legacy hook spool for backlog timing. Private. |

Both are private: corpora, manifests and receipts stay outside the checkout
(the command refuses a path inside it), and only aggregate numbers leave the
machine.

## Running

```bash
devtools bench fresh-build run --corpus DIR --work EMPTY_DIR [--profile] \
    [--max-rss-mib N] [--max-promotion-s N] [--max-terminal-s N]
```

Start runs through the declared AgentCTL operation
(`agentctl job start polylogue fresh_build_bench -- run ...`). The driver
isolates `HOME`, every XDG root and `POLYLOGUE_ARCHIVE_ROOT` inside the work
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
`checks` holds (including zero raw parse failures, exact FTS, a clean daemon
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

| Section | Source | Contents |
| --- | --- | --- |
| `timing_s` | event log | preparation, first and last intake chunk, promotion, terminal, derived phase |
| `stages` | ops `ingestion_batch` rows | the daemon's own stage timers summed over batches, and a declared rollup (acquire, parse, materialize, index, fts, derived) |
| `writer` | `daemon.writer.released` events | busy share to promotion, queue depth, holds per actor |
| `event_log` | complete event file scan | valid events and malformed JSON, non-object or invalid event records; missing files remain unavailable |
| `progress` | driver observations | accepted material, reduced required work and readiness/publication changes, separately from retry activity and declared next retry times |
| `by_source`, `projection` | `live.ingest.source_group` events, corpus population | seconds per MiB per origin and the intake projection for the sampled population |
| `thread_cpu_s` | in-daemon sampler | CPU seconds per writer actor and per other thread group |
| `process_tree` | driver samples | peak and p95 RSS, CPU seconds, mean cores, block I/O |
| `checks`, `budgets`, `qualified` | archive, samples | terminal checks and asserted budgets |
| `output_fingerprint` | promoted index | per-table digests over the differential harness's comparable relations, and a digest of every `messages_fts` posting by block |

`compare BEFORE AFTER` prints the deltas and whether the per-table output
digests are identical, and exits non-zero unless the receipts are comparable:
same corpus, same run configuration, same benchmark implementation (a digest of the driver and report code, since `--candidate` may name another checkout), interpreter (version, GIL mode, build string and resolved executable) and host (architecture, CPU model, CPU count, memory, the effective CPU and memory limits the daemon sizes itself from, work filesystem), and both qualified.
`--allow-unqualified` admits a run that promoted but did not settle (never one
that stalled before promotion), with a warning; it never admits a run whose candidate or corpus changed during the
build or whose event log lost events. An optimisation claims equivalence only on identical digests from
comparable receipts.

## Components

`components blob --corpus DIR --scratch DIR [--workers N]` times blob
acquisition of every corpus file, sidecars included. A selection that matches
no file fails. Any worker error, or a corpus file that changed during the
timed work, exits non-zero. It iterates in seconds; the end-to-end run proves
the total.

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
