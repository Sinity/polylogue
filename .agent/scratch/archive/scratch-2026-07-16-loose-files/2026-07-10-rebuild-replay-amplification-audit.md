---
created: 2026-07-10
purpose: Audit cold index-rebuild replay amplification and define a correctness-preserving first optimization
status: complete
project: polylogue
---

# Rebuild Replay Amplification Audit

## Scope and evidence

This is a source-only audit. I did not query, mutate, rebuild, or test against the
live archive. The measured production facts supplied to the audit are:

- `source.db` contains 172 repeated `(origin, native_id)` groups, 577 rows,
  and 11.34 GiB of raw blobs.
- 10.27 GiB is in rows preceding the largest row in each repeated group.
- Two Codex sessions account for roughly 10 GiB across 39 rows.
- Cold rebuild repeatedly parses a roughly 422 MiB / 133,706-message row for
  about 12 seconds, after which the write path reports `changed=0`.

Those measurements are consistent with the current implementation: cold rebuild
selects every source row, does the expensive parse first, and only then discovers
that an older or equivalent session does not change the index.

## Current path

### Selection and ordering

`polylogue ops maintenance rebuild-index` calls
`_all_index_rebuild_raw_ids()` unless explicitly scoped. That query selects every
`raw_sessions.raw_id` and orders by `(acquired_at_ms ASC, raw_id ASC)`
(`polylogue/cli/commands/maintenance.py:1067-1079`). No source row is classified
as a historical snapshot, append delta, bundle member, or current representative.

`parse_from_raw()` then batches those ids by record count and total blob bytes,
but it still submits every selected record to `process_ingest_batch()`
(`polylogue/pipeline/services/parsing_workflow.py:296-421`). Batching bounds
memory; it does not reduce parse work.

The rebuild plan already reports selected blob weight, but its semantics are
still “selected rows = rows to parse” (`maintenance.py:1122+`). The measured
amplification therefore cannot be avoided by changing batch size or worker count.

### Decode and lowering

Each raw row is loaded from the content-addressed blob store and decoded by
`ingest_record()` (`polylogue/pipeline/services/ingest_worker.py:580-646`). Parser
semantics depend on more than blob bytes:

- stored origin/provider;
- exact source path and its suffix;
- path classes such as Claude subagent paths and Drive cache paths;
- fallback id derived from the source path;
- `source_index == -1`, which marks append-only writes;
- file mtime, used as a timestamp fallback when parsed evidence has no timestamp;
- validation mode and parser/schema version.

For known Codex and Claude Code JSONL, the streaming path avoids materializing
the entire decoded record list, but the provider parser still constructs the
full normalized session. Claude Code streaming can also lower one file into
multiple contiguous session groups and merges repeated groups
(`polylogue/sources/dispatch.py:441-501`, `385-438`). A later full-file snapshot
can therefore cover multiple logical sessions, not merely the `native_id` stored
on its raw row.

Bundles and split artifacts are different. ZIP acquisition can serialize each
session payload separately with non-negative `source_index` values
(`polylogue/sources/source_acquisition_components.py:360-416`), while grouped
providers preserve the whole entry. These rows cannot be treated as a growing
single-session snapshot chain merely because an origin or native id repeats.

Append ingestion is different again: it writes only the newly observed JSONL
tail with `source_index=-1` (`polylogue/sources/live/append_ingest.py:60-99`).
Every append fragment may carry unique messages. Skipping all but the newest
append fragment loses data during a cold rebuild.

### Write-time precedence happens too late to save parse cost

The current write path provides several correct guards, but all run after the
raw row has been decoded, validated, parsed, normalized, hashed, and transferred
back from a worker:

- stale full replacements are rejected by parsed `updated_at`/`created_at`;
- a DOM fallback cannot replace a non-fallback session;
- a strictly smaller message set cannot replace a larger stored session;
- equal content refreshes `sessions.raw_id` to the latest accepted raw row;
- append-only rows are reduced to messages with previously unseen native ids.

See `polylogue/pipeline/services/ingest_batch/_core.py:394-510`.

This is the implementation of the closed `polylogue-0mu` freshness/precedence
work. It cannot simply be copied into the source-row selector: source rows do not
store parsed message counts, session timestamps, ingest flags, or output content
hashes. “Newest acquired row wins” would be a different and weaker rule.

There is also no universal raw native-id key. The ordinary acquisition path's
`RawSessionRecord` has no `native_id`, and `save_raw_session()` writes `NULL`
(`polylogue/storage/runtime/raw/records.py:10-28` and
`polylogue/storage/sqlite/queries/raw_writes.py:37-62`). The direct
`ArchiveStore` path does populate native id from the parsed session
(`polylogue/storage/sqlite/archive_tiers/archive.py:982-1001`). Consequently the
measured repeated `(origin,native_id)` groups are useful evidence, but not a
complete or architecture-safe rebuild selection key.

### Raw identity has two regimes

The ordinary acquisition path uses the blob SHA-256 as `raw_id`
(`polylogue/pipeline/services/acquisition_records.py:32-69`). Exact repeated bytes
there collapse to one row; a later observation updates path/mtime on the existing
row (`raw_writes.py:66-75`).

The direct archive writer instead derives `raw_id` from origin, source path,
source index, blob hash, and native id
(`polylogue/storage/sqlite/archive_tiers/source_write.py:130-151`). Exact blob
duplicates can therefore remain distinct raw rows on that path. Any dedup plan
must use `blob_hash`, not assume `raw_id == blob_hash`, and must preserve the
parser context and final raw-link semantics.

### Existing raw retention is not a safe rebuild selector

`polylogue/storage/raw_retention.py` ranks rows only by
`(source_path, source_index, acquired_at_ms)` and considers older `-1`/`0` rows
cleanup candidates when the source path still exists. It does not prove a byte
prefix, semantic containment, or parser-output equivalence. The daemon further
sets `keep_full_snapshots=1_000_000`, so automatic compaction only targets append
snapshots (`sources/live/batch.py:1546-1558`).

That maintenance path is destructive and relies on the external source file as
recovery evidence. It must not be reused as the cold-rebuild optimizer: the
desired optimization retains every durable raw observation and merely avoids
redundant parse execution.

## What is safe at each tier

### Tier 0: exact raw-id duplicate

There cannot be two rows with the same `raw_id` because it is the primary key.
This tier is already collapsed.

### Tier 1: exact blob duplicate

Potentially safe, but only within the same parser-context key. At minimum that
key is:

```
(origin, exact_source_path, source_index, validation_mode/parser_fingerprint)
```

For identical bytes and context, parse output is identical. The representative
must be the row that exhaustive replay would leave as `sessions.raw_id` (normally
the latest accepted observation), or the optimizer must explicitly replay the
raw-link update. Same blob under a different path is not safe: path determines
fallback identity, subagent classification, artifact classification, and stream
handling.

Expected payoff is small because the ordinary acquisition regime already uses
blob hash as the primary key.

### Tier 2: cryptographically proven growing full snapshots

This is the small, safe, high-payoff first slice.

An older raw row `A` may be covered by later row `B` only when all of the
following are true:

1. Both rows have the same origin and exact source path.
2. Both have `source_index = 0`; append rows and split/bundle members are excluded.
3. `B.acquired_at_ms > A.acquired_at_ms`; equal timestamps are not causal order.
4. Both blobs exist and `len(B) > len(A)`.
5. SHA-256 of the first `len(A)` bytes of `B` equals `A.blob_hash`; this proves
   exact byte-prefix containment without opening `A`.
6. The provider/path class is registered as a prefix-monotone full stream.
   Initially allow only ordinary Codex and Claude Code session JSONL; exclude
   subagent/path classes until their contract fixture is explicit.
7. `A` was previously parsed successfully, or the planner retains it so rebuild
   does not silently erase an unresolved validation/parse observation.

Codex and Claude Code compute session freshness as a maximum over observed
timestamps (`sources/parsers/codex.py:861-895` and
`sources/parsers/claude/code_parser.py:321-325`). With an exact later byte-prefix
extension, the later parse contains every prior record and cannot lower that
freshness. It also includes every prior contiguous Claude session group. Parsing
the later snapshot therefore produces the same final index state as exhaustive
`A` then `B`, while retaining `A` in source.db as durable evidence.

The prefix proof can be cheap: sort candidate sizes ascending and stream a
maximal blob once, taking copies of the incremental SHA-256 state at candidate
length boundaries. One 422 MiB maximal read can prove dozens of ancestors; it
does not need to reread the 10 GiB of historical candidates.

Rows that are not covered by the first maximal chain become roots of additional
divergent chains and are all parsed. This makes optimization proportional to the
sum of divergent maxima, not the sum of every historical snapshot.

### Tier 3: “latest full snapshot wins” without byte containment

Not safe with the current schema. Counterexamples include:

- source truncation or rewrite where a later snapshot drops old messages;
- two divergent branches written to the same path;
- DOM fallback and native capture of the same logical session;
- same message count but different session events/attachments/policy evidence;
- a parser correction that changes content without changing count;
- a full snapshot and append delta sharing a logical session;
- a bundle whose members lower to multiple sessions;
- path-sensitive parsing of equal bytes;
- acquisition timestamp ties, currently broken deterministically by `raw_id`,
  not by a causal sequence.

General semantic supersession would need trustworthy parse receipts: parser
fingerprint, parse mode, produced session ids, content hashes, timestamps,
message/event/attachment counts, ingest flags, and raw-link effects. Those facts
do not currently exist in source.db.

### Tier 4: append/delta coalescing

Do not optimize initially. `source_index=-1` is a delta contract. Even if one
fragment's bytes prefix another, the write semantics are merge-append by native
message id, not full replacement. A future delta optimizer would need an explicit
contiguous cursor/range proof and per-message identity receipt, not recency.

## Proposed invariant

> A cold rebuild may omit a durable raw row from parse execution only when a
> selected row in the same parser context is a strictly later, cryptographically
> proven byte-prefix extension and that provider/path class has an executable
> prefix-monotonicity contract. Every maximal, divergent, append, bundle,
> failed/unparsed, and ambiguous row remains exhaustive. Omitted rows remain in
> source.db and are reported as `covered_by`; optimized and exhaustive rebuilds
> must have identical logical index projections.

This is deliberately a proof-driven replay planner, not retention, heuristic
deduplication, or another storage abstraction.

## Implementation shape for the first slice

1. Add a pure read-only replay-plan function near the existing rebuild selection
   helpers. Input is source rows plus blob-store access; output has executable raw
   ids and explicit coverage edges (`covered_raw_id -> representative_raw_id`,
   proof kind, bytes avoided).
2. Keep `_all_index_rebuild_raw_ids()` as the exhaustive reference path. Make
   optimization opt-in until the differential lane is green and a real dry plan
   shows the expected chain.
3. Register prefix-monotone stream classes explicitly; do not infer from file
   suffix alone. Codex and ordinary Claude Code session JSONL are the initial
   candidates.
4. Apply the plan before `_iter_raw_id_batches()`. Do not change parser or write
   precedence.
5. Emit selection telemetry: source rows, executed rows, covered rows, selected
   blob bytes, parse bytes, prefix-proof scan bytes, bytes avoided, divergent
   chain count, and bounded top coverage groups.
6. Preserve exhaustive fallback on any missing blob, hash mismatch, unsupported
   path class, parse-state ambiguity, or planner exception.

No schema bump is needed for this slice. Prefix coverage is an execution plan,
not durable source meaning. If caching later proves worthwhile, use disposable
`ops.db` keyed by `(blob_hash, parser_fingerprint, context_key)`; never make the
cache correctness-authoritative.

A durable source-tier parse-receipt table would require an additive source v4
migration and backup gate. That is justified only for broader semantic
supersession or portable parser-fidelity evidence, and should be reconciled with
`polylogue-rii.3` rather than smuggled into a performance patch.

## Repro and benchmark harness

### Focused correctness fixture

Build a synthetic archive through real acquisition/blob/source/rebuild paths,
not a toy archive implementation. For both Codex and Claude Code:

1. Write 4 growing full JSONL snapshots of one source path where each blob is an
   exact prefix of the next.
2. Run exhaustive cold rebuild into index A.
3. Run optimized cold rebuild into index B.
4. Structurally diff every index table, excluding only the volatile allowlist
   owned by `polylogue-hjwr` (materialization times/run ids/generation counters).
5. Assert the optimized plan parsed one raw row, covered three, retained all four
   source rows, and linked final sessions to the same raw representative.

Add a Claude fixture whose later snapshot introduces a second contiguous
`sessionId` group, proving that the representative materializes both sessions.

### Counterexample matrix

The planner must retain all relevant rows for:

- one-byte divergence in the middle of an older snapshot;
- later truncation (larger earlier blob, smaller later blob);
- equal acquisition timestamps;
- every `source_index=-1` append fragment;
- `source_index > 0` split/bundle members;
- same blob under different source paths;
- same path under different origins;
- subagent and auto-compaction path classes until separately registered;
- a prior parse failure/unparsed row;
- missing blob evidence.

### Scale benchmark

Parameterize snapshot count `1/4/16/40` and maximal size
`1/16/256 MiB` in the optional scale lane. Report:

- exhaustive selected rows/bytes and wall/CPU time;
- optimized prefix-scan bytes/time;
- optimized parsed rows/bytes and wall/CPU time;
- process peak RSS;
- logical diff result.

The performance assertion is structural, not a fragile wall-clock threshold:
for a single N-snapshot prefix chain, executed parse rows must be 1 and parsed
bytes must equal the maximal blob size. A looser measured wall-time target can
be reported but should not gate unit CI.

The live follow-up is a read-only `--plan` against the measured 577-row set,
recording how many of the 10.27 GiB are actually prefix-proven. Do not assume all
repeated native-id rows qualify.

## Anti-vacuity faults

The harness must demonstrate failure when each fault is injected:

1. Disable coverage planning: the parse-count/bytes assertion fails even though
   logical equivalence remains green.
2. Remove the strict-later acquisition guard: the truncation/tie fixture changes
   the final raw link or logical projection.
3. Allow `source_index=-1`: the append fixture loses messages.
4. Drop source path or origin from the context key: the path/provider-sensitive
   fixture changes identity or classification.
5. Treat a same-size or one-byte-divergent blob as a prefix: the hash proof and
   differential fail.
6. Remove one table from the logical diff census: the `hjwr` auto-census contract
   fails because every derived table must be diffed or allowlisted.
7. Delete the representative blob after planning: execution falls back or fails
   explicitly; it must not report a successful covered chain.

These faults prove both halves: the optimization really reduces work, and its
guardrails really protect correctness.

## Bead mapping

This is the unresolved half of closed `polylogue-3wb` (“Optimize rebuild graph
resolution and huge-row replay”). That bead explicitly named repeated ~298 MiB
Codex raw rows and required determining whether they were variants or preventable
replay churn. Its close evidence showed zero *missing-materialization backlog*
after the index was already populated; it did not exercise a from-scratch replay
where every historical source row is selected. The new v29 rebuild evidence
falsifies the implication that the huge-row replay problem was complete.

Preferred surgery: reopen `polylogue-3wb` rather than create a duplicate, append
this evidence, and narrow its residual scope to cold-rebuild replay selection.
Relate it to `polylogue-hjwr` (differential safety proof), `polylogue-1xc.8`
(rebuild-safety scenario), and `polylogue-20d.15` (bulk throughput). It should not
block those broader programs, but its optimized path must consume the `hjwr`
logical-diff helper once available.

Proposed reopened description addition:

> Cold v29 rebuild on 2026-07-10 re-exposed the unresolved huge-row half of this
> bead: source.db has 172 repeated `(origin,native_id)` groups / 577 rows / 11.34
> GiB; 10.27 GiB precedes the largest snapshots, and two Codex sessions account
> for ~10 GiB across 39 rows. Rebuild selects all raw rows and repeatedly parses a
> ~422 MiB / 133,706-message snapshot for ~12s only to write `changed=0`. The prior
> closure proved missing-materialization backlog was zero after convergence; it
> did not prove cold-rebuild replay amplification was solved.

Proposed design replacement for the residual:

> Add a proof-carrying cold-rebuild replay plan. Preserve exhaustive replay as
> reference. First slice may cover an older source row only when a strictly later
> row in the same `(origin, exact source_path, source_index=0)` context is a
> cryptographically proven byte-prefix extension and its provider/path class is
> registered prefix-monotone (initially ordinary Codex and Claude Code JSONL).
> Retain all raw evidence; never prune append (`-1`), bundle/split, divergent,
> truncation, tie, failed/unparsed, missing-blob, or unsupported rows. Emit
> coverage edges and bytes avoided. Verify optimized vs exhaustive logical index
> equivalence through the deterministic-rebuild differential helper.

Proposed acceptance criteria:

> A real-path Codex + Claude growing-snapshot fixture proves exhaustive and
> optimized cold rebuilds are logically identical while N prefix snapshots cause
> exactly one parse and all source rows remain durable. Counterexamples cover
> divergence, truncation/ties, append fragments, bundles/split rows,
> path/origin-sensitive parsing, prior parse failure, and missing blobs. Mutation
> faults demonstrate the lane fails when pruning is disabled and when each safety
> guard is removed. `rebuild-index --plan` reports executed/covered rows,
> prefix-scan bytes, parse bytes, bytes avoided, and divergent chains. A read-only
> plan over the 2026-07-10 measured source tier records actual eligible bytes; no
> correctness claim is based on `(origin,native_id)` or recency alone.

## Judgment

**A small safe first slice exists.** It is prefix-proven replay coverage for
ordinary growing Codex/Claude Code full JSONL snapshots, with exhaustive fallback
and a differential proof. It likely addresses the two measured ~10 GiB chains
without deleting a byte of durable evidence or changing parser/write semantics.

What does **not** exist is a safe one-query “latest row per native id” shortcut.
That would violate append/bundle semantics, path-sensitive parsing, `0mu`
precedence, and raw-link behavior. Broader supersession needs richer parse
receipts and a source-fidelity design, not an opportunistic SQL window function.

## Follow-up: `full_replace.delete_messages` amplification

### New observation

During the same cold rebuild, journal timings for repeated Claude full-snapshot
replacement rose from approximately 16 seconds at 1,347 messages to 110 seconds
at 6,144 messages. This timing is not expected linear row deletion. Source review
identifies one specific missing foreign-key support index that turns the parent
delete into repeated global child-table scans.

### Exact transaction path

`_replace_full_session_messages_and_blocks()` owns the timing
(`polylogue/storage/sqlite/archive_tiers/write.py:1721-1789`). With ordinary
foreign keys enabled, its order is:

1. Delete this session's FTS rows in one scoped statement.
2. Drop the three block-backed `messages_fts` triggers.
3. `_clear_session_projection_rows()` deletes blocks and all other
   session-scoped message dependents.
4. Execute `DELETE FROM messages WHERE session_id = ?`; this is the measured
   `index.full_replace.delete_messages` stage.
5. Insert replacement messages and blocks.
6. Reinsert the session's FTS rows and restore the triggers.

All of this remains inside the caller's transaction. There is no intermediate
commit, so the replacement stays atomic.

For batches over the 64 MiB raw threshold, the outer ingest path instead turns
foreign keys off before `BEGIN IMMEDIATE`, drops FTS triggers, performs explicit
session-scoped FK validation, and commits the batch
(`pipeline/services/ingest_batch/_core.py:961-1115`). The observed Claude rows
can remain below that byte threshold despite containing thousands of messages,
so they follow the ordinary FK-on path above.

### Why message FTS is not the timed culprit

`messages_fts` is contentless-delete FTS5, with `blocks` insert/delete/update
triggers (`storage/sqlite/archive_tiers/index.py:316-344`). In the full-replace
path, however, its rows are deleted under the separately measured `fts_delete`
stage, and its triggers are dropped before blocks or messages are deleted. Block
deletion and any remaining block-backed work are measured under
`clear_projection_rows`, not `delete_messages`.

There is a small redundant operation: after the caller explicitly deletes FTS
rows and drops triggers, `_clear_session_projection_rows()` sees the delete
trigger missing and calls the scoped FTS deletion helper again
(`write.py:608-657`). The second delete should find no matching docsize rows; it
may add avoidable work to `clear_projection_rows`, but it cannot explain a
110-second `delete_messages` timer.

The external-content `blocks_command_trigram` delete trigger is not part of the
message-FTS trigger suspension set. It fires while `blocks` are cleared, so it is
also charged to `clear_projection_rows`, not the measured message delete.

### Foreign-key child lookup census

Deleting a `messages` parent row requires SQLite to find rows in every table
whose FK references `messages(message_id)`, even when the writer already deleted
those children. For each deleted parent, an unindexed child key requires a full
child-table scan.

The current DDL has usable leading indexes for the other references:

- `messages.parent_message_id`: `idx_messages_parent`;
- `blocks.message_id`: `PRIMARY KEY(message_id, position)`;
- `session_events.source_message_id`: `idx_session_events_source_message`;
- `session_agent_policies.source_message_id`:
  `idx_session_agent_policies_source_message`;
- `attachment_refs.message_id`: `idx_attachment_refs_message` plus its PK;
- `paste_spans.message_id`: `PRIMARY KEY(message_id, position)`;
- `session_provider_usage_events.source_message_id`:
  `idx_session_provider_usage_events_source_message`.

`web_content_constructs.message_id` is the exception. It has
`ON DELETE CASCADE` to messages (`index.py:278-303`), but its indexes are only:

- primary key `(block_id, position)`;
- `(session_id, construct_type)`;
- partial indexes on `url` and `query`.

The primary key supports the separate cascade from `blocks.block_id`, but none
supports a lookup by `message_id`. `_clear_session_projection_rows()` deletes
blocks first, so target web constructs are normally already gone; nevertheless,
the subsequent message delete must prove there are no remaining direct
`message_id` children. It does so by scanning the global
`web_content_constructs` table once for every deleted message.

This gives a work shape of approximately:

```
deleted_messages * current_global_web_construct_rows
```

not merely `deleted_messages + deleted_blocks`. The 4.56x increase in message
count from 1,347 to 6,144 would already multiply a fixed child-table scan by
4.56; growth of `web_content_constructs` during the rebuild, cache eviction, and
write/WAL pressure can plausibly produce the observed 6.9x wall-time increase.
The mechanism is superlinear over rebuild progress and directly explains why
small fixtures miss it.

### Small safe fix

The direct fix is the existing open bead `polylogue-ma2`, whose title and design
match this finding exactly: add a canonical index on
`web_content_constructs(message_id)` in the derived index-tier DDL. This changes
no semantics and allows each FK action to do an indexed child lookup instead of
a full scan.

The correct DDL shape is:

```sql
CREATE INDEX IF NOT EXISTS idx_web_constructs_message
ON web_content_constructs(message_id);
```

The leading column must be `message_id`. Extending the existing
`(session_id, construct_type)` index or adding `(session_id, message_id)` does
not serve SQLite's FK query, because the parent delete provides only a message
id.

Per the derived-tier regime, this belongs in canonical DDL with one batched
`INDEX_SCHEMA_VERSION` bump and a rebuild plan, not an in-place upgrade helper.
The fact that the v29 rebuild is now spending 110 seconds in this known path is
evidence that deferring `ma2` through the recent schema window was costly. The
next index window should include it; do not trigger yet another live rebuild
solely from this audit without coordinating the other queued index changes.

Lowering the 64 MiB “foreign keys off” threshold is not the safe fix. That path
depends on explicit FK emulation and scoped validation and would broaden a
correctness-sensitive mode merely to mask one missing index. Removing the direct
message FK from `web_content_constructs` is likewise unjustified; the index is
the narrow relational repair.

### Relationship to prefix replay coverage

The two fixes attack different multipliers:

- prefix-proven replay coverage reduces the number of historical full snapshots
  that reach parse and replacement at all;
- `ma2` removes the global child-table scan from every unavoidable full
  replacement, including divergent snapshots, ordinary reingest, and non-prefix
  providers.

Prefix coverage is therefore not the only correct mitigation. Even perfect
coverage of the measured snapshot chains would leave this FK pathology on every
legitimate replacement. Conversely, the index alone would make each replacement
cheaper but would still repeatedly parse and rewrite historical prefix snapshots.
Both are warranted.

### Focused benchmark and regression proof

Use the real canonical index DDL and real SQLite FK actions; do not mock the
delete or create a toy schema.

1. Create a fresh archive with `PRAGMA foreign_keys=ON`.
2. Seed a target session with parameterized message count `M` and a large set of
   unrelated sessions carrying total `W` web constructs. Assert both populations
   are non-zero and at the requested sizes.
3. Run the exact full-replace clear/delete path, preserving the production order
   where blocks and target constructs are cleared before messages.
4. Use `EXPLAIN QUERY PLAN` on SQLite's equivalent child probe:

   ```sql
   SELECT rowid FROM web_content_constructs WHERE message_id = ?;
   ```

   Require `SEARCH ... USING COVERING INDEX idx_web_constructs_message`, never
   `SCAN web_content_constructs`.
5. Record SQLite progress-handler/VDBE step counts for the timed message delete at
   `M = 10/100` and `W = 1,000/10,000`. Prefer a generous structural bound on
   step growth over a wall-clock assertion. With the index, increasing unrelated
   `W` must not multiply delete steps approximately by ten.
6. Verify the cascade contract separately with a target construct still present:
   deleting its message removes the construct with FKs on.
7. Keep the existing full replacement/FTS consistency assertions green so the
   benchmark cannot “optimize” by disabling foreign keys or triggers globally.

The optional scale lane should also report the existing
`index.full_replace.delete_messages` stage at message counts near 1k and 6k, with
and without the index, against the same seeded global web-construct population.
The acceptance signal is disappearance of the global-table-size multiplier, not
a hard-coded 300 ms budget.

### Anti-vacuity faults

The focused proof must fail under these deliberate faults:

1. Drop `idx_web_constructs_message`: the plan regresses to
   `SCAN web_content_constructs`, and progress steps regain the `M * W` shape.
2. Create the plausible but wrong `(session_id, message_id)` index: the
   message-only child probe still scans, proving leading-column order matters.
3. Set `PRAGMA foreign_keys=OFF`: the harness rejects the setup before timing,
   so a no-op FK path cannot appear fast and green.
4. Seed zero unrelated web constructs: the harness rejects the fixture, preventing
   the small/empty-table false green that hid the production behavior.
5. Create the index ad hoc only in the test: a separate fresh-schema DDL contract
   requires the canonical index name/columns, preventing fixture-only repair.
6. Leave a target construct present and break its cascade: the relational
   correctness assertion fails even if the planner/timing assertions pass.

### Judgment

The 16s-to-110s `delete_messages` escalation is a known missing-index bug, not
expected contentless-FTS cost. `polylogue-ma2` is correctly scoped and should be
treated as the direct fix. Prefix-proven replay coverage remains valuable because
it prevents redundant replacements, but it is complementary rather than a
substitute for the FK-supporting index.
