# Content, identity, and lineage architecture

Status: implementation design for `polylogue-a7xr.25`, `polylogue-6e7m`, `polylogue-4ts`, `polylogue-nas1`, `polylogue-2qx`, `polylogue-qj5x`, `polylogue-a7xr.23`, `polylogue-a7xr.24`, and `polylogue-83u`.

This design is the authority for the next implementation train. It is grounded in the code and schemas at `25434d0f0`, the complete Beads threads listed above, the live archive measured on 2026-08-03, and the relevant history through PRs #3246, #3250, #3252, #3416, #3643, #3655, #3662, #3663, #3669, and #3691. It does not assume that an older plan describes current behavior when source or history contradicts it.

## Decisions at a glance

| Bead | Decision | Architecture gate |
| --- | --- | --- |
| `polylogue-a7xr.25` | Transform `session_events`. Keep the ordered protocol timeline, replace duplicated content with typed references to messages or blocks, retain event-only transport and protocol evidence inline, and keep unknown event types losslessly inline. | Yes, for the coordinated index reparse. |
| `polylogue-6e7m` | Store only provider-authoritative titles as `sessions.title`; compute a deterministic display label on read; hash only authoritative title evidence under a versioned content-hash contract. | Yes, for hash v2 and the coordinated reparse. |
| `polylogue-4ts` | Keep `session_links` as the topology and composition relation. Permit exactly one inheritance-bearing composition parent, compose only an exact leading prefix, model compaction boundaries separately, and never perform global content-based transcript deduplication. | Yes. |
| `polylogue-nas1` | Reserve `resume` topology for exact provider-native evidence. Model context preparation, delivery, and successor association as a separate evidence graph joined at read time. | The native resume distinction gates lineage; the context-delivery extension can land in parallel. |
| `polylogue-2qx` | Extend the existing `OriginSpec` into the single executable admission and normalization contract. It owns title authority, positive material-origin evidence, lineage assertions, event disposition, provenance, fidelity, and coverage. | Yes, before parser and hash changes. |
| `polylogue-qj5x` | Remove normalized Beads issue sessions after an all-archive durable-data census. Keep Beads work evidence in the work-effect and claim graph. | Targeted follow-up, not a rebuild gate. |
| `polylogue-a7xr.23` | Make content-defined chunks the durable byte representation in `source.db`; keep an `ops.db` byte cursor only as a disposable tailing optimization. Preserve full observation identity and parser-visible byte streams. | The raw authority contract is a prerequisite for later blob cleanup, but it need not block the index reparse. |
| `polylogue-a7xr.24` | Continue the existing spec-driven mapper work after semantic schema changes settle. It is a mechanical reconciliation project, not an identity abstraction. | Independent and serialized after shared DDL hot spots. |
| `polylogue-83u` | Continue attachment integrity independently. Attachment bytes and metadata already participate in the canonical hash. CDC and attachment storage must share blob integrity and GC invariants, but attachment semantics do not gate lineage. | Independent. Blob compression or GC work rebases after CDC. |

The rejected umbrella alternative was to treat every item as one schema rewrite. That would couple durable raw storage, derived transcript normalization, display behavior, and user evidence migrations without a shared atomicity boundary. The selected train has one semantic index rebuild, separately backed durable migrations, and explicit rebase points at the few files that are actual hot spots.

## 1. Current-state inventory

### 1.1 Identity layers and generated identifiers

The current archive has several identities with different purposes. Conflating them caused the title and cursor proposals to look simpler than they are.

| Identity | Current authority | Stability and use |
| --- | --- | --- |
| Native session identity | Provider/runtime `native_id` interpreted within normalized `Origin` | Stable primary evidence. A native id is not globally meaningful without origin. |
| Normalized session id | `sessions.session_id`, generated as `origin || ':' || native_id` in `polylogue/storage/sqlite/archive_tiers/index.py` | Stable logical archive key. Display changes and content revisions do not change it. |
| Message id | `messages.message_id`, generated from `session_id` and `COALESCE(native_id, position || '.' || variant_index)` | Stable within a session revision when native identity or positional fallback is stable. Prefix extraction moves rows between parent and child physical storage, so composed reads must use logical provenance rather than assume every message is physically stored on the requested session. |
| Block id | `blocks.block_id`, generated as `message_id || ':' || position` | Stable within the owning message revision. |
| Canonical session content hash | `compute_session_content_hash()` and `_session_hash_payload()` in `polylogue/pipeline/ids.py`, using normalization helpers in `polylogue/core/hashing.py` | Revision and idempotency fingerprint. It is not a primary id. |
| Row-level message hash | `_message_content_hash()` in `polylogue/storage/sqlite/archive_tiers/write.py` | Derived-row freshness fingerprint. It currently includes storage identity and `material_origin`, so it is not suitable for cross-session duplicate measurement. |
| Identity-free block hash | `_block_content_hash()` in `write.py` | Derived content equality signal. It omits session/message identity and is the correct bounded signal for duplicate-content measurement. It is not permission to coalesce blocks. |
| Raw observation id | SHA-256 of acquired observation bytes in the source tier | Durable acquisition evidence. It remains independent of normalized session hashes and of CDC manifest identity. |
| Source identity | `Source` family/runtime root/originating lab plus acquisition artifact identity | Explains where bytes came from. It is richer than `Origin` and is not reversible from `Origin`. |
| Origin | `Origin` in `polylogue/core/enums.py` and the registry in `polylogue/sources/origin_specs.py` | Public normalized source token. It does not replace provider-wire or source identity. |
| Display label | `polylogue/insights/session_label.py` | Read projection. It must not affect acquisition, session primary identity, or content hash. |
| Lineage identity | `session_links` key `(src_session_id, dst_origin, dst_native_id, link_type)` | One asserted topology relation. Resolution to `resolved_dst_session_id` is late and repeatable. |
| Byte chunk identity | Not yet present | This design assigns SHA-256 of exact chunk bytes. It is storage identity only, not parser or session identity. |

The generated-column DDL is in `polylogue/storage/sqlite/archive_tiers/index.py`. `sessions` is keyed by `(origin, native_id)`. `messages` is keyed by `(session_id, position, variant_index)`. `blocks` is keyed by `(message_id, position)`. All three public ids are generated rather than redundantly written. The inspected schema versions are index 60, source 24, user 10, ops 1, and embeddings 4.

### 1.2 Canonical hash inputs and call sites

`_session_hash_payload()` currently hashes a normalized mapping containing title, timestamps, ordered messages, ordered session events, and sorted attachments. `_message_hash_payload()` includes message native id, role, text, timestamp, and content blocks. Event payloads include list index, type, timestamp, source provider-message id, and the complete event payload. Attachment payloads include metadata and, when acquired, a precomputed or inline byte-content hash. NFC normalization and distinct sentinels for null, empty, and missing values are supplied by `polylogue/core/hashing.py`.

Two corrections are load-bearing:

1. `material_origin` is not currently in the session-level hash payload. It is in the row-level `_message_content_hash()` in `write.py`. A material-origin correction can therefore change rebuilt message rows without invalidating a current raw normalized-content membership, which is dishonest under the intended semantic contract.
2. `semantic_type`, block language, structured tool-result outcome fields, `message_type`, and `stop_reason` are not all represented in the session hash today. A hash-contract migration must close these gaps deliberately rather than only remove title.

Production session-hash callers exist in ingest preparation and workers, prepared-row construction, revision-governance and replay paths, repair and backfill paths, and live batch ingestion. The durable copy also appears as `raw_session_memberships.normalized_content_hash` in `source.db`; accepted session content hashes and parsed rows in `index.db` are derived. A version change therefore needs a durable membership version marker even though the parsed tree is rebuilt.

### 1.3 Session events

`session_events` DDL is in `polylogue/storage/sqlite/archive_tiers/index.py`. Its generated event id is `session_id || ':' || position`; it stores an optional source-message reference, provider message id, ordered position, event type, summary, JSON payload, and timestamp. The write choke point is `_write_session_events()` in `polylogue/storage/sqlite/archive_tiers/write.py`.

The writer already omits redundant event families `token_count`, `message_usage`, `agent_policy`, `agent_message`, and `agent_reasoning`, after their typed projections or content rows are written. A live archive that predates the latest filter still contains `agent_reasoning`, so live row counts describe the deployed generation, not the exact output of a clean rebuild at current HEAD.

Readers are broader than transcript rendering. `polylogue/storage/sqlite/queries/session_events.py` provides single, batch, and synchronous reads. Events feed compaction counts and boundaries, phase extraction, workflow materialization, Hermes projections, active-duration calculations, API models, CLI and MCP read surfaces, demo verification, and insight rebuilds. This consumer census rules out deleting the relation.

Codex `function_call`, `function_call_output`, custom tool-call variants, and reasoning content are lowered into tool-use, tool-result, or thinking blocks. Their event copies are redundant for content but still useful as ordered protocol occurrences. Claude sidecar and execution-result events are different: they carry acquisition matching state, filenames, byte sizes and hashes, replacement state, sandbox/path facts, hunk metadata, or other evidence absent from result blocks. `patch_apply_end` likewise preserves provider structural changes or a unified diff. A family-wide payload deletion would lose meaning.

### 1.4 Lineage and composition

`session_links` DDL is in `archive_tiers/index.py`. Its key is `(src_session_id, dst_origin, dst_native_id, link_type)`. It stores late resolution, evidence, optional parent tool-use id, optional `branch_point_message_id`, and `inheritance` of `prefix-sharing`, `spawned-fresh`, or null. Ordinary resolvedness is expressed by `resolved_dst_session_id`; `TopologyEdgeStatus` is reserved for exceptional `repaired` and `quarantined` outcomes as well as the existing vocabulary.

The sole production write and resolution engine is in `polylogue/storage/sqlite/archive_tiers/write.py`: `_write_session_link()`, `_resolve_outbound_session_links()`, and `_resolve_session_graph()`, invoked unconditionally by `write_parsed_session_to_archive()`. The similarly named resolver in `polylogue/storage/sqlite/queries/session_links.py` is read-only test infrastructure and has no production writer call. The statement in `docs/internals.md` that names it as the production resolver is stale.

Prefix extraction uses the fully composed parent and removes only a contiguous, exactly equal leading prefix. Zero matching messages records `spawned-fresh`; a positive match records `prefix-sharing`, its final inherited message id, and stores only the divergent child tail. `branch_point_message_id` is deliberately not a foreign key because a parent full-replace deletes and reinserts rows within one transaction. An `ON DELETE SET NULL` action would permanently erase the branch point before the parent rows return.

Read composition is iterative and snapshot-held. It walks the resolved composition chain, verifies the branch point, prepends the parent through the branch message, and returns a completeness signal when a parent or branch point is unavailable. Existing focused tests cover parent-first and child-first ingestion, reingest, variants, delete indexes, deep chains, missing branches, synchronous and asynchronous reads, and cycle quarantine. History confirms that write-time `_resolve_session_graph()` superseded the dead standalone cycle engine removed by PR #3655.

Parsers currently assert one parent on `ParsedSession` in `polylogue/sources/parsers/base_models.py`. The complete current assertion audit is:

- Codex in `polylogue/sources/parsers/codex.py` reads the child's `session_meta.forked_from_id`, `source.subagent.thread_spawn`, and a legacy replayed second `session_meta` through `_has_continuation_evidence()`. Exact `forked_from_id` proves a parent but does not distinguish fork from resume. The legacy timestamp plus CWD/repository path currently promotes to continuation and is stronger than its evidence.
- Claude Code in `polylogue/sources/parsers/claude/code_parser.py` uses `agent-acompact-*`, `isSidechain`, fresh-task prompt shape, and parent/session ids to emit continuation, sidechain, or subagent. `acompact` is overloaded across fresh and continuation shapes.
- Hermes state in `polylogue/sources/parsers/hermes_state.py` uses explicit `parent_session_id`, parent rows, and compression/compaction end reasons. Hermes span and verification parsers use profile-root relationships, positive `hermes.subagent.*` marks, and `subagent_trajectories` child ids to emit parent/subagent evidence. The fidelity reports already distinguish observed from unavailable subagent evidence.

No other parser writes a production composition edge without flowing through these `ParsedSession` fields and `_write_session_link()`.

### 1.5 Origin and material-origin classification

The existing `OriginSpec` registry already owns origin identity, provider wires, lifecycle mode, acquisition and artifact rules, detector/parser references, collision policy, assembly, fixtures, coverage and fidelity metadata, and reparse policy. Detection dispatch remains manually sequenced, with parity validation against the registry. This is partial single authority, not a blank slate.

The shared classifier is `classify_material_origin()` in `polylogue/archive/message/artifacts.py`. Parser-specific positive-evidence overrides appear in parser modules and helpers, including `_codex_material_origin()` in `polylogue/sources/parsers/codex.py`, `human_authored_override()` in `polylogue/sources/parsers/base_support.py`, Claude common/code parsing, Drive, ChatGPT, Antigravity, Grok, and Hermes state/spans. `ParsedSession` also performs a runtime-root-based upgrade of otherwise unknown user messages. Current positive evidence is inconsistent: Codex still upgrades every user `MESSAGE` that survived shared classification to `HUMAN_AUTHORED`, while other origins require origin-specific structure.

### 1.6 Raw acquisition and cursors

`source.db` is durable and authoritative for acquired bytes, raw observations, memberships, artifact classification, blob references, publication reservations, and GC generations. `ops.db` is disposable and stores ingest cursors, attempts, lag, convergence debt, and operational logs. Current acquisition revision fields `revision_kind` and `append_end_offset` participate in replay, retention, revision-governance, live watcher, and raw-authority checks. The live archive has 4,344 append observations and every one has a non-null append end offset. The older Bead note that the field was universally null is no longer true.

Cursor readers and writers include live watcher and catch-up planning, cursor lifecycle and lag surfaces, append cursor resynthesis, and daemon status. This supports retaining the cursor as an optimization. It does not support treating a cursor as durable byte evidence.

### 1.7 DDL reconciliation and attachment integrity

PR #3662 already introduced `MESSAGES_SPEC` and `BLOCKS_SPEC` so message and block DDL, inserts, selects, mappers, and hydration share `TableColumnSpec` definitions. `polylogue-a7xr.24` therefore describes a partially landed program. Sessions, attachments, and session events still have manual seams; runtime record types and semantic facts should not be dynamically synthesized merely to increase the percentage.

Attachment records are first-class and attachment metadata plus acquired byte content already enter the canonical session hash in `pipeline/ids.py`. Blob acquisition status and real byte hashes were repaired in prior v13 work. Attachment compression or storage relocation must preserve the byte hash, the session hash, and blob-reference liveness. Those are constraints on CDC, not evidence that attachment semantics belong inside the lineage gate.

## 2. Live evidence

All commands in this section were read-only. SQLite was opened with `-readonly` or URI read-only mode.

The live archive at `/realm/db/polylogue` contained:

| Relation | Rows |
| --- | ---: |
| sessions | 23,496 |
| messages | 4,949,871 |
| blocks | 5,070,427 |
| session_events | 7,403,923 |
| session_links | 9,497 |

The largest event families were `function_call` 1,614,517 rows and 208.0 MiB of JSON payload, `function_call_output` 1,614,209 and 174.1 MiB, `reasoning` 1,153,236 and 52.7 MiB, `claude_tool_result_sidecar` 578,241 and 71.3 MiB, `claude_tool_execution_result` 448,463 and 39.3 MiB, `turn_context` 402,879 and 59.5 MiB, `patch_apply_end` 92,323 and 209.4 MiB, `claude_attachment` 83,418 and 150.9 MiB, and `queue` 57,396 and 331.3 MiB. The top measured families total about 1.8 GiB of payload, not tens of GiB. The event change primarily removes duplicate payload writes while preserving occurrence rows, with additional row removal only for families already represented by typed projections.

Link shapes were dominated by 7,371 resolved spawned-fresh subagent links, 1,413 unresolved subagent links without inheritance, 348 resolved prefix-sharing subagent links, 189 resolved prefix-sharing continuation links, and 119 resolved spawned-fresh continuation links. There were no persisted `fork` or `resume` link types. No source session had more than one inheritance-bearing link, but the schema does not yet enforce that observed invariant.

The title census showed all 3,204 Codex sessions with `title == native_id`; all 16,552 Claude Code sessions had non-null titles, of which 14,801 equaled native id, 273 were origin-authoritative, and 1,431 were heuristic. This is deployed-generation evidence. It explains why title fallback must be removed from stored identity even though recent parser fixes have improved fresh imports.

`source.db` contained 43,124 raw sessions totaling 95,503.2 MiB: 20,298 unknown revisions totaling 53,385.3 MiB, 18,482 full revisions totaling 32,014.2 MiB, and 4,344 append revisions totaling 10,103.7 MiB. The source and index archives both contained zero Beads-origin rows. The approximately 924 Beads sessions in `polylogue-qj5x` are a projected import count from configured artifacts, not a live migration count.

The full duplicate-block query in the Bead measured 258,396 duplicated eligible blocks out of 975,767, or 26.5 percent. A bounded deterministic 1/32 hash-range remeasurement completed against the live archive:

```sql
WITH eligible AS MATERIALIZED (
    SELECT b.content_hash, b.session_id
    FROM blocks b INDEXED BY idx_blocks_content_hash
    JOIN messages m ON m.message_id = b.message_id
    WHERE b.content_hash >= X'0000'
      AND b.content_hash < X'0800'
      AND b.block_type IN ('text', 'thinking', 'reasoning')
      AND length(COALESCE(b.text, '')) > 40
      AND m.material_origin IN ('human_authored', 'assistant_authored')
), duplicated AS (
    SELECT content_hash
    FROM eligible
    GROUP BY content_hash
    HAVING count(DISTINCT session_id) >= 2
)
SELECT
    (SELECT count(*) FROM eligible),
    (SELECT count(*) FROM eligible e JOIN duplicated d USING (content_hash)),
    round(100.0 * (SELECT count(*) FROM eligible e JOIN duplicated d USING (content_hash))
        / (SELECT count(*) FROM eligible), 1);
```

Command:

```text
sqlite3 -readonly /realm/db/polylogue/index.db '<query above>'
```

Result:

```text
30934|8583|27.7
```

The shard directionally reproduces 26.5 percent. Both measurements prove cross-session content duplication, not that every duplicate is a lineage prefix. The next lineage evidence pass must join duplicate blocks to resolved composition chains and report lineage-linked versus unrelated duplicate groups before any broader suppression is authorized.

## 3. Dependency graph and execution boundaries

```text
OriginSpec contract (2qx)
    |--> parser title/material/hash-v2 semantics (6e7m)
    |--> parser lineage evidence and branch types (4ts, native part of nas1)
    `--> event disposition declarations (a7xr.25)

parser lineage evidence --------> session_links/composition storage (4ts)
                                      |--> event content references after prefix remap (a7xr.25)
                                      `--> one coordinated semantic index reparse (818fy)

context delivery durable schema (nas1) --------> joined continuation read model
    parallel with parser/index work; never writes session_links

CDC source schema and reader (a7xr.23) --------> manifest cutover --------> later blob cleanup
    parallel with index work; source-tier backup and separate rollout

qj5x all-archive census --------> operator consent --------> targeted Beads-origin removal
    may ride the final derived rebuild, but does not gate it

a7xr.24 mechanical reconciliation
    starts after semantic changes to index.py, write.py, and runtime records settle

83u attachment census and forward capture
    parallel; compression and GC changes rebase after CDC blob-ref changes
```

Hard prerequisites are semantic, not merely issue dependencies:

1. `OriginSpec` contract fields and conformance laws land before individual parsers adopt them.
2. Parser lineage evidence lands before storage migration asserts that a `resume`, `fork`, or `continuation` token is trustworthy.
3. Composition and prefix-remap semantics land before event rows refer to a block position in a potentially inherited message.
4. The hash-contract version and durable membership version marker land before any v2 hash is written.

Serialized hot spots are `polylogue/storage/sqlite/archive_tiers/index.py`, `polylogue/storage/sqlite/archive_tiers/write.py`, parser base models and branch enums, and source-tier blob/ref DDL. `a7xr.25` follows the lineage storage lane. `a7xr.24` follows all semantic DDL lanes. CDC and attachment GC do not edit blob-reference authority concurrently.

## 4. Canonical identity model

### 4.1 Required separation

The canonical model has seven independent axes:

1. **Display title** is a read-time label for humans. It may change as projection quality or collision context improves. It never initiates reingest and never enters the canonical content hash.
2. **Provider title evidence** is normalized content. A provider-supplied title can change the canonical hash because it is part of the acquired session meaning. A parser-generated prompt echo, native-id fallback, or structural label is not provider title evidence.
3. **Canonical content hash** identifies a normalized revision under an explicit hash-contract version. Equal native identity with a different current-version hash is a new normalized revision. Equal hashes under different versions are not comparable.
4. **Native identity** is the provider/runtime key. Combined with normalized origin it generates `session_id`. Content changes do not change it.
5. **Normalized origin and source identity** answer different questions. `Origin` is the public normalized source token. `Source` and raw artifact identity preserve family, runtime root, lab, acquisition path, and provider-wire evidence. Code must not reverse-map origin into a guessed provider.
6. **Session lineage** is an asserted relation among stable session ids plus independent storage facts about inheritance. It does not change a session id or canonical hash.
7. **Byte and chunk identity** preserve exact acquired evidence. Raw observation identity is the hash of the complete observed bytes. Chunk identity and manifest identity are storage-level deduplication devices and are invisible to parsers.

### 4.2 Change matrix

| Change | Reingest/reparse | Hash change | Stable session id |
| --- | --- | --- | --- |
| Display-label algorithm or collision suffix | No | No | Yes |
| Provider title evidence changes | Yes | Yes | Yes |
| Heuristic/native fallback title changes | No after migration | No | Yes |
| `material_origin`, `message_type`, or normalized semantic block changes | Yes | Yes in hash v2 | Yes |
| Token count, derived cost, embedding, insight, or FTS repair | No | No | Yes |
| Parent resolves late without parser evidence changing | No raw reparse; derived composition repair | No | Yes |
| Parser changes asserted lineage or compaction evidence | Yes | Yes through canonical event/message semantics where applicable | Yes |
| Chunk boundaries or compression change while bytes are identical | No | No | Yes |
| Acquired bytes change | Yes | Usually yes after normalization | Yes if native identity is unchanged |

The hash contract must be named `session-content-v2` in code and stored as an integer or closed token next to durable normalized memberships. A version marker inside the hashed payload prevents an accidental v1/v2 equality.

## 5. Session-events disposition

### 5.1 Decision: transform the relation

Keep `session_events` as the ordered protocol-evidence relation. Add nullable `content_ref_kind` with values `message` or `block`, nullable `content_ref_message_id`, and nullable `content_ref_block_position`. Keep `source_message_id` independent because the provider message that emitted an event is not always the normalized content row that carries its authored/tool meaning. A block ref is `(content_ref_message_id, content_ref_block_position)` and reconstructs the generated block id; a message ref has a null block position. These are deliberately not foreign keys because parent full-replace and prefix re-extraction can temporarily remove their target rows within the write transaction. Rebuild and read validation enforce referential integrity. `payload` is redefined as event-only evidence, not a second content envelope.

Each event type has one OriginSpec disposition:

| Disposition | Stored event shape | Examples |
| --- | --- | --- |
| Typed projection only | Existing writer omission remains. Meaning is fully represented in typed tables. | `token_count`, `message_usage`, `agent_policy`, `agent_message`, `agent_reasoning` after clean rebuild. |
| Content reference | Keep position, type, timestamp, summary, provider ids, content ref, and unique transport metadata. Remove content duplicated by the referenced block/message. | Codex `function_call`, `function_call_output`, custom tool-call variants, and reasoning content. |
| Inline protocol evidence | Keep the event payload losslessly because it contains meaning not represented in the transcript tree. | lifecycle, queue, turn context, sidecars, execution-result envelopes, patch structural evidence, acquisition and enrichment facts. |
| Unknown | Keep inline losslessly and record an OriginSpec coverage gap. | A newly observed provider event type. |

This is a closed decision per known event family and a lossless default for future families. Event content cannot be removed merely because its type name contains `tool` or `reasoning`.

### 5.2 Compatibility and reads

`SessionEventRecord` and public envelopes expose an optional typed content reference. `payload` remains available but is documented as event-only metadata. Consumers that need authored text, thinking, tool input, or tool result content must read the referenced message/block or the `actions` view. Consumers that need ordering, transport lifecycle, active duration, compaction, sidecar acquisition, workflow evidence, or protocol diagnostics continue to read events.

There is no long-lived compatibility shim that reconstructs the old duplicated payload. Such a shim would preserve the storage mistake and make its removal untestable. The consumer audit found no legitimate production reader that requires the duplicated content specifically from lowered event payloads. During implementation, a field-by-field event retention table must be checked in beside OriginSpec coverage so that a new consumer cannot infer old semantics from an arbitrary JSON shape.

Prefix-sharing requires one rule: event content references are remapped with the same physical-to-logical message mapping used during `_extract_prefix_tail()`. If the referenced content was inherited, the event row on the child points to the composed parent message/block. This phase does not suppress event occurrences across sessions because no lineage-composed event relation exists today. It removes duplicated payload bytes and preserves each provider-observed event timeline. Event-prefix normalization needs a separate measurement and a composed-event read contract before it can be authorized.

### 5.3 Hashing, migration, and proof

Canonical hashing occurs over the complete parsed event before storage compaction. Transforming an event payload into a reference therefore does not erase evidence from `session-content-v2`. The storage relation and the canonical hash payload are intentionally not byte-for-byte twins.

This is a derived, semantic index change. Bump the index version with a `SEMANTIC_REPARSE` delta and include it in the coordinated `polylogue-818fy` raw replay. No in-place row rewrite can prove that every content ref points at the block produced by the current parser.

Anti-vacuity proof must use real parser output and the production write/read path. For every reference-disposition family, assert that the old content bytes are absent from event JSON, the referenced block contains them, and a composed child read still resolves the reference. Removing the sibling block or the remap must fail the test. For every inline family, mutate one event-only field such as sidecar byte hash, patch path, or replacement state and assert that round-trip evidence changes. A generated matrix that merely restates the disposition enum is insufficient.

The rejected alternatives were full deletion, because ordered protocol and compaction consumers are real, and broad inline retention, because it preserves 7.4 million duplicate-heavy writes. Narrowing without typed references was also rejected because it would force consumers to rediscover content by timestamp or tool id.

## 6. Title derivation and content-hash migration

### 6.1 Stored title and display label

`sessions.title` contains only a provider-authoritative title, with `title_source = origin`. Null, empty, and whitespace-only provider titles normalize to SQL null. Unicode is NFC normalized and surrounding whitespace is trimmed; internal whitespace is preserved. A raw native id, first-prompt echo, parser heuristic, repository fallback, or generated phrase does not enter `sessions.title`.

Existing `title_source = heuristic` rows may be retained only for a short migration diagnostic. They are excluded from the canonical hash and display projection, then disappear on the coordinated reparse. New writers do not persist them as titles.

The read-time display label uses this deterministic precedence:

1. authoritative provider title;
2. provider-supplied `display_name` when its OriginSpec classifies it as a label rather than identity;
3. repository name;
4. dominant normalized path plus message count;
5. origin label plus message count and UTC start date;
6. origin label plus native-id prefix for a zero-message session.

`polylogue/insights/session_label.py` remains the implementation home. Its current structural label is reused, but `_summary_from_row()` must stop treating heuristic title sources as authoritative.

Collision handling is an archive-wide read projection. Compute base labels for every current session in one batch, group exact normalized labels, and leave singleton labels unchanged. Every member of a collision group receives `[<origin>:<native-prefix>]`, using the shortest native-id prefix that is unique within that archive-wide group and extending through the full native id if needed. If duplicate native ids exist across origins, origin is already part of the suffix. Query subsets reuse the archive-wide result, so the same session does not acquire a different label merely because a filter changed. A label may become more specific when another session enters its collision group; that is allowed because display identity is not content identity.

Provider-specific fallback rules are declared in OriginSpec. Codex and Claude Code UUID/native-id fallback becomes null. ChatGPT and Claude AI retain actual exported titles. Drive, Hermes, Gemini CLI, Grok, and Antigravity retain only documented provider title fields; absent evidence follows the structural projection. Cross-provider conformance fixtures must include missing, blank, UUID-like, prompt-echo, and real-title cases.

### 6.2 Hash v2

`session-content-v2` hashes:

- normalized origin-authoritative title evidence only;
- created and updated timestamps under the existing normalization policy;
- ordered messages including native id, role, `message_type`, `material_origin`, text, timestamp, stop reason, and normalized content blocks;
- content-bearing block semantics including block type, semantic type, language where semantic, tool use/result structure, provider-reported error and exit outcome, and attachment references;
- ordered complete canonical session events before event-storage slimming;
- sorted attachment metadata and acquired byte-content hashes;
- an explicit hash-contract marker.

Derived measurements such as token counts, inferred cost, embeddings, summaries, and display labels remain outside the hash. Provider-reported usage already represented as canonical event evidence follows the OriginSpec event policy rather than being added twice.

Changing every hash through the contract marker is intentional. It avoids a mixed archive in which a v1 hash happens to equal a v2 hash while omitting new semantic fields. `raw_session_memberships` gains `normalized_content_hash_version INTEGER NOT NULL DEFAULT 1` through a numbered additive source migration. Replay writes v2 and version 2 atomically. Revision comparisons never compare hashes across versions.

The coordinated replay parses every accepted raw observation, computes v2 before prefix trimming or event storage transformation, writes the new derived generation, and updates durable membership hashes only after the normalized session write succeeds. Idempotency remains honest: same origin/native id plus same v2 canonical normalized evidence skips; a semantic classification or authoritative title change produces a new revision; a display-label or chunk-boundary change does not.

The rejected alternative was simply removing title from the existing payload. It would fix prompt-echo churn but leave material-origin and block-semantic changes invisible. Keeping heuristic titles in the hash was rejected because a display improvement would masquerade as new provider content. Creating a second permanent “identity hash” was rejected because session id already supplies stable identity and two revision hashes would drift.

## 7. Lineage and composition protocol

### 7.1 Edge and composition invariants

Relationship type and inheritance are orthogonal. `link_type` says what the provider asserted: `fork`, `resume`, `continuation`, `subagent`, `sidechain`, or generic `branch`. `inheritance` says what byte-normalized transcript comparison proved: exact prefix sharing, spawned fresh, or not evaluated/not a composition relation.

All asserted parent references are persisted even if the parent is absent. Replace the one-parent parser envelope with `lineage_edges: tuple[ParsedLineageEdge, ...]`; each edge carries destination origin/native id, relationship type, positive evidence, optional parent tool-use id, and `is_composition_parent`. Exactly zero or one parsed edge may be the composition candidate. This model is required to preserve multiple topology assertions without letting insertion order choose physical inheritance. Resolution always uses exact `(dst_origin, dst_native_id)`. A generic provider label must not be guessed into another origin. Late parent ingestion reruns `_resolve_session_graph()` and may populate `resolved_dst_session_id`, inheritance, and branch point without reparsing the child.

At most one link per source session may bear non-null inheritance and act as its composition parent. Add a partial unique index over `src_session_id WHERE inheritance IS NOT NULL`. The writer evaluates prefix inheritance only for the parsed edge marked `is_composition_parent`; other topology edges remain queryable with null inheritance. When evidence names multiple plausible parents without a unique composition authority, every edge is topology-only and the full child transcript remains physical. Do not choose by timestamp, insert order, or first resolution.

Prefix suppression remains exact and local: compose the parent first, compare the child's leading messages in order with current canonical equality, and remove only the maximal contiguous prefix. Never suppress an interior repeat, a non-leading block, or content shared with an unrelated session. A zero-length match records `spawned-fresh`. A positive match records `prefix-sharing` and the final inherited message id. `branch_point_message_id` remains non-FK.

A would-be composition edge that creates a cycle is retained as `quarantined`, has null inheritance, and is excluded from composition. Ordinary unresolved edges remain unresolved rather than mislabeled quarantined. Recomposition is iterative, cycle-bounded, and snapshot-consistent. Public reads expose `complete = false` and the missing/quarantined reason rather than silently returning a complete-looking partial transcript.

Aggregate and query contracts must name their grain: `physical` counts only rows stored on the session, `logical` counts the full lineage-composed transcript, and `effective-context` applies compaction replacement semantics. A generic `message_count` without a documented grain is prohibited on new lineage-aware surfaces.

### 7.2 Fork, resume, continuation, and subagent evidence

Add `BranchType.RESUME` to parser models and map it to `LinkType.RESUME`. The evidence rules are:

- **Resume:** exact provider/runtime evidence that this session resumes a named prior session. No time-window or same-repository inference.
- **Fork:** exact provider/runtime fork or branch marker that distinguishes a new divergent branch from resume.
- **Continuation:** exact provider/runtime continuation marker that is neither explicitly fork nor resume.
- **Subagent:** explicit parent session and subagent/spawn evidence. `parent_tool_use_id` is retained when supplied.
- **Generic branch:** an exact parent is known, but the relationship subtype is ambiguous.

Codex `forked_from_id` remains generic branch unless the wire adds a distinct resume/fork marker. It proves parent identity and permits prefix comparison, but the field name alone does not establish operator intent. A second `session_meta` may retain an exact parent reference as generic branch when its structure proves the reference; timestamp, CWD, and repository agreement must not promote it to continuation. This directly satisfies the positive-evidence constraint in `polylogue-2qx`.

Claude `acompact` is not itself a link type. The parser uses surrounding provider evidence to decide whether the new session is a continuation, subagent, generic branch, or fresh session, then records compaction separately. Hermes retains its explicit branch and subagent distinctions.

### 7.3 Compaction protocol

Add a derived index relation `session_compaction_boundaries` with key `(session_id, boundary_index)` and fields `summary_message_id`, `replaces_through_message_id`, optional `evidence_event_id`, and `occurred_at_ms`. The parser produces a normalized boundary from explicit provider compaction evidence. It does not infer one from a short summary-looking message.

Transcript and effective-context reads differ:

- The transcript view returns full lineage-composed history, including the earlier messages and compaction summary.
- The effective-context view applies the latest relevant boundary: the summary replaces messages through `replaces_through_message_id`, followed by later messages.
- An in-session compaction creates only a boundary.
- Cross-session auto-compaction creates the independently evidenced parent edge plus a boundary on the child.
- Fresh self-compaction records spawned-fresh inheritance when a parent relation is asserted but no prefix matches, plus its boundary.

This avoids overloading `branch_point_message_id` with context-window semantics. Branch point states where physical prefix storage ends; compaction boundary states which logical messages a summary replaces for effective context.

### 7.4 Provider-native resume versus context-assisted continuation

`session_links.resume` is reserved for provider-native topology. Calling a Polylogue context tool, preparing a resume brief, or delivering context must never create or upgrade a session link.

Durable context delivery remains in `user.db`. Add a typed optional `seed_session_ref` to `context_deliveries`, and add an append-only `context_delivery_successors` table keyed by delivery snapshot and successor session ref, with evidence ref and link timestamp. `ops.db` MCP call logs remain operational evidence that a preparation tool ran; they are disposable and cannot prove delivery or successor use.

A joined read model reports four distinct states:

1. provider-native resume, from exact `session_links.resume`;
2. context-assisted continuation, from a durable delivery with seed and explicitly associated successor;
3. provider continuation without recorded context delivery;
4. context prepared or delivered without a resolved successor.

The association actuator requires an explicit successor ref or later provider evidence that names the delivery. No tool-name plus temporal proximity heuristic is accepted. Work-graph `relates-to` edges can enrich this view once `polylogue-1vpm.6` is complete, but they are not a prerequisite for preserving and querying delivery evidence.

The rejected alternative was mapping context assistance into resume topology. It would conflate how a new session received context with the provider's session genealogy and make absence of MCP telemetry look like absence of lineage.

## 8. OriginSpec as the normalization authority

Extend the existing `OriginSpec`; do not create another registry. Every admitted origin declares:

- raw artifact classes, acquisition roots, lifecycle mode, and source/provenance fields;
- detector predicate and tightness rank, lowering mode, parser binding, assembly policy, and collision policy;
- provider title fields and their authority class;
- positive structural evidence for every non-unknown `material_origin` assignment;
- lineage markers and the strongest allowed `BranchType` for each marker;
- known session-event families and their typed-projection, content-reference, or inline disposition;
- normalized construct coverage, ignored fields with reasons, degraded fields, fidelity assertions, and representative fixtures;
- reparse policy and generated public origin/help/schema documentation.

Detection order is derived from detector tightness plus an explicit stable tie-break, while parser functions remain ordinary code referenced by the spec. This removes manual-order drift without trying to express parser implementation as data. Conformance validation checks that every public origin, detector, lowering path, parser, and fixture belongs to exactly one spec.

The shared material classifier remains conservative: absent positive evidence, return `UNKNOWN`. Origin-specific code may upgrade only when the spec declares the structural marker. Remove Codex's unconditional user-`MESSAGE` to `HUMAN_AUTHORED` fallback. Move the current `ParsedSession` runtime-root blanket upgrade behind an origin-declared structural guarantee; runtime-root absence alone is not universal proof of human authorship. Workflow-generated prompts, operator commands, runtime protocol/context, generated context packs, and tool results keep their distinct material origins even when the provider wire role is `user`.

Normalization order is fixed:

```text
acquire exact bytes -> classify raw artifact -> detect and lower -> parse
-> apply OriginSpec title/material/lineage/event policies -> assemble session
-> compute complete session-content-v2 hash -> persist durable membership
-> write normalized tree -> resolve lineage and extract physical prefix
-> materialize insights and indexes
```

Hashing before prefix extraction ensures a child's canonical revision describes the provider-observed normalized session, not the storage optimization chosen after its parent resolves. Hashing after OriginSpec classification ensures a material-origin correction is visible.

Changing an OriginSpec title, material-origin, lineage, normalized-field, or event-content rule is `SEMANTIC_REPARSE`. Changing documentation or a detector label without behavior is not. Coverage and fidelity changes identify affected origins and raw observations so rebuild selection can be explicit, but hash-v2 rollout intentionally reparses all accepted sessions once.

Cross-provider tests must run each representative fixture through production detection, lowering, parsing, hashing, writing, and reading. Required mutations remove or alter one positive marker and must fall back to `UNKNOWN` or a weaker generic branch; move a fixture into the shape of an earlier loose detector and ensure the tight detector still owns it; replace real title with prompt echo and ensure stored title is null; inject unknown event type and ensure lossless inline retention. A test that constructs an `OriginSpec` and asserts its own fields is vacuous.

## 9. Content-defined chunking and cursor architecture

### 9.1 Authority boundary

CDC is the durable byte representation in `source.db`. The full raw observation remains the unit of acquisition truth, admission, parser replay, retention, and normalized membership. Chunks are an implementation of exact-byte storage, not independent observations.

`ops.db.ingest_cursor.byte_offset` remains a disposable watcher optimization. It identifies where a live tailer last completed useful work. Deleting `ops.db` may cause a complete source reread and rechunk, but must never lose an admitted byte, change raw identity, or prevent replay. A cursor is never accepted as proof that bytes before it exist.

The Bead's original choice between CDC and durable cursors is therefore rejected as a false binary. CDC replaces durable prefix snapshots. A disposable cursor still prevents needless live rereads.

### 9.2 Chunk and manifest contract

Use a versioned FastCDC profile named `fastcdc-v1` with a checked-in fixed gear table, minimum chunk size 256 KiB, target average 1 MiB, and maximum 4 MiB. A chunk id is SHA-256 of exact chunk bytes. A manifest id is SHA-256 of canonical CBOR or canonical JSON containing chunker version, total length, and the ordered sequence of `(chunk_hash, chunk_length)`. The existing provenance-bearing raw observation id remains stable even when equal bytes are observed under distinct source/native identities; `whole_sha256` is the byte-deduplication key and is verified independently after stream reconstruction. Any future raw-id identity migration must explicitly preserve observation provenance and memberships rather than collapsing equal bytes by itself.

Add additive source-tier tables equivalent to:

```text
raw_payload_manifests(manifest_id, chunker_version, total_length, whole_sha256, created_at_ms)
raw_payload_manifest_chunks(manifest_id, position, chunk_hash, chunk_length)
```

Chunk bytes use the existing blob store with a new `raw_chunk` reference type whose `ref_id` is the `manifest_id`. Extend `BLOB_REF_LIVENESS_JOIN`, orphan-census, integrity, privacy-deletion, and retirement tooling to resolve that manifest and its ordered chunk rows; a missing manifest is an integrity failure, not an unknown ref to retain forever. A raw-session row refers to its manifest during migration while legacy whole-blob refs remain readable. The exact column names may follow current source DDL conventions, but the identities and constraints above are fixed. Include a shared-chunk deletion test proving that deleting one manifest retains chunks still referenced by another and reclaims only chunks with no live manifest.

Parsers receive a `RawPayloadReader` abstraction that can `readall`, stream ordered bytes, or open a seekable logical stream. It verifies chunk length, manifest length, and whole SHA-256 before the parse is accepted. A parser never sees chunk boundaries and cannot branch on them. The eager and streaming Claude Code paths therefore keep identical normalized semantics, including the chunk-order invariant repaired in PR #3669.

### 9.3 Publication, partial writes, resume, and deduplication

Publication follows the existing blob publication reservation protocol. Write chunks atomically, reserve their publication, write manifest plus ordered refs and raw observation in one source transaction, then release reservations. Do not introduce a second lease system; the current architecture uses publication reservations and snapshot reference checks for the acquire-to-commit gap.

An incomplete append may store chunks for all observed bytes, but admission records the complete observation length and parser-safe frontier separately. The ops cursor advances only through the last complete provider record. On resume, the chunker rereads enough overlap to restore rolling state, validates the existing prefix against manifest bytes, reuses equal chunks, and publishes a new complete observation manifest. If the prefix differs, it records a new full observation rather than trusting the cursor.

Deduplication has three layers:

- Equal whole observation bytes share the `whole_sha256` dedup key, but do not automatically share the provenance-bearing `raw_id`; distinct source/native observations remain distinct until an explicit identity rule says otherwise.
- Different observations reuse equal chunks through blob refs.
- Normalized sessions reuse source memberships only when current-version canonical hashes agree.

CDC reduces duplicated byte storage. It does not by itself reduce `raw_sessions` row count, parser CPU, or normalized revisions. Incremental parser reuse is allowed only for an OriginSpec lifecycle whose parser declares append-safe semantics and proves equivalence with replay from reconstructed complete bytes.

### 9.4 Migration and retirement

Roll out in four phases:

1. Add manifest/chunk schema and reader fallback behind a verified source backup.
2. Dual-write whole blob and manifest for new observations; shadow-read both and assert byte equality, parser equality, and raw-id equality.
3. Make manifest the primary reader with whole-blob fallback, then stop new whole-blob publication after sustained equivalence.
4. Separately request operator authorization to reclaim legacy whole blobs and copy-forward/remove durable `revision_kind` or append-offset columns after every replay, retention, repair, and governance consumer has migrated.

The live 4,344 append offsets prove that immediate column deletion is unsafe. Retain `revision_kind` and `append_end_offset` during rollout as revision provenance. Their final disposition is deletion only after CDC manifests and current/previous observation relationships express the needed facts. A cheap interim policy that discards older raw observations was rejected because it would destroy replay evidence before CDC proves replacement.

## 10. Independent follow-ups

### 10.1 `polylogue-qj5x`: remove normalized Beads issue sessions

Decision: remove `Origin.BEADS_ISSUE`, its provider/session parser admission, watcher/config path, and public origin surface. Preserve Beads information as work evidence through `insights/work_effects.py` and the mandate/claim graph. Rich `issues.jsonl` work-graph import belongs to `polylogue-5jnq` and must not be rebuilt as synthetic chat sessions.

The current `interactions.jsonl` parser turns field-change protocol into English user messages with `RUNTIME_PROTOCOL`; it is neither authored conversation nor a faithful issue model. Keeping it was rejected. Improving the synthetic prose was rejected because it leaves the wrong domain boundary.

This is targeted follow-up rather than a gate because the live source and index archives contain zero Beads rows. Before implementation, census every configured archive and raw artifact root. `Origin` values are embedded in durable source-tier CHECK constraints, so “zero index rows” is not enough to call removal index-only.

If every durable archive has zero Beads rows, take a verified backup, obtain explicit operator consent, perform a copy-forward source migration that narrows checks, delete source admission and generated surfaces, and rebuild the derived index. If any rows exist, first transform their raw evidence into work-effect/claim facts while preserving exact bytes and provenance, verify counts and references, then request consent for the same copy-forward. The projected 924 sessions quantify possible input, not an authorization to delete it.

### 10.2 `polylogue-a7xr.24`: DDL, mapper, and field reconciliation

Coupling verdict: independent and ordered after semantic index changes because it shares DDL and mapper files. It does not define content, session identity, lineage, or raw authority.

Execution packet: extend the existing `TableColumnSpec` mechanism to the sessions, attachments, and session-event families where column order, select aliases, insert bind order, and hydration are mechanical twins. Generate checked-in runtime record mappings only where static typing remains legible. Keep semantic facts, lifecycle classification, hash membership, relationship constraints, and domain models explicit. A real new column still requires the appropriate schema-version declaration; adding it to a spec cannot authorize schema evolution.

Proof uses production DDL, insert, select, and hydration round trips with null and generated columns. Mutating column order or omitting a mapped field must fail. Do not add a grep test that forbids an old spelling or merely asserts that two lists share text.

### 10.3 `polylogue-83u`: attachment integrity

Coupling verdict: semantically independent with one shared storage invariant. Attachment metadata and acquired byte hashes already enter the canonical session hash. CDC may change how those bytes are stored but must not change attachment byte identity, acquisition status, blob-reference liveness, or session hash.

Execution packet: complete the provider/artifact census, close remaining forward-capture gaps, verify attachment acquisition and reacquisition through production parsers and blob readers, and measure nullable/failed/verified states. This work can run parallel to index semantics while avoiding CDC blob DDL. Any compression, GC, or blob-ref-type change rebases after CDC and verifies both `raw_chunk` and attachment refs under snapshot GC. Mutating an attachment byte must change its byte hash and session-content-v2; moving or recompressing identical bytes must change neither.

## 11. Schema classification, rebuild, rollback, and safety

| Proposed change | Tier | Regime | Procedure |
| --- | --- | --- | --- |
| Hash-v2 parsed fields, authoritative title cleanup, event references, compaction boundaries, composition-parent unique index | `index.db` | Derived, semantic | Canonical DDL edit, version bump declared `SEMANTIC_REPARSE`, new generation from accepted raw. |
| `normalized_content_hash_version` | `source.db` | Durable additive | Numbered SQL migration, one `user_version` step, verified backup manifest. |
| CDC manifests/chunk refs and optional raw manifest reference | `source.db` | Durable additive | Numbered migrations, verified backup, dual-write and shadow-read rollout. |
| Removal of legacy raw columns or whole blobs | `source.db` | Durable destructive | Separate copy-forward design and explicit operator authorization after cutover proof. |
| Beads-origin CHECK narrowing/removal | `source.db` | Durable destructive | All-archive census, preservation transform if needed, verified backup, explicit authorization, copy-forward. |
| Context delivery seed and successor relation | `user.db` | Durable additive, irreplaceable | Numbered additive migration and verified backup. Never rebuild user evidence. |
| MCP call log or ingest cursor changes | `ops.db` | Disposable | Recreate or additive local change; no evidence authority assigned. |
| A7xr.24 mapping consolidation | Usually none | Mechanical | No bump unless the packet also changes a real schema field. |
| Attachment parsing/integrity fixes | `source.db` evidence plus derived index | Additive or semantic by exact change | Preserve blobs durably; reparse affected origins for normalized hash changes. |

Each semantic PR declares its own index delta; implementations must not reuse a version number across independently mergeable PRs. Operationally, batch the merged semantic deltas into one live archive rebuild for `polylogue-818fy`.

Safe live procedure:

1. Verify and back up `source.db` and `user.db`; validate backup manifests before migration.
2. Pause the single writer, or put the daemon into an acquire-only mode that cannot write mixed-version normalized memberships.
3. Apply additive durable migrations. Do not apply destructive copy-forward follow-ups without separate authorization.
4. Build a new derived index generation from accepted raw bytes, computing only hash v2 and applying all current OriginSpec rules.
5. Validate row counts, per-origin coverage, hash versions, event retention/ref integrity, lineage completeness, physical/logical grains, FTS, and insights.
6. Atomically promote the new generation, resume the writer, and retain the previous derived generation through the verification window.

Rollback before promotion discards the new derived generation and must leave durable memberships unchanged. If candidate construction necessarily commits a v2 membership, the operation must either stage that write outside the authoritative source tier or, on every pre-promotion failure after the first such commit, restore/reconcile the verified source-tier backup before exposing the old active generation. Rollback after promotion switches to the retained previous index only if the writer has not committed v2 durable memberships that old code cannot interpret. Otherwise stop the writer, restore the verified source backup, and then reselect the old index generation. `user.db` is restored only for a failed user migration, never rebuilt. CDC dual-write permits reader fallback until old whole blobs are deliberately reclaimed.

Existing authorization in `polylogue-a7xr.25` covers inclusion of the event transform in the `818fy` derived rebuild. It does not authorize Beads-origin durable CHECK narrowing, raw-column deletion, whole-blob reclamation, or any other destructive durable copy-forward.

## 12. Luna implementation packets

All lanes work from isolated worktrees. They do not edit Beads. Each lane commits after its focused production-route check passes, then runs `devtools verify --quick` before handoff. The coordinator rebases at the stated points, runs `devtools verify` once for the affected merge state, and runs `devtools verify --all` plus `devtools lab policy schema-versioning` and `devtools render all --check` at the terminal merge-train boundary. Tests below are selectors, not permission to run whole unit directories.

### Packet A: OriginSpec, title authority, and hash v2

**Own:** `polylogue/sources/origin_specs.py`, `polylogue/pipeline/ids.py`, `polylogue/core/hashing.py`, title-policy portions of parser assembly, `polylogue/insights/session_label.py`, source membership hash-version migration and its narrow writer/readers, `tests/unit/sources/test_origin_specs.py`, `tests/unit/pipeline/test_pipeline_ids.py`, `tests/unit/pipeline/test_content_hash_determinism.py`, `tests/unit/insights/test_session_label.py`, `tests/unit/archive/test_codex_title_census.py`, and focused durable-migration tests.

**Avoid:** `archive_tiers/index.py`, lineage functions in `write.py`, event storage, CDC blob tables, attachment GC, and context-delivery schema.

**Implement:** the OriginSpec contract fields needed by downstream lanes; authoritative-title normalization; display projection and collision suffix; session-content-v2 payload; durable normalized-hash version marker. Parser-specific material rules may be declared here, but parser behavior belongs to Packet B.

**Verification:** `devtools test tests/unit/sources/test_origin_specs.py`; `devtools test tests/unit/pipeline/test_pipeline_ids.py`; `devtools test tests/unit/pipeline/test_content_hash_determinism.py`; `devtools test tests/unit/insights/test_session_label.py`; the focused durable-migration node added for hash versioning.

**Anti-vacuity:** production parser output with a real provider title must hash differently when that title evidence changes; changing only the projected display label must preserve the hash. Removing `material_origin` from the v2 payload must fail. Equal v1/v2 payload content must still produce version-distinct hashes. A collision corpus must return unique display labels while leaving stored rows untouched.

**Commit cadence:** commit OriginSpec/title projection after its focused checks, then commit hash v2 plus durable migration after migration and pipeline checks. Packet B rebases after the first commit; all storage packets rebase after the second.

### Packet B: parser material origin and lineage evidence

**Own:** `polylogue/sources/parsers/base_models.py`, `base_support.py`, `codex.py`, Claude parser files, Hermes lineage/material files, other parser-local evidence adapters, and focused parser/Origin regression fixtures such as `tests/unit/sources/test_parsers_codex.py`, `test_codex_event_stream_contract.py`, `test_claude_code_normalization_laws.py`, `test_compaction.py`, and `tests/unit/sources/parsers/test_origin_regression_pack.py`.

**Avoid:** all SQLite DDL/write files, `pipeline/ids.py`, source migrations, context delivery, and display labels.

**Implement:** positive-evidence material-origin policy, `BranchType.RESUME`, the `ParsedLineageEdge` tuple with at most one composition candidate, generic Codex parent handling, provider compaction boundary facts, and declared event dispositions in parsed output. Do not change physical prefix storage.

**Verification:** focused parser files named above plus `devtools test tests/unit/sources/test_source_laws.py` when generated conformance affects it.

**Anti-vacuity:** remove each provider marker from a real fixture and assert the output becomes `UNKNOWN` or generic branch, not the stronger class. Feed the parsed object to production hashing and prove a material-origin change changes v2. An ambiguous second Codex `session_meta` must never yield continuation. A compaction-looking message without the provider marker must not yield a boundary.

**Commit cadence:** one commit for material-origin evidence, one for lineage/compaction evidence. Rebase on Packet A's OriginSpec commit before coding and hand off before Packet C begins final integration.

### Packet C: lineage storage, composition, and compaction reads

**Own:** lineage portions of `polylogue/storage/sqlite/archive_tiers/index.py` and `write.py`, session-link and transcript query modules, compaction derived readers/models, `tests/unit/storage/test_lineage_normalization.py`, `test_session_topology.py`, `test_topology_cycle_quarantine_live.py`, `test_bulk_fts_prefix_reextract.py`, `test_incremental_rebuild_equivalence.py`, and related property state-machine nodes.

**Avoid:** parser source files, event writer/query functions until Packet D, `pipeline/ids.py`, source/user DDL, and display labels.

**Implement:** one composition-parent partial unique index, resume mapping, topology/inheritance separation, exact-prefix behavior under hash-v2 equality, compaction-boundary DDL and transcript/effective-context reads, explicit physical/logical/effective grains, and current doc correction for the production resolver.

**Verification:** focused lineage tests above; `devtools test tests/property/test_write_path_state_machine.py`; `devtools test tests/unit/devtools/test_lineage_validation.py` if the validator contract changes.

**Anti-vacuity:** production write/read composition must fail if prefix extraction is removed, if a second inheritance edge is allowed, or if the cycle edge is composed. The same fixture must produce different transcript and effective-context views after a real compaction boundary. Interior duplicate content must remain present.

**Commit cadence:** commit schema plus core composition after focused storage tests, then commit compaction read projection after its behavior tests. Packet D rebases after both commits.

### Packet D: session-event transformation

**Own:** session-event columns and indexes in `archive_tiers/index.py`, `_write_session_events()` and event remapping in `write.py`, `polylogue/storage/sqlite/queries/session_events.py`, event runtime records and public adapters, directly affected phase/workflow/Hermes consumers, and focused tests discovered by `SessionEventRecord` and event-type references.

**Avoid:** lineage algorithms outside the ref-remap hook, parsers except minimal typed event-contract adapters agreed with Packet B, source/user DDL, CDC, and attachment code.

**Implement:** typed content refs, event-only payloads per OriginSpec disposition, lossless unknown fallback, consumer migration, and event-ref integrity validation. Bump the derived schema as a semantic reparse delta.

**Verification:** `devtools test tests/unit/pipeline/test_archive_write.py`; focused nodes in `tests/unit/storage/test_archive_tiers_write.py`; `devtools test tests/unit/archive/test_phase_extraction.py`; affected sidecar/Hermes/workflow tests selected by testmon.

**Anti-vacuity:** parse and write actual Codex tool and reasoning records. Assert content is absent from event JSON but present through the exact referenced block. Delete or mis-remap that block and require validation/read failure. Mutate Claude sidecar-only evidence and assert it survives inline. Unknown event payload must round-trip byte-semantically.

**Commit cadence:** one schema/write commit, then one consumer migration commit. Rebase on Packet C immediately before starting because both own `index.py` and `write.py`.

### Packet E: context-assisted continuation evidence

**Own:** numbered `user.db` migration for context deliveries, context delivery writer/reader models, successor-association operation, joined continuation projection, MCP/API contracts for that projection, `tests/unit/mcp/test_context_resume_intent.py`, `tests/unit/mcp/test_mcp_call_log.py`, and focused durable user-state tests.

**Avoid:** `session_links` write path, parser branch types, hash code, index DDL, and ops log semantics except read-only joining.

**Implement:** seed session ref, append-only successor association with evidence, four-state joined view, explicit association actuator. Do not infer topology from time or tool name.

**Verification:** the two MCP tests, focused `tests/unit/storage/test_archive_tiers_user_write.py` and `test_durable_migrations.py` nodes, and affected context view tests.

**Anti-vacuity:** a real durable delivery plus explicit successor association yields context-assisted continuation; deleting the association yields prepared/delivered without successor. An adjacent MCP call alone never changes the state. Restart with an empty `ops.db` and preserve the durable result.

**Commit cadence:** migration and durable CRUD first, then joined surface. This packet can run parallel to C and D and rebases before final surface integration.

### Packet F: CDC raw byte authority

**Own:** source-tier DDL and numbered migrations for manifests/chunks, blob-ref type and publication logic, raw payload reader, acquisition writers/readers, append/cursor adapters, and focused tests including `tests/unit/pipeline/test_blob_publication_crash_matrix.py`, `test_ingest_append_replay.py`, `tests/unit/sources/test_cursor_lifecycle.py`, `test_live_append_cursor_resynthesis.py`, `tests/unit/storage/test_blob_integrity.py`, `test_blob_gc_raw_authority_verdict_invariant.py`, `test_raw_revision_authority.py`, and durable migration tests.

**Avoid:** `pipeline/ids.py` normalized-session payload, index/session/event DDL, lineage, attachment parsers, and Beads-origin removal.

**Implement:** FastCDC-v1 constants, manifest identity, raw chunk refs, verified logical reader, publication atomicity, dual-write and shadow-read, safe cursor resume, and metrics required for cutover. Do not delete legacy columns or blobs in this packet.

**Verification:** focused crash-matrix, cursor, raw-authority, blob-integrity, and migration tests; a bounded synthetic multi-append benchmark comparing stored bytes and parse output.

**Anti-vacuity:** production acquisition of two observations with a long equal prefix must store fewer unique chunk bytes than two whole blobs while reconstructing identical whole SHA-256 values. Changing one byte must change raw id and only the affected chunk neighborhood. Removing `ops.db` must still replay exactly. A crash at every publication boundary must leave either no admitted manifest or a fully readable one.

**Commit cadence:** schema/identity commit, reader/dual-write commit, cursor/resume commit. Attachment blob work rebases only after these land.

### Packet G: targeted Beads-origin removal

**Own:** the census tool or read-only report, Beads origin/provider enums and source config, `polylogue/sources/parsers/beads.py`, origin-spec/dispatch registration, generated surfaces, source copy-forward migration after authorization, and `tests/unit/sources/parsers/test_beads.py` plus origin/schema policy tests.

**Avoid:** work-effect and mandate graph semantics except preservation adapters, general content hash, lineage, CDC implementation, and any durable delete before explicit authorization.

**Implement:** census first. Stop after the report if any durable archive contains Beads rows or operator consent is absent. After authorization, preserve work evidence, narrow durable checks by copy-forward, delete normalized-session admission, and regenerate public surfaces.

**Verification:** exact pre/post row and raw-byte counts; work-evidence preservation query when rows exist; durable backup restore rehearsal; focused parser deletion fallout, origin-spec conformance, `devtools lab policy schema-versioning`, and render check.

**Anti-vacuity:** before removal a representative `interactions.jsonl` routes to the Beads parser; after removal it is rejected as a session source while its intended work-effect path remains queryable. Restoring the pre-migration backup must recover every raw byte.

**Commit cadence:** census/report commit if a durable artifact is warranted; implementation only after authorization. Rebase after Packet A and Packet F source migrations. It may join the final index rebuild but cannot delay it.

### Packet H: spec-driven DDL reconciliation

**Own:** `polylogue/storage/sqlite/archive_tiers/column_spec.py`, `archive_tiers_specs.py`, the mechanical spec-driven portions of `index.py` and `write.py`, `polylogue/storage/hydrators.py`, `polylogue/storage/sqlite/queries/mappers_archive.py`, `message_query_reads.py`, `tests/unit/storage/test_column_spec_reordering.py`, and `test_spec_driven_hydration.py`.

**Avoid:** changing field meaning, hash membership, lineage rules, raw authority, or schema versions without a separately justified semantic change.

**Implement and verify:** follow section 10.2. Start only after Packets C and D have settled `index.py` and `write.py`. Use the owned spec tests plus `tests/unit/pipeline/test_roundtrip_hydration_laws.py`, `tests/unit/storage/test_archive_tiers_ddl.py`, `test_prepared_session_rows.py`, and `test_query_mappers.py`. Mutating a column mapping must break a production round trip.

### Packet I: attachment integrity

**Own:** provider attachment acquisition and census code, attachment/blob integrity readers and tests. **Avoid:** CDC source DDL until Packet F lands, lineage, event slimming, and display/hash policy except asserting existing v2 attachment behavior.

**Implement and verify:** follow section 10.3 using `tests/unit/storage/test_attachment_acquisition.py`, `test_attachment_reacquisition.py`, `test_blob_integrity.py`, `tests/unit/sources/test_drive_attachment_fetch.py`, and `tests/unit/security/test_attachment_security.py`. Commit census/forward-capture independently; rebase compression/GC work after Packet F.

## 13. Acceptance matrix

Legend: **Existing** means current production code already satisfies the criterion; **Implement** means the train must change code; **Evidence** means a measurement or proof is required before closure; **Follow-up** means intentionally outside the architecture gate with a named closure path.

| Bead | Acceptance area | Status | Closure evidence |
| --- | --- | --- | --- |
| `a7xr.25` | Inventory 7.4M event rows and real consumers | Existing plus Evidence complete | Live counts and consumer census in sections 1 and 2. |
| `a7xr.25` | Remove duplicate tool/reasoning content without losing protocol meaning | Implement | OriginSpec disposition matrix, typed refs, production parser/write/read anti-vacuity tests. |
| `a7xr.25` | Replay, compatibility, retention, query behavior | Implement | Semantic index rebuild, consumer migration, unknown inline fallback, ref integrity validation. |
| `6e7m` | Separate display title from hash identity | Implement | Stored-title census after replay, hash invariance under display change, collision-free read projection. |
| `6e7m` | Deterministic null/empty/provider fallbacks | Implement | Cross-origin title conformance fixtures. |
| `6e7m` | Honest historical idempotency | Implement | Hash-v2 versioned membership migration and full replay. |
| `4ts` | Persist parent references before parent exists | Existing | Current `session_links` write path and child-first tests. |
| `4ts` | Resolve, cycle-handle, and recompose exact prefixes | Existing plus Implement | Existing engine remains; add one-parent schema invariant and explicit grain contracts. |
| `4ts` | Fork/resume/subagent/compaction distinctions | Implement | Parser positive-evidence tests and separate compaction-boundary relation. |
| `4ts` | Reduce duplicate blocks safely | Existing for exact prefixes; Evidence required for remaining 26.5 percent | Lineage-linked versus unrelated duplicate report. No global suppression before it. |
| `nas1` | Provider-native resume decision | Implement | Exact `BranchType.RESUME` only where wire evidence exists; ambiguous Codex stays generic. |
| `nas1` | Context-assisted continuation decision | Implement | Durable delivery seed/successor graph and joined four-state read model. |
| `nas1` | Avoid telemetry/topology conflation | Implement | Empty-ops restart and no-time-heuristic tests. |
| `2qx` | Single origin admission contract | Existing partially; Implement remaining fields | OriginSpec conformance drives detection, title, material, lineage, event, provenance, fidelity, and coverage. |
| `2qx` | Close live material-origin gaps | Implement | Positive-marker mutation pack across every admitted origin. |
| `2qx` | Positive Codex continuation evidence | Implement | Second-meta fixture cannot assert continuation without declared marker. |
| `2qx` | Fidelity/provenance/coverage reporting | Implement | Generated per-origin matrix plus real fixture round trips and unread/degraded-field report. |
| `qj5x` | Decide Beads origin | Decision complete: remove | Section 10.1. |
| `qj5x` | Determine blast radius | Evidence required | All configured source/index archives and artifact roots, not current live archive alone. |
| `qj5x` | Preserve existing evidence and migrate safely | Follow-up | `polylogue-5jnq` work graph, preservation transform if rows exist, authorized durable copy-forward. |
| `a7xr.23` | Decide CDC versus cursor | Decision complete | CDC durable authority plus disposable ops cursor. |
| `a7xr.23` | Chunk identity, partial write, resume, dedup | Implement | FastCDC-v1 manifests, publication crash matrix, byte equality, ops-loss replay proof. |
| `a7xr.23` | Parser replay boundary | Implement | `RawPayloadReader` hides chunks and proves eager/streaming equivalence. |
| `a7xr.23` | Retire legacy snapshot fields | Follow-up | Only after dual-write cutover, consumer census, and destructive authorization. |
| `a7xr.24` | Reconcile DDL/record/mapper/hydrator | Existing for messages/blocks; Implement remaining mechanical families | Production round-trip tests, ordered after semantic DDL. |
| `a7xr.24` | Identity/lineage coupling | Decision complete: independent | No semantic fields delegated to mechanical generation. |
| `83u` | Attachment integrity coupling | Existing invariant plus independent work | Attachment byte hash remains session hash input; CDC preserves refs. |
| `83u` | Remaining provider coverage and storage work | Follow-up packet | Provider census and production acquisition/reacquisition tests; GC rebase after CDC. |

## 14. Evidence commands and unresolved operator decisions

The design evidence was gathered with read-only Beads inspection, `rg`, `git log`, `git show`, focused source/test reads, and SQLite read-only queries. Key reproducible commands include:

```text
bd --readonly show polylogue-a7xr.25 polylogue-6e7m polylogue-4ts polylogue-nas1 polylogue-2qx polylogue-qj5x polylogue-a7xr.23 polylogue-a7xr.24 polylogue-83u
git log --all --oneline --decorate -- polylogue/pipeline/ids.py polylogue/storage/sqlite/archive_tiers/write.py polylogue/sources/origin_specs.py
sqlite3 -readonly /realm/db/polylogue/index.db 'SELECT count(*) FROM sessions; SELECT count(*) FROM messages; SELECT count(*) FROM blocks; SELECT count(*) FROM session_events; SELECT count(*) FROM session_links;'
sqlite3 -readonly /realm/db/polylogue/source.db 'SELECT revision_kind, count(*), round(sum(size_bytes)/1048576.0,1), sum(append_end_offset IS NOT NULL) FROM raw_sessions GROUP BY revision_kind;'
sqlite3 -readonly /realm/db/polylogue/index.db '<bounded duplicate-block query from section 2>'
```

No design choice remains open for implementation. The following operations require later operator authorization because they are destructive changes to durable evidence:

1. Narrowing durable source-tier Origin checks and removing Beads-origin rows or admission after the all-archive census.
2. Copy-forward removal of `revision_kind`, `append_end_offset`, or other durable raw columns after CDC cutover.
3. Reclaiming legacy whole raw blobs after dual-write and restore proof.

The already recorded decision to include `a7xr.25` in `polylogue-818fy` authorizes the derived semantic rebuild slice. It does not imply any of the three durable destructive authorizations above.
