# Session Lineage Model — one representation for forks, resumes, compaction, subagents

Status: **draft / proposed** (design before implementation)
Tracking: [#2467](https://github.com/Sinity/polylogue/issues/2467) (lineage/dedup),
[#2468](https://github.com/Sinity/polylogue/issues/2468) (attachment bytes).
Evidence: measured against the live 37 GB / 16 K-session archive and the Codex
(`/realm/project/_inactive/codex`) and Claude Code sources, 2026-06-28.

## The problem

Providers encode the *same* structural facts in incompatible, often denormalized
ways. If polylogue faithfully ingests each encoding as-is, it both **duplicates
content** and **misrepresents context**:

- 16,414 physical sessions vs 8,807 logical roots (1.86×); ~32 % of message rows
  are duplicate-bearing; Codex token total 189.7 B physical vs 139.3 B
  authoritative (`state_5.sqlite`), the excess being fork replays.

The fix is **not** to mirror each provider's quirks. It is to recognize that
behind the encodings there are only **two structural concepts**, model those once
internally, and derive every provider-agnostic view from that model. The hard
constraint: **unify without lying about content** — every real message is
preserved exactly once; nothing the model actually saw or the user actually wrote
is dropped or fabricated.

## The two structural concepts

### 1. Prefix inheritance (fork / resume / continuation)

A session begins from another session's content up to some point, then diverges.
Its effective context = *parent's content through the branch point* + *its own*.

- **Codex** (fork / subagent `thread_spawn` / resume): **denormalized** — the
  parent's full history is *copied* into the child rollout (verified in
  `thread_manager.rs::spawn_subagent` → `stored_thread_to_initial_history`; the
  child rollout physically replays ~62 K inherited messages with fresh
  ids/timestamps, plus ~30 of its own). `session_meta.forked_from_id` marks the
  parent explicitly.
- **Claude Code** (resume): **normalized** — threads via `uuid`/`parentUuid`,
  no copy across files. Claude already does the right thing.

Same concept, opposite encodings. We model it **one way**: the child has a typed
lineage edge to the parent at a branch point and stores **only its own
(post-branch) messages**; the inherited prefix is *referenced*, never copied.

### 2. Context discontinuity (compaction)

At a point in a session the prior context is evicted and replaced, for all
subsequent turns, by a **summary**. From that point the model sees
`[summary, later…]`, not the full prefix. The pre-compaction messages still
*happened* (they stay in the archive) — they are simply no longer in context.

- **Codex**: explicit — a `compacted` rollout record carries `message` (the
  summary) and `replacement_history` (the post-compaction context =
  kept user messages + summary). Verified in `compact.rs`.
- **Claude Code inline**: a `type=user`, `isCompactSummary: true` record
  ("This session is being continued from a previous conversation that ran out of
  context…") inserted in the same session; prior messages retained.
- **Claude Code auto-compact agent** (`agent-acompact-*`): an overloaded
  artifact family. Roughly 148/187 observed files replay the main-session
  transcript and add a small summary tail; ~39 are Task-subagent
  self-compactions with less than 90% parent membership (9 observed at 0%).

Same concept, three encodings. We model a true main-session auto-compact as a
compaction boundary at point N with a summary message S: messages before N stay
stored once; effective context after N = `[S] + post-N`. A Task self-compaction
under the same filename prefix is instead a fresh sidechain: its messages stay
whole and its parent edge is `spawned-fresh`, so composition never prepends the
main transcript.

ChatGPT / Gemini / Antigravity have neither concept (confirmed: no compaction or
fork machinery in their parsers).

## The model: store content once, structure as edges, derive context

- **Messages are stored exactly once**, in the session where they originated.
- **Copies are edges, not rows.** A copy-bearing session (Codex fork/resume,
  Claude acompact) stores a typed lineage edge to its parent + only its unique
  tail. The copied prefix is never materialized. (The raw artifact with the full
  copy is still preserved as a blob in `source.db` — index.db is derived, so
  normalizing it loses nothing recoverable.)
- **The compaction summary is a real message**, stored once at the boundary. It
  is not flagged "derived/fake" — semantically it *is* the message that replaces
  the prefix in context. Its meaning is structural (where it sits in the
  context-lineage), carried by the compaction boundary, not by a content type.
- **Fresh spawns stay whole.** A Task subagent that starts from a fresh context
  (Claude `agent-<hex>`, ~99 % unique content) is real distinct work: stored
  fully, with a `spawned-fresh` edge to its parent (relationship without prefix
  sharing).
- **Effective context is derived**, never stored redundantly. "What the model
  saw at turn T" = walk inheritance edges + apply compaction boundaries over the
  once-stored messages. Aggregates (counts, cost, search, embeddings) count each
  real message once; copies contribute nothing (they are edges); the summary
  counts once (it is a real message with real generation cost).

Physical artifacts remain first-class and recoverable (raw blobs in `source.db`);
we are normalizing the *derived index*, not erasing sessions. A "logical session"
is just the composed view across edges, not a replacement for the physical ones.

## Ingest

1. **Detect the relationship from explicit markers**, never from content alone:
   `forked_from_id` / `thread_spawn` (Codex fork), Codex `compacted` record,
   Claude `isCompactSummary`, `agent-acompact-*` id, sidechain/subagent markers.
   The marker fixes the *type* (copy-with-prefix vs fresh-spawn vs compaction).
2. **Extract the unique tail for copy-bearing sessions** by content-diff against
   the *known* parent — scoped to the lineage, so coincidentally-identical text
   in unrelated sessions never collides. Use **conservative contiguous
   prefix-alignment**: classify a message as inherited only inside the matching
   prefix run, never by isolated equality, so a genuinely-new block that happens
   to equal a parent block is never dropped (no content loss).
3. **Defer resolution for out-of-order ingest** (child before parent) using the
   existing late-resolution machinery already used for `parent_session_id` /
   `session_links` (#1258/#1259). Compaction needs no parent lookup (the boundary
   is in-band).

## Reads and aggregates

- Single composition point: `get_messages(conn, session_id)`
  (`storage/sqlite/queries/message_query_reads.py`) resolves the inheritance edge
  and composes `parent prefix + own tail` for a full transcript. The same point
  serves Claude resume chains, which need composition anyway and don't get it
  today (Claude `continuation` count is currently 0 — a pre-existing gap this
  fixes).
- Session-keyed derived tables (FTS, `session_model_usage`, work-events, phases,
  embeddings) hold only real (once-stored) content, so they are correct by
  construction instead of needing per-consumer dedup. Search over an inherited
  prefix resolves through the parent; embeddings never pay twice.
- `message_id` / position keying must accommodate "this session's messages start
  after an inherited range" — resolved at the composition layer, not by copying
  parent rows into the child's PK space.

## Attachment preservation (Ref #2468)

Same root failure (no real content store), separate fix. Attachments are
**metadata-only by construction**: 8,425 rows claim 8.4 GB but **0 blobs exist**;
56 % are zero-byte. `attachments.blob_hash` is a *synthetic metadata hash*
(`write.py:_attachment_blob_hash` → `bytes.fromhex(attachment_id)`), not bytes;
`download_assets` is accepted then `del`'d (`api/ingest.py`); parsers set
`path=None` and the real path is hashed then discarded; pasted text is
marker-only; **fork/duplicate-message attachments are silently dropped**. The
structural work must:

- store attachment bytes in the blob store keyed by **true SHA-256 of bytes**,
  ref-counted (the `attachments` table already has the right shape, just a fake
  hash);
- add an ingest **acquisition step** that fetches/reads bytes (Drive/OAuth API,
  local path, inline base64, export-zip member) and records per-attachment status
  (acquired / unavailable / unfetched) instead of fabricating a hash;
- preserve the real `path`/`source_url` for re-acquisition;
- carry attachments through lineage instead of dropping them; surface them in
  read paths and make extractable text searchable.

Recoverability is per-source (Drive via API; browser-capture chats likely need
official export zips; repo tarballs regenerable) — re-ingest classifies each.

## What we deliberately do NOT do

- **No content-addressing of messages/blocks.** Identical-but-unrelated text
  ("continue", a bare `git status`) must stay distinct events; dedup is by
  *lineage*, not content hash. (Earlier draft got this wrong.)
- **No storing copies, even flagged.** Flagged duplicates force every consumer to
  special-case them; normalization avoids that entirely.
- **No per-provider branches downstream.** Provider quirks are resolved at ingest
  into the one model; nothing past ingest knows about `forked_from_id` vs
  `isCompactSummary`.
- **No deleting physical artifacts.** Raw rollouts stay in `source.db`; the index
  is rebuildable.

## Decisions

- **Default reporting unit: logical** (composed view); physical available
  explicitly. Physical artifacts remain fully recoverable.
- **Approach: normalize copies** (store edge + tail), don't store-and-flag.
- **Re-ingest is acceptable** and is the occasion to optimize ingest throughput
  first (ties to the operator's ingest-snappiness goal and #2391).

## Phased implementation

Status as of the prefix-inheritance slices (index schema **v12**):

0. **Done** — `agent-acompact-*` uses a fresh Task-head marker or a bounded
   parent-content membership test: true main-session copies are continuations,
   while mismatched Task self-compactions are `SIDECHAIN` + `spawned-fresh`.
   Codex `forked_from_id` / `source.subagent.thread_spawn` are detected and set
   `branch_type` SUBAGENT/FORK.
1. **Done (prefix-inheritance)** — `session_links` carries
   `branch_point_message_id` + `inheritance` (`prefix-sharing` / `spawned-fresh`).
   Compaction boundaries (range columns) + attachment real-blob-hash remain for
   the compaction and attachment slices.
2. **Done (parent-known path)** — marker detection + conservative contiguous
   prefix-alignment (per-message content signature) extract the divergent tail at
   write time; only the tail + edge are stored. **Remaining:** deferred
   re-extraction when a child is ingested before its parent (the child is stored
   whole and the edge left unresolved until then); real attachment byte
   acquisition.
3. **Done (prefix-inheritance)** — both read paths compose: the async
   `get_messages` query and the sync `read_archive_session_envelope` (used by MCP
   `get_session_summary` / CLI `read`) prepend the parent transcript up to the branch
   point. Effective-context derivation lands with the compaction slice.
4. Optimize ingest, then re-ingest the real archive.
5. Validate against authoritative stores (`state_5.sqlite`, `stats-cache.json`):
   per-thread token ratio → 1.00; attachment blobs present > 0; no duplicate
   message rows across fork families.

Verified on the live corpus: Codex subagent `019ccbf9` (forked from `019ccbf8`)
shares an 18-message prefix; after normalization the parent keeps 110 rows and
the child stores its 162-message tail (was 180 with the full replay), and a
composed read reconstructs the full 180-message transcript exactly.

## Verification anchors (established)

- Codex per-thread vs `state_5.sqlite`: median ratio 1.000 after the disjoint-lane
  cost fix; aggregate excess = fork replay.
- Claude per-model vs `stats-cache.json`: 1.09× (stale window + attribution), no
  resume duplication.
- `agent-acompact-*`: 187 total in the measured corpus; roughly 148 main-session
  copies and ~39 Task self-compactions below 90% parent membership (9 at 0%).
- Fork families: real on-disk rollouts, `forked_from_id` explicit, 100 % block
  overlap with parent, ~30 unique blocks each.
