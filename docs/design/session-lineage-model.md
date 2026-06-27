# Session Lineage Model — forks, resumes, sidechains, subagents, compaction

Status: **draft / proposed** (design before implementation)
Tracking: [#2467](https://github.com/Sinity/polylogue/issues/2467)
Author evidence: measured against the live 37 GB / 16 K-session archive, 2026-06-27.

## Why this document exists

Polylogue ingests every physical session artifact (one rollout/transcript file =
one physical session) as an independent session with its own full message tree.
But agentic tooling routinely produces **multiple physical artifacts that share
content**: a resumed Codex thread replays its parent, a parallel agent fan-out
forks one context into N rollouts, Claude auto-compaction re-emits a whole
conversation to summarize it. When the shared content is stored and counted once
per physical artifact, the archive **misrepresents its own contents**:

- headline session count is inflated (16,414 physical vs 8,807 logical roots, 1.86×);
- ~32 % of message rows are duplicate-bearing (Codex fork replays);
- ~42 % of block texts are non-unique (includes legitimate recurrence, but a
  large slice is fork replay);
- cost/token totals over-count (Codex 189.7 B physical vs 139.3 B authoritative
  `state_5.sqlite`, a ~50 B / 1.36× inflation entirely inside fork families);
- FTS indexes the same content many times → duplicate hits, skewed ranking;
- embeddings (when enabled) would pay to embed the same content many times.

This is a correctness problem in the **data model**, not a reporting cosmetic.
The lineage is already *detected* (`session_profiles.logical_session_id` groups
fork families correctly); the failure is that **storage and nearly every
aggregate ignore it**.

## Taxonomy of session relationships (measured)

| Relationship | Origin | Real count | Content sharing | Correct accounting |
|---|---|---|---|---|
| **Linear resume** | Codex (33 families) | child ⊇ parent prefix | parent's prefix replayed in child | count shared content **once** |
| **Parallel fork / fan-out** | Codex (42 families, ≤35 siblings) | all siblings share a common prefix | prefix replayed in every sibling | count shared prefix **once**, each branch's unique tail once |
| **Subagent (Task)** | Claude (7,383) | ~99 % unique work | only the task prompt/result echoes parent | count **fully** — real distinct work |
| **Sidechain** | Claude (131) | embedded sub-thread | likely overlaps parent (to verify) | dedupe overlap |
| **Auto-compaction** | Claude `agent-acompact-*` (187) | 100 % of parent | whole conversation re-emitted to summarize | **not a session** — derived view; exclude from counts/cost/search |
| **Main / root** | all (8,427 `branch_type=null`) | unique | — | count |

Key correction to an earlier hasty read: **real Task subagents are not
duplication.** They are genuine distinct work and must be counted. The
duplication is narrow and concentrated: Codex resume+fork families and Claude
auto-compaction.

## Where the current model breaks (code-confirmed)

Ingestion (`pipeline/ids.py`, `storage/sqlite/archive_tiers/write.py`): **no
cross-session content dedup.** Each physical session writes its full message and
block rows keyed by `(session_id, …)`. `content_hash` is per-session (folds in
session id/position/native id), so identical text across forks gets different
hashes and cannot dedupe. The byte-level blob store dedupes large tool-output
*bytes*, but the message/block *rows* are duplicated.

Aggregates — physical (over-count) vs logical (correct):

- **PHYSICAL / over-count:** `cost_compute`, `session_model_usage` aggregation,
  `stats.get_stats_by`, `aggregate_message_stats`, `list_summaries`,
  `fts5`/`hybrid` search, `embeddings`, `aggregate_cost_rollup_insights`,
  per-session work-events/phases.
- **ALREADY LOGICAL:** `session_tag_rollups` (dedupes via `logical_session_ids`),
  `get_logical_session` (structure), `ThreadInsight` (root-keyed).

Parser misclassification: `claude/code_parser.py` keys subagent detection on
`fallback_id.startswith("agent-")`, which captures `agent-acompact-*`
auto-compaction and labels it `branch_type=subagent`. Compaction is a derived
re-summarization of the parent, not an independent session.

## The correct model

**Content is the atom.** Messages/blocks should be content-addressed and stored
once; a physical session is an *ordered view* (a sequence of references) over
shared content; a logical session is the **union DAG** of content across its
fork family. Every aggregate counts content **once per logical session**.

This is a git-like model: blocks ≈ blobs (content-addressed), messages ≈ tree
entries, physical sessions ≈ refs/commits, fork families ≈ branches sharing
history. It makes the duplication impossible by construction rather than
correcting for it in every consumer.

Token/cost attribution under this model: a turn's token usage is attributed to
the **canonical (earliest) physical session** that produced that content;
replays in later forks/resumes are references, not new consumption — except
where a fork genuinely *re-ran* the model over the shared prefix (real cache-read
consumption). The cumulative-vs-delta and cached-in-input subtleties already
handled in `_provider_usage_disjoint_lanes` compose with this: dedupe first by
content lineage, then price the deduped lanes.

## Phased plan (proposed)

**Phase 0 — stop the misclassification (small, clearly correct).**
Classify `agent-acompact-*` as compaction, not subagent. Decide its kind
(`compaction` branch/edge type or `metadata_document` artifact) and exclude it
from session counts/cost/search. Re-ingest or backfill the 187 rows.

**Phase 1 — make aggregates lineage-aware (read-time correctness, no schema
change).** Route cost, stats/counts, FTS result dedup, and embedding selection
through a logical-session/content-dedup view. Establishes a single
`logical_session` accounting helper that all surfaces consume. Validates against
authoritative stores (`state_5.sqlite`, `stats-cache.json`). Fixes the *numbers*
without re-ingest; storage stays duplicated.

**Phase 2 — structural content-addressing (the "unlikely by construction"
fix).** Content-address blocks/messages so shared prefixes are stored once and
forks reference them. Schema change + re-ingest. Eliminates storage bloat and
makes every current and future aggregate correct at the source. Large; gated on
Phase 1 evidence and an explicit re-ingest plan (see CONTRIBUTING "Schema-Touching
Changes").

## Open decisions (need operator input)

1. **Default reporting unit.** Should headline session count / cost / search
   default to **logical** sessions (subagents and forks folded into their work
   session) or **physical**, with the other available via a flag? Recommendation:
   logical as the default headline, physical available explicitly.
2. **Phase 2 commitment.** Is the structural content-addressed overhaul in scope,
   or do we stop at Phase 1 read-time correctness for now?
3. **Re-ingest tolerance.** Phases 0 and 2 want a fresh re-ingest of the real
   archive; acceptable, and a chance to dogfood ingest throughput.

## Verification anchors (already established)

- Codex per-thread vs `state_5.sqlite`: median ratio 1.000 after the disjoint-lane
  fix; aggregate excess = fork replay, not pruned history.
- Claude per-model vs `stats-cache.json`: 1.09× (stale 6-day window + minor
  model-attribution drift), no resume duplication.
- Fork families confirmed as real on-disk rollouts (parallel fan-out, e.g.
  parent `019d1c2a` + 20 children created within ~45 s), 100 % block-text overlap
  with parent, ~30 unique blocks each.

## Remaining verification before implementation

- Sidechain (131) content overlap with parent — dedupe needed?
- Whether Codex fork rollouts re-emit parent `token_count` events (determines
  whether per-turn delta summation also double counts).
- Completeness of `logical_session_id` for out-of-order / missing-parent forks
  (`topology_edges` #1258 resolution).
