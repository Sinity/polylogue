---
created: "2026-06-28"
purpose: "Validation plan + pre-re-ingest baseline for the #2467 lineage normalization (index schema v12-v14, PR #2469)"
status: "active"
project: "polylogue"
---

# Lineage normalization (#2467 / PR #2469) — validation plan + baseline

## 2026-07-04 update — executable v24 artifact

This note is historical below. The live archive is no longer the
pre-normalization v11/v14 baseline, and the scratch SQL checklist has been
replaced by an executable devtools gate. Beads issue `polylogue-4ts.1` is the
authoritative task state for the gate; residual repair is tracked in
`polylogue-9p0y`.

Subagent research on 2026-07-03 found that the implementation is mostly in
place: prefix-tail extraction, late child re-extraction, sync/async composed
reads, cycle quarantine, and logical-session profile materialization all have
code and synthetic test coverage. The remaining gap is an executable
cold-reader artifact, not another scratch SQL checklist.

Current read-only command:

```bash
env POLYLOGUE_ARCHIVE_ROOT=/home/sinity/.local/share/polylogue \
  devtools workspace lineage-validation \
  --archive-root /home/sinity/.local/share/polylogue \
  --sample-prefix-sharing 100 \
  --max-sample-stored-messages 500 \
  --out-dir .agent/demos/lineage-validation/current \
  --json
```

Measured 2026-07-04 against `/home/sinity/.local/share/polylogue/index.db`:

- schema v24
- physical sessions: 16,635
- logical sessions: 9,517
- physical/logical ratio: 1.748x
- stored messages: 4,269,978
- session profile rows: 16,635
- missing session profiles: 0
- session links: 8,048
- prefix-sharing links: 345
- spawned-fresh links: 7,561
- unresolved/null-inheritance links: 142
- unsupported prefix-sharing origins: 0
- dangling branch points: 6
- sampled prefix-sharing composed reads: 100 rows, 1,589 stored tail messages,
  607,316 composed messages, 382.2x composed/stored ratio, 0 read errors.

Current verdict: `external_counts_citable=false` because six prefix-sharing
branch points do not resolve to message rows. The generated demo artifact lives
at `.agent/demos/lineage-validation/current/` and is intentionally small
(`README.md`, `summary.json`, `lineage-validation.report.json`).

Do not use `polylogue ops diagnostics workload` planner-estimated table counts
as the lineage evidence artifact; exact lineage/citation evidence now comes from
`devtools workspace lineage-validation`.

## What #2469 actually does (implementation read)

Source: `polylogue/storage/sqlite/archive_tiers/write.py`,
`.../index.py`, `polylogue/storage/sqlite/queries/message_query_reads.py`.

### Schema (index.py, `INDEX_SCHEMA_VERSION = 14`)
- `session_links` gains two columns:
  - `branch_point_message_id TEXT` — last *parent* message the child inherited.
  - `inheritance TEXT CHECK(inheritance IN ('prefix-sharing','spawned-fresh') OR inheritance IS NULL)`.
- `branch_point_message_id` is **plain TEXT, deliberately NOT a FK** (v14 / audit
  H1). A parent's full-replace re-ingest (`DELETE FROM messages` then re-INSERT)
  would otherwise null the child's branch point via `ON DELETE SET NULL`. The
  reference survives because `message_id` is deterministic:
  `_message_id` → `archive_message_id(session_id, native_id, position, variant_index)`
  — re-inserting the same parent reproduces the identical message_id.
- v14 also adds `idx_messages_session_sortkey` (keyset/paginated reads stop doing
  a per-chunk temp B-tree sort).

### Write-time extraction (parent already ingested) — `_extract_prefix_tail`
- Computes a per-message **content signature** = `role + ordered (block_type,
  text, tool_name, canonical_json(tool_input))`. `_composed_db_signatures`
  produces the parent's *composed* (prefix+tail, recursive) signature list.
- **Conservative contiguous prefix-alignment**: walk k forward while
  `parent_composed[k].sig == child_sigs[k]`. Stop at first mismatch.
  - `k == 0` → `(None, "spawned-fresh", all messages)` — fresh Task subagent,
    stored whole.
  - `k > 0` → `branch_point = parent_composed[k-1].message_id`,
    `inheritance = "prefix-sharing"`, store only `messages[k:]` (the tail).
- Conservative-contiguous is the no-content-loss guarantee: a genuinely-new block
  that coincidentally equals a parent block is never dropped because alignment
  only classifies inside the matching prefix run.

### Deferred extraction (child ingested before parent) — `_reextract_prefix_tail_db`
- Child is stored **whole** with edge `inheritance = NULL` until the parent
  arrives. On parent save, resolution aligns the child's *already-stored* rows
  against the parent's composed signatures, `DELETE`s the inherited-prefix
  `message_id`s, sets the edge, and calls `_refresh_session_counts`.
- Guard: only runs while `inheritance IS NULL` (idempotent re-ingest safe).

### Read composition (two mirrored paths)
- Async `get_messages` (`message_query_reads.py`) and sync
  `read_archive_session_envelope` (`write.py`).
- Both: look up the resolved prefix-sharing edge → recursively compose the parent
  transcript **in strict position order** → take the prefix up to **and
  including** `branch_point_message_id` → append this session's own tail.
- Recursion bounded by `_MAX_LINEAGE_DEPTH = 64`.
- **Dangling branch point safety**: if `branch_point_message_id` is not found in
  the parent's composed transcript (e.g. parent message hard-deleted), the reader
  **bails to the child's own tail only** — it does NOT splice the whole parent
  in (avoids an over-long transcript). This is the load-bearing safety branch.

### Invariants claimed (from tests/unit/storage/test_lineage_normalization.py)
1. Prefix-sharing child stores only its tail at original positions; `message_count`
   reflects tail only; edge has `inheritance='prefix-sharing'`, non-null branch
   point, resolved parent. Both read paths recompose the full transcript.
2. Child-before-parent: stored whole + `inheritance NULL`; after parent arrives,
   re-extracted to tail; composition still exact.
3. Parent full-replace re-ingest keeps the child composing (branch point survives,
   not a FK).
4. Spawned-fresh child (no shared prefix) stored whole; `inheritance='spawned-fresh'`,
   `branch_point_message_id IS NULL`; read does NOT prepend parent.

---

## Pre-re-ingest BASELINE (measured 2026-06-28, read-only mode=ro)

Live archive: `/home/sinity/.local/share/polylogue/index.db` (38.8 GB,
daemon-active). **`PRAGMA user_version = 11`** and `session_links` has NEITHER
`branch_point_message_id` nor `inheritance` — confirmed this is the OLD
pre-normalization (pre-v12) data. This is the correct "before" snapshot.

| Metric | Value |
|---|---|
| Physical sessions | **16,483** |
| Logical sessions (`COUNT(DISTINCT session_profiles.logical_session_id)`) | **8,809** |
| Physical/logical ratio | **1.87×** (7,674 sessions are non-root lineage children) |
| Physical messages (`COUNT(*) messages`) | **5,736,468** |

### Messages by branch_type (`SUM(message_count)` from `sessions`)
| branch_type | n_sessions | messages | share |
|---|---|---|---|
| (root) | 8,429 | 2,889,482 | 50.4% |
| **continuation** (Codex fork/resume) | 286 | **1,772,548** | **30.9%** |
| subagent (Claude Task + acompact) | 7,637 | 830,264 | 14.5% |
| sidechain | 131 | 248,968 | 4.3% |

The ~31% `continuation` share corroborates the design doc's "~32% of message rows
are duplicate-bearing" — and it is concentrated in **286 Codex sessions**, each a
near-total replay of its parent.

### Prefix-sharing copy candidates (should collapse to edge+tail after re-ingest)
- **286 Codex continuations** = 1,772,548 messages, almost all inherited prefix.
- **187 Claude `agent-acompact-*`** copies = 63,897 messages, ~100% parent overlap
  (~2 unique blocks each per design doc).
- The remaining ~7,450 Claude subagents are real fresh Task work →
  `spawned-fresh`, NOT deduped (legit, must keep all messages).

### Smoking-gun fork family (concrete before-number)
Parent `codex-session:019d1c2a-…` has **74,870** messages. **20 children** fork
from it, totaling **548,492** messages — i.e. ~547K duplicate prefix-replay rows
in one family. Top-3 Codex parent families by child-replay volume:
`019d1c2a` (20 kids / 548,492 msgs), `019cc941` (35 / 488,804),
`019cbcef` (19 / 342,404) — ~1.38M of the 1.77M continuation messages in 3 families.

**Expected post-re-ingest:** physical messages drop by ~1.5-1.8M (continuation
prefix replay + acompact copies collapse to tails); logical session count
unchanged (~8,809); composed reads identical to the current full transcripts.

---

## VALIDATION DESIGN (run against the POST-re-ingest archive, read-only)

Run each probe with `DB="file:<index.db>?mode=ro&immutable=1"`. First assert
`PRAGMA user_version = 14` and that `session_links` has the two new columns —
otherwise the re-ingest did not take and the rest is moot.

### (a) Dedup ratio — physical stored vs logical served
```sql
-- physical messages now (expect ~1.5-1.8M lower than 5,736,468)
SELECT COUNT(*) FROM messages;
-- prefix-sharing edges materialized
SELECT inheritance, COUNT(*) FROM session_links GROUP BY inheritance;
-- tail-only storage for forks: continuation message total should crater
SELECT COALESCE(branch_type,'(root)'), COUNT(*), SUM(message_count)
FROM sessions GROUP BY branch_type;
```
Python cross-check — for every prefix-sharing child, served (composed) length must
exceed stored tail length, and the *sum of stored* = sum of unique messages:
```python
# composed length per child via read path, stored via SUM(message_count)
# served_total = sum(len(get_messages(child)))  over a sample
# stored_total = SELECT SUM(message_count) FROM sessions
# dedup_ratio = served_total / stored_total  (expect ~1.6-1.9x for fork-heavy sample)
```
PASS: `SUM(message_count)` for `continuation` drops from 1,772,548 to roughly
`n_parents * tail` (tens of msgs/child); `inheritance='prefix-sharing'` count ≈
286 (Codex) + 187 (acompact) + any resolved Claude resume chains.

### (b) Branch-point integrity — no dangling references; safe bail
```sql
-- every prefix-sharing edge must resolve a parent and name a branch point
SELECT COUNT(*) FROM session_links
WHERE inheritance='prefix-sharing'
  AND (resolved_dst_session_id IS NULL OR branch_point_message_id IS NULL);
-- DANGLING: branch_point_message_id that does not exist as a message_id
-- in the resolved parent's OWN rows (composition will then recurse; a
-- truly dangling one is caught by the reader's "not found" bail).
SELECT l.src_session_id, l.branch_point_message_id
FROM session_links l
WHERE l.inheritance='prefix-sharing'
  AND NOT EXISTS (
    SELECT 1 FROM messages m
    WHERE m.message_id = l.branch_point_message_id
  );
```
For any rows the 2nd query returns, drive the reader (`get_messages` /
`read_archive_session_envelope`) on that child and assert it returns the child's
**own tail only** (the dangling-bail branch), never an over-long splice. Also
assert `spawned-fresh` rows have `branch_point_message_id IS NULL` (CHECK already
enforces the enum domain).

### (c) Compose-on-read correctness — composed == pre-normalization full transcript
Capture **NOW** (pre-re-ingest) the full stored transcript text for a fixed sample
of soon-to-be-normalized children, then after re-ingest compose them and compare.
Sample IDs to snapshot now (all currently store the full replay):
- `codex-session:019d4eea-48bb-7912-a446-3c0dfdfd530f` (62,776 msgs, parent 019d1c2a)
- `codex-session:019d4efc-a707-76b2-a2fb-701862c664f5` (62,774 msgs)
- 3-5 `agent-acompact-*` subagents (`native_id LIKE '%agent-acompact-%'`)
- a few `spawned-fresh` Claude subagents (must be byte-identical, no parent prepend)
```python
# BEFORE (v11): baseline = [msg.text for msg in get_messages(conn, sid)]
# store baseline JSON per sid under .agent/scratch/research/lineage-baseline/<sid>.json
# AFTER (v14):  composed = [msg.text for msg in get_messages(conn_v14, sid)]
# assert composed == baseline   (exact list equality, ordered)
```
Because `message_id` is deterministic, the composed ordering and content must match
exactly. PASS: every sampled child's composed transcript equals its pre-re-ingest
transcript; spawned-fresh children unchanged.

### (d) Per-origin breakdown
```sql
SELECT s.origin, l.inheritance, COUNT(*)
FROM session_links l JOIN sessions s ON s.session_id = l.src_session_id
GROUP BY s.origin, l.inheritance ORDER BY 3 DESC;
```
Expectations:
- `codex-session` + `prefix-sharing`: ≈286 (the fork/resume continuations).
- `claude-code-session` + `prefix-sharing`: the 187 `agent-acompact-*` **plus**
  any Claude resume chains (design doc notes Claude `continuation` was 0 before —
  this path now composes resume too; a non-zero Claude prefix-sharing count is the
  pre-existing-gap fix, verify it is acompact + resume, not real Task subagents).
- `claude-code-session` + `spawned-fresh`: ~7,450 real Task subagents (unchanged
  message counts).
- ChatGPT / Gemini / Antigravity: **no** prefix-sharing edges (no fork/compaction
  machinery) — assert zero.

### Authoritative cross-check (design doc anchor, optional heavy)
Codex per-thread token ratio vs `~/.codex/state_5.sqlite` should move toward
**1.00** (was 189.7B physical vs 139.3B authoritative). Compare
`session_provider_usage_events` / `session_model_usage` totals after re-ingest.

---

## Gotchas for the validator
- The 38 GB DB is daemon-active; always `mode=ro&immutable=1` and bound every
  query with `timeout`. `COUNT(*) FROM messages` took <7 min; avoid joins over
  the full `blocks` table.
- Do NOT trust a green `gh pr checks` as evidence the re-ingest ran — the schema
  change only takes effect after `polylogue ops reset --database && polylogued run`.
- `logical_session_count` is expected to be **stable** across the migration; only
  *physical* message/row counts should drop. If logical count changes, lineage
  roots were mis-resolved.
- Capture the (c) baselines BEFORE the operator re-ingests, or the before-number
  is lost (source.db raw replay is still there, but re-deriving is the whole point).
