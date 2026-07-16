## [assistant 320]
Got it. I read `A20-compaction-event.md` as a new design brief, not as a completed subagent report. Here is the finished version I would want from that agent, folded into the context-memory / query-object / artifact-linking design we were already building. fileciteturn0file0 fileciteturn1file0

## Recommendation

[evidence] Today Polylogue already detects compaction boundaries, preserves them as session events, and has v12/v14 lineage normalization where prefix-sharing children store only their divergent tail while reads recompose parent prefix plus child tail. The current code path is visible in `detect_context_compaction`, which extracts summary, timestamp, trigger, `pre_tokens`, preserved-segment id, and modern/legacy status (`polylogue/pipeline/semantic_capture.py:82-108`); Claude Code parsing turns that into a `ParsedSessionEvent(event_type="compaction")` plus a summary message (`polylogue/sources/parsers/claude/code_parser.py:261-289`); the index schema only has generic `session_events` rows for `compaction|capture_gap` (`polylogue/storage/sqlite/archive_tiers/index.py:345-354`); and lineage currently lives in `session_links.branch_point_message_id` + `inheritance` (`index.py:376-401`). The lineage model itself is documented as “fork/resume/spawned subagent/auto-compaction copy physically replays the parent prefix; writer drops inherited prefix; reads compose parent prefix + child tail.” (`docs/internals.md:214-227`) fileciteturn0file10

[proposal] Add **dedicated first-class `CompactionEvent` and `CompactionLossItem` derived tables in `index.db`, backed by raw snapshot blobs in `source.db`/`blob/`, and promote only the context-injection summaries into `user.db.assertions`.** That is the right split: the event/loss facts are rebuildable derived archive truth; the pre-compaction snapshot is raw evidence; the “remember this next time” injection is an operator/agent overlay. It also matches existing tier doctrine: `source.db` is raw evidence, `index.db` is parsed/rebuildable read model, `user.db` is irreplaceable human/agent overlay, and `blob/` is content-addressed binary payloads. (`docs/archive-backup.md:12-18`)

[proposal] Runner-up rejected: just stuff everything into `session_events.payload_json` or `user.db.assertions`. Overloading `session_events` would keep compactions second-class and make epidemiology/querying awkward. Storing all forensics as assertions would put rebuildable derived facts into the irreplaceable overlay tier and blur “measured loss” with “agent judgment.” The current assertion table is great for context-policy overlays because it has `target_ref`, `scope_ref`, `evidence_refs_json`, `status`, `confidence`, `staleness_json`, and `context_policy_json`, but it should not be the primary table for deterministic loss diffs. (`polylogue/storage/sqlite/archive_tiers/user.py:7-31`)

## Layer 1 — Today’s substrate

[evidence] Current Polylogue is already close enough that A20 is not a moonshot. It has four relevant foundations.

[evidence] First, compaction is already recognized as a semantic event. The current extractor recognizes both legacy `type="summary"` and modern `type="system", subtype="compact_boundary"` records. Modern records already expose exactly the fields A20 asks for in embryonic form: summary, trigger, pre-token count, preserved segment id, and timestamp (`semantic_capture.py:82-108`). Claude Code parser preserves the summary as a real system/summary message instead of discarding it (`code_parser.py:275-289`). Tests cover legacy/modern Claude Code compactions, Codex compacted records, emitted session events, and profile counts (`tests/unit/sources/test_compaction.py`, multiple blocks).

[evidence] Second, lineage normalization is already the right conceptual base. The schema comments say `session_links` records prefix-sharing lineage for forks/resumes/spawned subagents/auto-compaction copies, with `branch_point_message_id` as the last inherited parent message and `inheritance='prefix-sharing'|'spawned-fresh'` (`index.py:376-401`). Read composition walks up the parent chain, then composes down from root prefix to child tail, and bails to the child tail if the branch point dangles (`polylogue/storage/sqlite/queries/message_query_reads.py:65-84`, `119-159`). The uploaded lineage audit also shows why this area matters: wrong parent assignment and reads bypassing composition can make compactions/forks lie. fileciteturn1file9

[evidence] Third, raw artifact and blob infrastructure exists. `raw_artifacts` can classify source artifacts by kind, path, cohort/link group, support status, malformed line counts, etc. (`polylogue/storage/sqlite/archive_tiers/source.py:84-105`). The backup docs state that `blob/` is content-addressed by SHA-256 and must be backed up when referenced by `source.db` or `user.db` (`docs/archive-backup.md:16-18`). That is enough to store pre-compaction snapshots as content-addressed raw evidence.

[evidence] Fourth, the user-overlay tier already supports the “loss record must itself survive the next compaction” part. The assertion schema stores evidence-linked annotations/corrections/recall packs/workspaces/judgments and includes `context_policy_json` for injection behavior (`user.py:7-31`). The broader bead-set characterization already frames `37t` as the context/memory loop, with `37t.11` as the context scheduler that arbitrates what enters agent context. fileciteturn1file14

[proposal] So the gap is not “can Polylogue know compaction happened?” It can. The gap is that compaction is currently an event count and lineage edge, not an archived object with boundaries, snapshots, loss records, recall surfaces, and corpus metrics.

## Layer 2 — Near-term substrate change

### `CompactionEvent` schema

[proposal] Add a new derived table in `index.db`:

```sql
CREATE TABLE compaction_events (
    compaction_id              TEXT PRIMARY KEY,
    session_id                 TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,

    -- Boundary identity
    pre_boundary_message_id     TEXT,
    boundary_summary_message_id TEXT,
    post_continuation_message_id TEXT,

    -- Existing lineage link, if this compaction is represented as a prefix-sharing child
    parent_session_id           TEXT,
    continuation_session_id     TEXT,
    session_link_src_session_id TEXT,
    branch_point_message_id     TEXT,

    -- Harness event details
    provider                    TEXT NOT NULL,
    origin                      TEXT NOT NULL,
    trigger                     TEXT CHECK(trigger IN ('auto','manual','unknown') OR trigger IS NULL),
    harness_summary_text        TEXT NOT NULL DEFAULT '',
    pre_tokens                  INTEGER,
    token_budget_at_fire        INTEGER,
    preserved_segment_id        TEXT,
    occurred_at_ms              INTEGER,

    -- Snapshot evidence
    snapshot_ref                TEXT,
    snapshot_blob_hash          TEXT,
    snapshot_source             TEXT NOT NULL CHECK(snapshot_source IN (
        'precompact-hook',
        'jsonl-boundary',
        'reconstructed-composed-context',
        'none'
    )),
    snapshot_confidence         REAL NOT NULL DEFAULT 1.0 CHECK(snapshot_confidence BETWEEN 0 AND 1),
    snapshot_lossy              INTEGER NOT NULL DEFAULT 0 CHECK(snapshot_lossy IN (0,1)),

    -- Content-addressed identity / replay
    source_raw_id               TEXT,
    source_artifact_id          TEXT,
    source_record_index         INTEGER,
    event_content_hash          TEXT NOT NULL,
    extractor_version           INTEGER NOT NULL,
    degraded_reasons_json       TEXT NOT NULL DEFAULT '[]',

    created_at_ms               INTEGER NOT NULL
) STRICT;
```

[proposal] Keep the current `session_events` row for compatibility, but treat it as a compact summary/index row. `compaction_events` becomes the full relation object. The row should be rebuildable from raw session evidence plus lineage resolution, so `index.db` is correct. The associated hook snapshot itself belongs in the content-addressed blob store, with a `raw_artifacts` or `raw_hook_events` link in `source.db`.

[proposal] Identity should be stable across rebuilds and source re-ingest:

```text
compaction_id =
  "compaction:" + sha256(
    "compaction-event-v1\0" +
    origin + "\0" +
    source_native_session_id + "\0" +
    provider_boundary_uuid_or_source_line + "\0" +
    pre_boundary_provider_message_id_or_empty + "\0" +
    post_continuation_provider_message_id_or_empty
  )[:32]
```

[evidence] Do not make `compaction_id` depend on SQLite row ids or on normalized `message_id` alone. The lineage docs explicitly rely on message ids being deterministic across re-ingest (`docs/internals.md:197-204`), but raw provider identity is still the safer primary identity for a source-derived event. The normalized message ids should be boundary pointers, not the only event key.

[proposal] `event_content_hash` is separate from `compaction_id` and should hash the normalized event payload:

```text
sha256(canonical_json({
  trigger,
  harness_summary_text,
  pre_tokens,
  token_budget_at_fire,
  preserved_segment_id,
  snapshot_blob_hash,
  pre_boundary_message_id,
  post_continuation_message_id
}))
```

This lets rebuilds detect “same event identity, changed interpreted content,” which is exactly the kind of thing Polylogue should expose loudly.

### Snapshot model

[proposal] Add `context_snapshots` or extend existing `session_context_snapshots` for real pre-compaction snapshots. Existing `session_context_snapshots` already has `snapshot_ref`, `session_id`, `run_ref`, `position`, `boundary`, `inheritance_mode`, `segment_refs_json`, `evidence_refs_json`, `metadata_json`, `payload_json`, and `search_text` (`index.py:1000-1015`). That is close, but A20 needs a **raw pre-compaction payload snapshot**, not just a derived insight snapshot.

[proposal] Use two-level snapshotting:

```text
snapshot_ref
  -> snapshot manifest row:
       session_id
       boundary = "pre_compaction"
       compaction_id
       snapshot_source
       blob_hash nullable
       segment_refs_json
       evidence_refs_json
       metadata_json
       payload_json
```

If `snapshot_source='precompact-hook'`, store the exact assembled context payload as a blob and reference it. If `snapshot_source='jsonl-boundary'`, store a deterministic reconstructed manifest: message refs from the composed logical transcript up to the boundary, plus a degraded flag saying this is not the real model-context payload. If `snapshot_source='reconstructed-composed-context'`, the source is even weaker: use composed archive messages, not raw hook context.

[proposal] This is how to satisfy the “snapshotting every compaction is near-free” requirement honestly. Do not store huge duplicated text when a manifest of message/block/blob refs is enough. Store exact hook payload blobs only when available. Content-addressed blob dedupe keeps repeated prefixes cheap, but the manifest approach is the real win.

## Layer 2 — Loss-forensics diff algorithm

[proposal] The core algorithm should be deterministic and structural, with optional LLM annotation later. Do not use an LLM to decide the base retained/lost/transformed classification. The base pass should be boring enough to test.

Inputs:

```text
pre_snapshot:
  source = precompact-hook | jsonl-boundary | reconstructed-composed-context
  payload blob or segment/message/block refs

post_continuation:
  harness summary message
  first N continuation messages
  child tail messages if prefix-sharing
  later continuation window for later-reference detection
```

Normalize both sides into item sets.

```python
PreItem = {
  "item_id": stable hash,
  "tier": "file-path" | "tool-outcome" | "marked-decision" | "cited-ref",
  "anchor_key": canonical key,
  "display": short text,
  "source_refs": [...],
  "span_refs": [...],
  "salience": numeric,
  "metadata": {...},
}
```

### Tier 1: file-path

[proposal] Extract from tool inputs, tool outputs, code blocks, Markdown refs, stack traces, and text mentions. Normalize paths relative to known repo/cwd when available.

Canonical key:

```text
file:<repo_ref_or_unknown>:<normalized_relpath>
```

Classification:

```text
retained     exact path appears in harness summary or continuation context
transformed  basename/parent-dir appears with enough deterministic context,
             or path is represented by a commit/diff/ref that includes it
lost         no structural representation found
```

Examples:

```text
polylogue/storage/sqlite/archive_tiers/index.py  -> retained
storage/sqlite schema file                       -> transformed
(no mention)                                     -> lost
```

### Tier 2: tool-outcome

[evidence] Polylogue already has an `actions` view pairing tool-use blocks to tool-result blocks with `tool_name`, `tool_command`, `tool_path`, `tool_input`, `output_text`, `is_error`, and `exit_code` (`index.py:324-343`). This is exactly the substrate needed for outcome loss.

[proposal] Canonical key:

```text
tool:<session_id>:<tool_use_block_id>
```

or, when crossing source/rebuild boundaries:

```text
tool-hash:<tool_name>:<command/path digest>:<exit_code>:<output digest prefix>
```

Classification:

```text
retained     same tool id, command, exit status, or exact result ref appears
transformed  summary preserves the fact of the outcome
             e.g. "devtools verify passed" or "mypy failed in X"
lost         tool result not represented
```

Special weighting: failed tool outcomes, nonzero exit codes, and `is_error=true` are high-loss-risk even if old. This directly supports claim-vs-evidence and post-compaction recovery.

### Tier 3: marked-decision

[evidence] `user.db.assertions` already supports durable decision/judgment-like overlays with target refs, evidence refs, confidence, status, and context policy (`user.py:7-31`). The bead-set also treats context/memory and judgments as a core direction under `37t`. fileciteturn1file14

[proposal] Marked decisions include:

```text
user.db assertions kind=decision|lesson|caveat|blocker|handoff|judgment
bd remember entries, once imported as assertions
Bead decisions or doctrine refs
explicit "we decided" / "operator said" messages when parser can mark them
```

Canonical key:

```text
assertion:<assertion_id>
decision-key:<scope_ref>:<key>:<stance_hash>
```

Classification:

```text
retained     assertion id/key or exact decision text is in summary/context
transformed  same target+stance is summarized under different wording
lost         no equivalent decision representation
```

Marked decisions should rank higher than raw file paths, because losing them is exactly how post-compaction agents repeat settled debates.

### Tier 4: cited-ref

[proposal] Cited refs include:

```text
polylogue refs: session/message/block/assertion/query/cohort/etc.
git refs: commit SHA, branch, tag
GitHub refs: #NNNN, PR URLs, issue URLs
file:line refs
external URLs
artifact refs
```

Canonicalization:

```text
commit short SHA -> full SHA if repo has it
#2547 + repo -> gh:repo#2547
file path + line -> file:<repo>:<path>:Lx-Ly
polylogue public refs -> resolved object refs
```

Classification:

```text
retained     exact canonical ref survives
transformed  equivalent alias survives (#2547 vs PR URL; short vs full SHA)
lost         no equivalent canonical ref
```

### Loss item table

[proposal] Add:

```sql
CREATE TABLE compaction_loss_items (
    loss_item_id             TEXT PRIMARY KEY,
    compaction_id            TEXT NOT NULL REFERENCES compaction_events(compaction_id) ON DELETE CASCADE,
    item_tier                TEXT NOT NULL CHECK(item_tier IN ('file-path','tool-outcome','marked-decision','cited-ref')),
    item_key                 TEXT NOT NULL,
    item_display             TEXT NOT NULL DEFAULT '',
    classification           TEXT NOT NULL CHECK(classification IN ('retained','lost','transformed','unknown')),
    confidence               REAL NOT NULL DEFAULT 1.0 CHECK(confidence BETWEEN 0 AND 1),

    pre_anchor_refs_json      TEXT NOT NULL DEFAULT '[]',
    post_anchor_refs_json     TEXT NOT NULL DEFAULT '[]',
    later_reference_refs_json TEXT NOT NULL DEFAULT '[]',
    evidence_refs_json        TEXT NOT NULL DEFAULT '[]',

    loss_score                REAL NOT NULL DEFAULT 0,
    later_reference_score     REAL NOT NULL DEFAULT 0,
    salience_score            REAL NOT NULL DEFAULT 0,
    explanation               TEXT NOT NULL DEFAULT '',
    degraded_reasons_json     TEXT NOT NULL DEFAULT '[]',

    extractor_version         INTEGER NOT NULL,
    created_at_ms             INTEGER NOT NULL
) STRICT;
```

Useful indexes:

```sql
CREATE INDEX idx_compaction_loss_items_event_score
ON compaction_loss_items(compaction_id, classification, loss_score DESC);

CREATE INDEX idx_compaction_loss_items_tier
ON compaction_loss_items(item_tier, classification);

CREATE INDEX idx_compaction_events_session
ON compaction_events(session_id, occurred_at_ms);
```

## Ranking lost items

[proposal] Rank lost items by predicted usefulness, but keep the score decomposed so it is auditable.

```text
loss_score =
  100 * later_reference_signal
+  40  * marked_decision_weight
+  35  * failed_tool_outcome_weight
+  25  * cited_ref_weight
+  15  * file_path_weight
+  recency_weight
+  repeated_reference_weight
+  branch/repo/current-task overlap
-  retained_or_transformed_credit
-  low_confidence_penalty
```

[evidence] This mirrors the postmortem design pattern already present: aggregate fields carry evidence refs, degraded states are explicit, and missing signal becomes `unavailable` rather than fabricated (`polylogue/insights/postmortem.py:1-14`, `127-167`, `198-208`). A20 should reuse that posture.

[proposal] “Later-reference likelihood” should not mean vibes. It should be measured where possible:

```text
later_reference_signal = item_key reappears after compaction
                        in a user request,
                        tool call,
                        file path,
                        cited ref,
                        failing command,
                        final answer,
                        or imported assertion
```

The best item is not merely “lost.” It is **lost, then later needed**. That is the actual harm proxy.

## Honest degradation

[proposal] `snapshot_source` must be first-class, not a footnote.

`precompact-hook`: strongest. Polylogue captured the actual assembled context payload immediately before compaction. Loss-forensics can claim: “this was in the model context.”

`jsonl-boundary`: medium/weak. Polylogue found a provider compaction boundary in raw JSONL but did not capture the actual assembled payload. Loss-forensics can claim only: “this was in the archive’s composed logical transcript before the boundary,” not “this was certainly in the model context.”

`reconstructed-composed-context`: weak. Polylogue reconstructed the likely pre-context from current archive state and lineage. It is useful but can be wrong if provider context included hidden system/developer/tool state not represented in messages.

`none`: no snapshot; only event epidemiology can run.

[proposal] Every `CompactionEvent` and `CompactionLossItem` should carry `degraded_reasons_json`. Examples:

```json
[
  "snapshot_source=jsonl-boundary: no PreCompact hook payload captured",
  "post_continuation_missing: no continuation message found after summary",
  "lineage_unresolved: session_links.resolved_dst_session_id is null",
  "branch_point_dangles: composed parent prefix unavailable",
  "tool_outcomes_partial: tool_result blocks missing for some tool_use blocks"
]
```

[evidence] This is aligned with existing Polylogue design doctrine: postmortem fields degrade honestly when no signal exists instead of fabricating (`postmortem.py:37-44`, `127-145`). The hot-daemon and composer reports make the same product-level argument for explicit degraded state and stale/partial markers. fileciteturn0file7 fileciteturn0file5

## Read and recall surfaces

### CLI

[proposal]

```bash
polylogue compactions SESSION
polylogue compaction read COMPACTION_REF
polylogue compaction forgot SESSION [--latest] [--tier tool-outcome] [--top 20]
polylogue compaction inject SESSION --top 8 --budget 1200 --format markdown
```

`forgot` returns ranked loss items:

```json
{
  "session_id": "...",
  "compaction_id": "compaction:...",
  "snapshot_source": "jsonl-boundary",
  "degraded": true,
  "lost_count": 37,
  "transformed_count": 19,
  "retained_count": 112,
  "items": [
    {
      "rank": 1,
      "tier": "tool-outcome",
      "classification": "lost",
      "display": "devtools verify failed: 3 tests",
      "loss_score": 184.2,
      "why": "failed tool outcome; later referenced by continuation; absent from compact summary",
      "pre_anchors": ["message:...", "block:..."],
      "later_refs": ["message:..."]
    }
  ]
}
```

### MCP

[proposal] Add the tool A20 names:

```text
compaction_forgot(session_id, latest=true, top_k=20, tiers=[...])
```

Return the ranked list plus anchors. Do not return a prose summary only; agent loops need stable refs.

[proposal] Also add:

```text
compaction_reground(session_id, compaction_id?, budget_tokens=1200, top_k=8)
```

This returns a bounded context-injection pack:

```json
{
  "context_pack_ref": "context-pack:...",
  "target_session_id": "...",
  "compaction_id": "...",
  "budget_tokens": 1200,
  "included_loss_items": [...],
  "rendered_markdown": "## Lost across compaction..."
}
```

### Re-grounding recursion

[proposal] The re-grounding pack must itself become an assertion or context snapshot so it survives the next compaction.

Use `user.db.assertions`:

```json
{
  "target_ref": "compaction:...",
  "scope_ref": "session:...",
  "kind": "handoff",
  "key": "compaction.loss_regrounding.v1",
  "body_text": "Critical items lost across previous compaction: ...",
  "evidence_refs_json": ["compaction_loss_item:...", "message:...", "block:..."],
  "context_policy_json": {
    "inject": true,
    "condition": "next_compaction_or_session_resume",
    "budget_class": "loss-forensics",
    "max_tokens": 1200
  }
}
```

[evidence] The assertion table is already designed to carry `context_policy_json`, evidence refs, staleness, and supersession (`user.py:22-31`). The A20 recursion requirement should use that instead of inventing “compaction memory” as a separate store.

## Epidemiology

[proposal] Once `compaction_events` and `compaction_loss_items` exist, corpus-level epidemiology becomes simple relation algebra.

Views:

```sql
CREATE VIEW compaction_loss_summary AS
SELECT
  e.provider,
  e.origin,
  e.trigger,
  e.snapshot_source,
  CASE
    WHEN s.message_count < 100 THEN '0-99'
    WHEN s.message_count < 500 THEN '100-499'
    WHEN s.message_count < 2000 THEN '500-1999'
    ELSE '2000+'
  END AS session_length_bucket,
  COUNT(*) AS item_count,
  SUM(i.classification = 'lost') AS lost_count,
  SUM(i.classification = 'transformed') AS transformed_count,
  SUM(i.classification = 'retained') AS retained_count,
  AVG(i.loss_score) AS avg_loss_score
FROM compaction_events e
JOIN sessions s ON s.session_id = e.session_id
LEFT JOIN compaction_loss_items i ON i.compaction_id = e.compaction_id
GROUP BY 1,2,3,4,5;
```

Product queries:

```text
compactions where trigger:auto
  | group by provider,session_length_bucket
  | count
```

```text
compaction-loss where classification:lost tier:tool-outcome
  | group by provider,trigger
  | count
```

```text
compactions where snapshot_source:jsonl-boundary
  | project compaction-forensics-card
```

Metrics:

```text
event_count
snapshot_coverage_rate
precompact_hook_rate
jsonl_boundary_degraded_rate
lost_item_rate
transformed_item_rate
lost_but_later_referenced_rate
failed_tool_outcome_loss_rate
marked_decision_loss_rate
average top-K re-grounding token cost
```

[evidence] This fits the existing read-algebra direction: the swarm brief defines query units including observed events, context snapshots, messages, actions, blocks, assertions, files, and runs, with query/projection/render as the read model. fileciteturn0file10 B8’s contract report says the substrate should expose query/read/preview/complete/act/status/facets as one typed client contract, which is where these compaction surfaces should live rather than becoming a one-off side CLI. fileciteturn0file3

## Layer 3 — Full direction

[proposal] The full direction is not merely “store compaction events.” It is **memory loss observability**.

The object graph should become:

```text
Session
  -> CompactionEvent
      -> PreContextSnapshot
      -> HarnessSummaryMessage
      -> ContinuationMessage
      -> LossItems
      -> RegroundingContextPack
      -> Assertion/Handoff injected into later context
```

This plugs into the broader design from the previous answers:

```text
raw sessions + artifacts
  -> typed units
  -> query runs/result relations
  -> cohorts
  -> annotation batches/assertions
  -> analysis/context runs
  -> reports/context packs
```

[proposal] In the best design, every compaction boundary becomes an analyzable experiment:

```text
What was present before?
What did the harness summarize?
What vanished?
What was later needed?
What did the next agent waste time rediscovering?
What should be injected next time?
```

That is far stronger than Claude Code’s current behavior in the uploaded excerpt, where compaction yields a summary plus a flat persistent memory dump and then the next agent proceeds with whatever made it into text. fileciteturn0file1 Polylogue should not merely preserve the dump; it should measure what was lost and select re-grounding context from evidence.

[proposal] `CompactionEvent` should also become a basis for agent benchmarking:

```text
handoff-pack A: ordinary summary only
handoff-pack B: summary + top-K measured lost items
measure: tokens-to-first-correct-action, repeated-file-read count,
         repeated-failed-command count, correction requests, final task success
```

This connects to the “uplift artifact” problem in the situation brief: the strategic gap is not substrate capability but externally visible demonstrated use. A compaction loss/re-grounding demo could be one of the cleanest uplift demos because it is naturally two-arm and measurable. fileciteturn0file11

## How this improves current designs

[proposal] The previous “context compiler” design needs one correction: it should not only compile from durable doctrine/assertions. It should compile from **measured context-loss events** too.

So `37t` should not be only:

```text
claims -> judgment -> preamble -> reboot
```

It should become:

```text
compaction event
  -> loss forensics
  -> loss assertions / handoff assertions
  -> context compiler
  -> next compaction survives the previous loss record
```

[proposal] The query-object design also gets sharper. A loss-forensics run should itself be a query/analysis object:

```text
analysis_run: compaction-loss-forensics
  input: compaction_event
  query_runs:
    - pre_snapshot extraction
    - post_summary extraction
    - later_reference scan
  result_relation: compaction_loss_items
  render_artifact: forgotten-items report
  promoted_assertion: top-K regrounding pack
```

[proposal] The artifact-linking design gets a new important artifact kind:

```text
artifact_kind = precompact-context-snapshot
artifact_kind = compaction-loss-report
artifact_kind = regrounding-context-pack
```

Again: not a `scratchpad` special case. These are generic artifacts produced/consumed by session/analysis/context machinery.

## Bead implications

[proposal] I would make A20 a real bead, probably under `gjg` if `gjg` is indeed the compaction-lifecycle theme the audit wanted to retype into an epic. The uploaded audit output explicitly had `gjg→epic` in the held type-change list, and A20 is exactly the kind of child that justifies that promotion. fileciteturn1file8

Recommended bead structure:

```text
gjg — Compaction lifecycle / memory-loss truth [epic]
  gjg.1 — CompactionEvent schema + parser backfill
  gjg.2 — PreCompact snapshot capture + blob-backed manifests
  gjg.3 — Deterministic loss-forensics extractor
  gjg.4 — compaction_forgot CLI/MCP/read surfaces
  gjg.5 — Re-grounding context-pack injection via assertions/context_policy
  gjg.6 — Corpus epidemiology metrics
  gjg.7 — Two-arm uplift demo: summary-only vs measured-loss-regrounding
```

Dependencies / relates-to:

```text
gjg.* relates-to 4ts lineage truth
gjg.* relates-to 37t context/memory loop
gjg.4 depends-on B8/A2/t46 contract surface if implemented through daemon/MCP
gjg.5 depends-on assertion batch/context policy semantics
gjg.7 depends-on 3tl/cfk-style external artifact lane
```

[proposal] If you do not want a new `gjg` epic, put it under `37t`; but I think that is the runner-up. `37t` is the broader context scheduler/memory loop. Compaction-event/loss-forensics is a distinct archive-truth capability and should not be buried as just “context injection.”

## Open questions for the operator

1. Should `precompact-hook` capture the exact model payload text, or a manifest of message/block/tool refs plus hashes? My recommendation: capture both when cheap; manifest is mandatory, exact payload blob optional but preferred.

2. How aggressive should re-grounding injection be? My default: top 5–8 lost-but-later-referenced items, hard token budget, never inject all lost items.

3. Should loss-forensics run automatically at ingest/convergence, or lazily on first `compaction_forgot`? My recommendation: derive the `CompactionEvent` eagerly, compute loss items lazily at first read, then cache/materialize.

4. Should LLM-based semantic/paraphrase matching ever affect `retained/transformed/lost`? My recommendation: no for v1. LLM annotations can add a separate judgment layer later, but deterministic structural matching should own the base classification.

5. Is `gjg` the intended compaction-lifecycle epic, or should this attach directly to `37t` and `4ts`? The audit hints `gjg→epic`, but I did not inspect live Beads state after the uploaded A20.

## What’s missing

I did not run the live daemon or inspect the current live Beads database, so `gjg` status and exact parentage need local verification. The snapshot is from the uploaded package and reports, so code may have moved since July 5, 2026. I inspected the extracted code snapshot for the core lineage/compaction/assertion/blob pieces, but I did not run tests, build migrations, or verify whether `insights/postmortem.py` has changed after the package. I also did not inspect actual Claude Code PreCompact hook payload formats beyond the uploaded excerpt and current parser tests, so the precise `precompact-hook` raw schema still needs a fixture pass.
