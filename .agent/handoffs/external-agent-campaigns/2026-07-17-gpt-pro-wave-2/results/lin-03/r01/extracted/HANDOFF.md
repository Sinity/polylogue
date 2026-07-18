# HANDOFF — polylogue-4ts.3

## Mission and result

This package implements bead `polylogue-4ts.3`: stop treating every Claude Code `agent-acompact-*` artifact as a main-session continuation. The prefix is an overloaded provider marker. A true main-session compactor inherits the main transcript; a Task-subagent self-compactor is a fresh sidechain and must never receive a composed main-session prefix.

The implementation uses two cooperating production checks:

1. The Claude parser recognizes a narrow, structural fresh-Task head (`type=user`, root `parentUuid`, `isSidechain=true`, non-human origin, and explicit `agentId`/`promptId`) and emits sidechain topology immediately.
2. The archive writer performs the authoritative content-membership comparison once the asserted parent is available. Membership below 90% changes the persisted session/link to `sidechain` + `spawned-fresh`, retains the child transcript whole, and leaves the branch point null. Membership at or above 90% preserves or restores `continuation` + `prefix-sharing` and uses the existing stricter contiguous-prefix extraction before deleting replay rows.

This split is necessary because a normal live-ingest call often receives only one `agent-acompact-*` sidecar; the parser cannot compare bytes from a separate main-session file it was not given. The writer is the first shared route that has both the child and resolved parent, including child-before-parent ingestion.

## Snapshot identity

The patch is against:

- Repository: Polylogue
- Snapshot source recorded by the archive: `/realm/project/polylogue`
- Branch: `master`
- Commit: `536a53efac0cbe4a2473ad379e4db49ef3fce74d`
- Commit subject: `fix(repair): harden raw authority convergence (#3046)`
- Commit timestamp: `2026-07-17T18:55:47+02:00`
- Snapshot generated: `2026-07-17T18:09:50Z`
- Supplied project archive SHA-256: `ec3fa193b2f99a6daee0bc620af4f69b5728834e99f8196792a17c3fbae11155`
- Supplied project archive size: `128314788` bytes

The snapshot manifest says `dirty=true`, but its branch-delta patch, changed-file list, and branch-delta log are all empty, with merge base equal to the named commit. The extracted repository was also clean before this implementation. No recoverable dirty patch was supplied, so `PATCH.diff` is deliberately based on the named commit above. An integrator with additional unexported local changes must resolve any overlap manually.

## Authority and evidence inspected

The implementation was grounded in the current repository, not only the task summary. The inspection covered:

- Repository instructions: `CLAUDE.md`, `AGENTS.md`, and `TESTING.md`.
- Bead authority: the complete `polylogue-4ts.3` row in `.beads/issues.jsonl`, plus `.agent/handoffs/polylogue-gpt-pro-2026-07-07/prework-v2/task_packets/080_polylogue_4ts_3.md` and the later handoff note `.agent/handoffs/external-agent-campaigns/2026-07-16-gpt-pro-wave/03-lineage-compaction-truth/080_polylogue_4ts_3.md`.
- Corpus investigation evidence: the archived GitHub issue #2471 material, including the measured `~39/187` below-90%-membership files and `9` zero-overlap files.
- Parser and dispatch: `polylogue/sources/parsers/claude/code_parser.py`, `polylogue/sources/dispatch.py`, source acquisition/streaming dispatch, and parser model definitions.
- Persistence and composition: `polylogue/storage/sqlite/archive_tiers/write.py`, `session_links` schema and edge vocabulary, branch-to-edge mapping, session graph resolution, and `read_archive_session_envelope`/message composition.
- Tests and fixtures: Claude parser/dispatch suites, lineage-normalization real-route tests, writer graph-resolution tests, and delegation-fact regression coverage.
- Design: `docs/design/session-lineage-model.md`.
- History: classifier introduction `7412b69a9b64863a1bf4df11d7feeb99f61262ff`, streaming merge `a651bb95f1fc3595cff8f545ec3768831b858daf`, and investigation note `aa4a520b44861220c5ad16370b4c99634119e98d`.

## Root cause

`parse_code_stream` used the `agent-acompact-*` filename/fallback prefix as enough evidence to set `branch_type=continuation` and `parent_session_provider_id=<main sessionId>`. That was valid for the originally documented main-session replay shape, but false for Task-subagent self-compactions emitted under the same prefix.

The downstream writer interpreted continuation edges as eligible for prefix extraction. A child with a small accidental leading match could therefore lose local rows and inherit main-session content. A zero-overlap child avoided row deletion because contiguous alignment returned zero, but still kept the wrong continuation topology. Composition and derived lineage then represented a parent context the Task agent never had.

There are two ingest orders to repair:

- Parent known when the child is saved: classification and extraction happen in the main `write_session` route.
- Child saved before parent: the child is initially stored whole with an unresolved link; `_resolve_session_graph` invokes `_reextract_prefix_tail_db` when the parent later arrives.

Fixing only the parser or only the parent-known write route leaves a production twin wrong.

## Classifier and persistence mechanism

### Parser-side structural signal

`polylogue/sources/parsers/claude/code_parser.py` adds `_is_fresh_task_prompt_head`. The predicate is deliberately narrow and provider-structural rather than filename-only. It requires:

- the first plain user message;
- a root `parentUuid`;
- `isSidechain=true`;
- no meta/compact/tool-result shape;
- no explicit human origin;
- a user-role message body;
- non-empty `agentId` and `promptId`.

For `agent-acompact-*`, this signal yields `BranchType.SIDECHAIN`. An ambiguous artifact remains `BranchType.CONTINUATION` until parent evidence exists. `parent_session_provider_id` is retained for both classes because a spawned-fresh sidechain still has a relationship to its spawning session; what changes is topology/inheritance, not whether the relationship exists.

`Role` and `MaterialOrigin` logic are untouched. The synthetic Task prompt continues to normalize as generated context rather than authored user material.

### Authoritative content-membership gate

`polylogue/storage/sqlite/archive_tiers/write.py` adds a Claude-acompact-specific membership gate at 0.90, matching the bead's measured classification boundary.

The comparison:

- is scoped to Claude Code sessions whose provider-native suffix begins `agent-acompact-`;
- compares normalized message-content signatures before the first summary boundary;
- treats matches as a multiset bounded by parent multiplicity, so repeated boilerplate cannot manufacture overlap;
- uses membership rather than contiguous order for topology classification;
- still delegates physical replay deletion to the existing contiguous-prefix aligner, which is the loss-prevention gate.

Outcomes:

- `< 0.90`: update session branch to `sidechain`; write/update the parser-parent link as `sidechain`, `spawned-fresh`, null branch point; retain every child row.
- `>= 0.90`: update a conservative parser sidechain hint back to `continuation`; preserve existing contiguous prefix extraction and `prefix-sharing` behavior.
- no pre-summary evidence: preserve a positive parser sidechain signal as spawned-fresh; otherwise retain existing conservative behavior.

The same decision is implemented in `_reextract_prefix_tail_db`. When a parent arrives after its child, it may move the `session_links` row between the continuation and sidechain primary-key lanes, removes a stale duplicate target key first, updates `sessions.branch_type`, and either returns without deleting rows or continues through existing prefix-tail extraction.

### Streaming-path cost

The memory-bounded Claude JSONL route and eager route both call the same Claude parser; there is no duplicate classifier to keep synchronized. The audit did find a sibling divergence: eager multi-session grouping replaced the first group's filename fallback with its `sessionId`, while the streaming route preserved the supplied fallback for the first group. That erased `agent-acompact-*` identity only in eager parsing. The patch makes eager grouping follow the streaming rule and adds full-model parity coverage.

Memory behavior remains bounded:

- Parser fresh-head detection keeps only booleans and fields from the current record: O(1) additional state and no retained raw transcript.
- Writer membership is O(P + C) time for parent-composed signatures plus the child's pre-summary prefix, with O(U) counter space for unique parent signatures. It does not buffer sibling JSONL bytes. Parent/child signature materialization already exists in the prefix-extraction/resolution route; the parent-known path passes the computed parent signature list into `_extract_prefix_tail` to avoid a second query/hash pass.

## Changed files

- `polylogue/sources/parsers/claude/code_parser.py` — narrow fresh-Task-head classifier and overloaded acompact branching.
- `polylogue/sources/dispatch.py` — eager/stream first-group fallback parity so the shared classifier receives identical artifact identity.
- `polylogue/storage/sqlite/archive_tiers/write.py` — authoritative 90% membership gate in parent-known and delayed-parent routes; safe session/link reclassification.
- `docs/design/session-lineage-model.md` — correct the obsolete assertion that all 187 acompacts are complete main-session copies and document the two outcomes.
- `tests/fixtures/claude-code/normalization-lineage-subagent-acompact.jsonl` — privacy-safe, fully synthetic zero-overlap Task self-compaction shape.
- `tests/unit/sources/test_compaction.py` — direct parser classification for true-main and fresh-Task shapes.
- `tests/unit/sources/test_claude_code_normalization_laws.py` — archive/write/read behavior in both ingest orders, membership authority, no-prefix assertion, and semantic preservation.
- `tests/unit/sources/test_dispatch_payloads.py` — eager vs memory-bounded route equality and fallback identity regression.

No schema file, enum vocabulary, `Role`, or `MaterialOrigin` semantics changed. `FILES/` is omitted because the unified patch is complete and unambiguous.

## Fixture derivation

The true-main case uses the repository's existing synthetic lineage fixtures: a parent user/assistant transcript followed by an `agent-acompact-*` artifact that replays those two messages and adds a summary. It must store only the summary in the child and compose parent + summary.

The Task case is derived from the recorded zero-overlap “12-change/12-commit testing overhaul” shape but contains no live-corpus text, IDs, paths, or secrets. It begins with a root sidechain Task prompt carrying synthetic `agentId`/`promptId`, followed by a task-local assistant response and summary. It must store all three rows, expose `sidechain` + `spawned-fresh`, and compose exactly those three messages.

A third synthetic case intentionally omits the fresh-head marker and includes one coincidental main-session message followed by four task-local rows. Its pre-summary membership is below 90%. This prevents a vacuous implementation that only keys on the new parser predicate: the writer must override the parser's conservative continuation in both ingest orders.

A fourth case decorates a 100%-matching main replay as if it had a fresh Task head. It proves resolved parent content is authoritative over the parser hint and restores continuation/prefix-sharing in both ingest orders.

## Acceptance matrix

| Requirement | Implemented proof | Result |
|---|---|---|
| True main acompact keeps parent and prefix sharing | `test_acompact_resume_replayed_prefix_is_stored_once_and_composed`, parent-first and child-first | Pass |
| Fresh Task self-compaction is sidechain/spawned-fresh | direct parser test plus `test_subagent_self_compaction_stays_whole_and_never_composes_main_prefix`, both orders | Pass |
| Composed Task read never receives main prefix | explicit absence and exact-message assertions in the real archive read | Pass |
| Ambiguous below-90% shape is content-classified | `test_ambiguous_acompact_is_reclassified_by_parent_content_membership`, both orders | Pass |
| Parent evidence can correct a conservative false sidechain hint | `test_parent_membership_overrides_conservative_fresh_head_hint`, both orders | Pass |
| Child-before-parent rows re-resolve | every primary case is parametrized over ingest order; dedicated existing subagent delayed test retained | Pass |
| Streaming and eager routes do not diverge | full `ParsedSession.model_dump()` equality and acompact identity/topology assertions | Pass |
| Role/MaterialOrigin semantics unchanged | Task composed envelope asserts roles and material origins | Pass |
| No schema change | patch inspection | Pass |
| Live-corpus count movement | requires operator archive and rebuild | Unverified here |

## Reclassification path for existing data

`index.db` is rebuildable. The operator route in this snapshot is:

```bash
polylogue ops reset --index --yes
polylogue ops maintenance rebuild-index
```

`ops reset --index` preserves source evidence and deletes the rebuildable index tier; its current CLI output points to `ops maintenance rebuild-index` for replay from `source.db`.

On each full-replace save, the writer clears the session's old projection rows, writes a fresh parser-parent `session_links` row, and invokes `_resolve_session_graph`. If the parent already exists, classification happens before row extraction. If the parent arrives later, `_resolve_session_graph` resolves the inbound link and calls `_reextract_prefix_tail_db`, which now repeats the membership decision. Existing `session_links` therefore do not freeze the old classification during a complete reparse/rebuild.

The current schema stores the relevant topology in `sessions` and `session_links`; there is no separate `topology_edges` table to query in this snapshot.

Run the following query against the old `index.db`, rebuild, and run it again against the new `index.db`:

```sql
WITH acompact AS (
    SELECT
        s.session_id,
        s.branch_type,
        l.link_type,
        l.inheritance,
        l.branch_point_message_id
    FROM sessions AS s
    LEFT JOIN session_links AS l
      ON l.src_session_id = s.session_id
     AND l.method = 'parser-parent'
    WHERE s.origin = 'claude-code-session'
      AND (
          s.native_id LIKE 'agent-acompact-%'
          OR s.native_id LIKE '%:agent-acompact-%'
      )
)
SELECT
    COALESCE(branch_type, '<NULL>') AS branch_type,
    COALESCE(link_type, '<NO-LINK>') AS link_type,
    COALESCE(inheritance, '<NULL>') AS inheritance,
    COUNT(DISTINCT session_id) AS sessions
FROM acompact
GROUP BY 1, 2, 3
ORDER BY sessions DESC, 1, 2, 3;
```

Expected movement for the measured corpus, not verified in this container:

- total acompact session count remains about 187;
- about 148 true main-session artifacts remain `continuation / continuation / prefix-sharing`;
- about 39 Task self-compactions move to `sidechain / sidechain / spawned-fresh`;
- the 9 zero-overlap artifacts may already have had `spawned-fresh` inheritance from zero contiguous alignment, but their branch/link topology must still move from continuation to sidechain;
- partial-overlap Task artifacts that previously stole a leading prefix must retain their complete physical child transcript after rebuild.

Post-rebuild integrity query:

```sql
WITH acompact AS (
    SELECT
        s.session_id,
        s.branch_type,
        l.link_type,
        l.inheritance,
        l.branch_point_message_id
    FROM sessions AS s
    LEFT JOIN session_links AS l
      ON l.src_session_id = s.session_id
     AND l.method = 'parser-parent'
    WHERE s.origin = 'claude-code-session'
      AND (
          s.native_id LIKE 'agent-acompact-%'
          OR s.native_id LIKE '%:agent-acompact-%'
      )
)
SELECT COUNT(*) AS malformed_sidechain_acompacts
FROM acompact
WHERE branch_type = 'sidechain'
  AND (
      link_type IS NULL
      OR link_type <> 'sidechain'
      OR inheritance IS NULL
      OR inheritance <> 'spawned-fresh'
      OR branch_point_message_id IS NOT NULL
  );
```

Expected result: `0`.

To inspect any failures rather than only count them:

```sql
SELECT
    s.session_id,
    s.native_id,
    s.branch_type,
    l.link_type,
    l.inheritance,
    l.branch_point_message_id,
    l.resolved_dst_session_id
FROM sessions AS s
LEFT JOIN session_links AS l
  ON l.src_session_id = s.session_id
 AND l.method = 'parser-parent'
WHERE s.origin = 'claude-code-session'
  AND (
      s.native_id LIKE 'agent-acompact-%'
      OR s.native_id LIKE '%:agent-acompact-%'
  )
  AND s.branch_type = 'sidechain'
  AND (
      l.link_type IS NULL
      OR l.link_type <> 'sidechain'
      OR l.inheritance IS NULL
      OR l.inheritance <> 'spawned-fresh'
      OR l.branch_point_message_id IS NOT NULL
  )
ORDER BY s.native_id;
```

## Apply order

1. Confirm the target checkout contains commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d` or an understandable descendant.
2. From the repository root, run `git apply --check PATCH.diff`.
3. Apply with `git apply PATCH.diff`.
4. Run the commands in `TESTS.md`.
5. Stop any writer/daemon according to normal operator procedure, back up the archive, run the index reset/rebuild commands above, and compare the pre/post SQL output.
6. Inspect any malformed rows and sample composed reads for both classes before promotion.

## Verification completed

The implementation worktree passed:

- 56 focused parser/dispatch/normalization tests.
- 91 writer and lineage-normalization tests.
- 42 additional Claude history/dispatch/artifact route tests.
- 1 targeted delegation regression.
- Total: 190 non-overlapping tests.
- Ruff lint on all changed Python source/tests.
- Ruff format check on changed Python source/tests.
- Mypy on all three changed production modules.
- `git diff --check`.

The generated patch was also checked and applied in a detached pristine worktree at the exact base commit. A representative eight-case archive normalization test selection passed there with `PYTHONPATH` pointing at the applied worktree, confirming the patch does not depend on untracked implementation files.
After a delivery ZIP candidate was built, its extracted `PATCH.diff` was independently applied to a second fresh detached worktree at the same commit; eight targeted acompact/fresh-head tests passed from that extracted patch, and module inspection confirmed imports came from the fresh worktree.

Exact commands and results are in `TESTS.md`.

## Risks and remaining verification

- The live 37 GB corpus, operator daemon, secrets, and deployed archive were not available. The measured ~39-row reclassification and the SQL outputs remain operator verification.
- The complete repository test suite was not run. Focused tests cover the changed parser, both dispatch routes, both writer ingest orders, composition, lineage normalization, and delegation, but unrelated platform/integration lanes remain unverified.
- The 90% boundary is bead/corpus authority, not a newly recomputed statistic in this container.
- Content signatures intentionally ignore UUID equality and use normalized content membership because the known replay shapes may carry fresh message UUIDs. The parser's fresh-head predicate uses UUID topology (`parentUuid`) and Task identifiers where available.
- Ambiguous one-file parser consumers that do not persist through the archive writer can temporarily see `continuation` until a parent comparison is possible. Archive-backed production reads receive the authoritative classification.
- The snapshot's unexplained `dirty=true` flag cannot be reconstructed from an empty exported delta. The patch is therefore guaranteed against the named commit, not unknown external local edits.

Another iteration would add modest value unless it has access to the live archive: run the rebuild, capture exact before/after class counts, inspect the full set of flipped records, and add a corpus-derived distribution report. A substantial second implementation pass is not indicated by the repository tests; the highest-value remaining work is deployment/corpus verification rather than a different code design.
