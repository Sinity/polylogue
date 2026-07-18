# EVIDENCE — polylogue-4ts.3

## Inputs and identity

| Evidence | Value |
|---|---|
| Project-state archive | `/mnt/data/polylogue-all.tar(125).gz` |
| Project-state archive SHA-256 | `ec3fa193b2f99a6daee0bc620af4f69b5728834e99f8196792a17c3fbae11155` |
| Project-state archive size | `128314788` bytes |
| Mission file | `/mnt/data/lin-03-subagent-compaction(1).md` |
| Mission file SHA-256 | `4634ed5425fb122221f736b2f5ae167c2e87d4decaf535a2771abdde425f95b8` |
| Mission file size | `7925` bytes |
| Snapshot branch | `master` |
| Snapshot commit | `536a53efac0cbe4a2473ad379e4db49ef3fce74d` |
| Snapshot source | `/realm/project/polylogue` |
| Snapshot generated | `2026-07-17T18:09:50Z` |

The snapshot metadata records `dirty=true`, while all exported branch-delta artifacts are empty and the merge base is the same commit. The extracted checkout was clean before implementation. This is an unresolved packaging contradiction, not evidence of a hidden patch that can be reconstructed.

## Bead authority

The current `.beads/issues.jsonl` record for `polylogue-4ts.3` states:

- `agent-acompact-*` is overloaded: about 39 of 187 files have less than 90% overlap with the main session; 9 have 0% overlap.
- The parser's unconditional main-parent classification corrupts composed lineage.
- Detection must use prefix content/UUID membership or a fresh Task-prompt head.
- A mismatch must become a fresh sidechain with no inherited prefix.
- Required regressions are one true-main acompact and one Task self-compaction.
- Acceptance includes an explicit composed-read assertion that the main transcript is not prepended.

The later task packet repeats this acceptance and says the Bead wins over conflicting issue-thread wording. The mission further requires rebuild verification and a streaming/eager sibling-route audit.

## Source findings

### Claude parser

File: `polylogue/sources/parsers/claude/code_parser.py`.

Before the patch, artifact identity was established with `fallback_id.startswith("agent-")` / `fallback_id.startswith("agent-acompact-")`. Any agent artifact with a `sessionId` received that `sessionId` as `parent_session_provider_id`; every acompact then received `BranchType.CONTINUATION`. No record-content or UUID-topology evidence participated in the class decision.

The parser receives an iterator for one grouped artifact. In common standalone sidecar ingestion, it has no access to the separate main-session transcript, so a general parent-membership test cannot be performed there without adding cross-file state or buffering unrelated files. This is why the patch uses a positive fresh-head signal in the parser and defers ambiguous authority to persistence.

### Eager and streaming dispatch

File: `polylogue/sources/dispatch.py`.

Both eager and memory-bounded Claude Code paths ultimately call the same provider parser (`claude.parse_code` / `claude.parse_code_stream` over the same parsing implementation). The classifier was not duplicated.

A real sibling divergence did exist: `_claude_code_stream_sessions` preserved the original fallback for the first grouped session, while `_claude_code_grouped_record_specs` replaced every group fallback with its `sessionId`. On an eager aggregate whose first source file is named `agent-acompact-*`, that erased the very artifact identity used by the classifier. The patch aligns eager behavior with streaming behavior and locks full normalized-model equality in a regression.

### Writer and graph resolution

File: `polylogue/storage/sqlite/archive_tiers/write.py`.

The parent-known write route already queried the parent's composed signatures and called `_extract_prefix_tail`. That routine performs strict contiguous prefix alignment and returns either `prefix-sharing` with a tail or `spawned-fresh` with the child whole. It did not alter a wrong continuation branch/link type, and a small coincidental leading match could still be physically removed.

`_write_session_link` stores parser-parent assertions in `session_links`. `_resolve_session_graph` runs on every save, resolves outbound links, finds children waiting for the newly saved parent, and invokes `_reextract_prefix_tail_db`. That delayed route is the required child-before-parent twin.

The patch inserts the same membership rule before extraction in both routes. It changes branch/link topology as well as inheritance, so zero-overlap artifacts no longer remain mislabeled continuations and partial-overlap artifacts cannot donate a false branch point.

### Composition

Archive session reads compose inherited prefixes only through prefix-sharing lineage with a resolved branch point. Persisting Task self-compactions as sidechain/spawned-fresh with a null branch point therefore makes “no main prefix” a structural property, not a presentation-layer filter.

### Schema vocabulary

The current repository already expresses both outcomes:

- `sessions.branch_type`: `continuation` or `sidechain`;
- `session_links.link_type`: mapped topology edge type;
- `session_links.inheritance`: `prefix-sharing` or `spawned-fresh`;
- `session_links.branch_point_message_id`: populated only for an inherited prefix.

No schema change or new topology vocabulary is needed. The mission mentions probing `session_links/topology_edges`, but this snapshot has no separate `topology_edges` table; `session_links` is the persisted edge authority used by reads and graph projection.

## History findings

- `7412b69a9b64863a1bf4df11d7feeb99f61262ff` introduced the `agent-acompact-*` continuation classifier under the then-current assumption that the artifacts were parent copies.
- `a651bb95f1fc3595cff8f545ec3768831b858daf` introduced/merged the memory-bounded Claude stream route, making route parity a required audit point.
- `aa4a520b44861220c5ad16370b4c99634119e98d` recorded the later investigation showing the mixed corpus shape.

The current source therefore supersedes the older design sentence claiming all 187 acompacts are 100% parent copies.

## Contradictions resolved

1. **Old design vs measured corpus.** `docs/design/session-lineage-model.md` said every acompact was a 100% copy plus summary. The Bead and later corpus evidence say ~39 are Task self-compactions. The patch updates the design statement.
2. **Parser-only wording vs production data availability.** The Bead says fix the classifier and permits either membership or a fresh Task head. A standalone parser cannot compare a separate parent file. The implementation uses the permitted structural head signal there and performs authoritative membership in the writer, the first route that can safely see both sessions.
3. **“Twin classifier” concern vs actual code.** The classifier is shared; the twin defect was fallback propagation. Both routes are now compared end-to-end.
4. **Snapshot dirty flag vs empty delta.** No dirty bytes were provided. The named commit is the only applyable authority.
5. **Probe wording vs schema.** There is no `topology_edges` table in this snapshot. The delivered queries use `sessions` + `session_links`, which are the actual production tables.

## Evidence deliberately not claimed

- No live corpus files were opened beyond the synthetic/public repository fixtures and archived textual investigation evidence.
- No operator `index.db` was rebuilt or queried.
- The exact post-rebuild counts are expected from the recorded corpus distribution, not observed in this container.
- No daemon, deployment, browser, external archive, or secrets were available.
