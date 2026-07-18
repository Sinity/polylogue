# EVIDENCE — authority, source findings, and design resolutions

## Mandate authority

The attached mandate requires implementation of `polylogue-2qx.2`, use of the existing OriginSpec/admission path, raw revision preservation, generic work-evidence materialization, evidence-backed membership, the exact `wf_54d4fb2e-841` corpus, a quantified reparse plan, and a cohesive apply-ready package.

The mandate explicitly rejects a Workflow-only archive or second registry. It also requires missing members to become degraded/unresolved rather than fabricated and requires the 38 unrelated coordinator children to remain excluded.

## Bead evidence

### `polylogue-2qx.1.2`

Status in the authority snapshot: open.

The record requires the current origin vocabulary to converge on OriginSpec and states that the Claude/Codex extension hooks are the only admission path for `2qx.2`; no private inventory may be introduced. This constrained the implementation to add artifact-family declarations and projections to the existing `polylogue/sources/origin_specs.py` contract.

### `polylogue-2qx.2`

Status in the authority snapshot: open.

The record states that the prior Claude source retained only attempt transcripts as ordinary subagent sessions, acquired the journal without parsing it, and missed sidecars, run-state JSON, and the adopt manifest. Its seven acceptance criteria specify the exact artifact family, counts, evidence fields, unrelated-child exclusion, authorship correction, missing-member behavior, and reparse quantification.

The latest note dated 2026-07-18 says the prior claim was orphaned because the claiming session was closed and no matching commits existed on master, then resets the Bead to open. This supersedes any older note that might imply the work was already complete.

## Snapshot and history evidence

The supplied repository authority resolves to:

- `master` / `origin/master`;
- commit `bf8191b3f56aa40da8f271df7f3385c712825497`;
- parent `4b574ce66533ac114961a9533ba2f2c4a0c45b83`.

The supplied worktree carried unrelated modifications only in `polylogue/archive/query/unit_results.py`, `polylogue/daemon/http.py`, and `polylogue/hooks/__init__.py`. Their combined binary-diff SHA-256 is `f30102453e3576c9049eb65dcea7265fa99ac694c45531b06b0a34dd85279e9f`. They are not present in the delivered patch.

Relevant history includes `bce7336d3bb2e493080b37fd0bc76b429b0c1cbd`, titled `feat: harden archive continuity closure (#3051)`. That change established the OriginSpec and generic work-evidence substrate. The present patch extends those mechanisms instead of replacing them.

## Source findings

### Origin and admission

`polylogue/sources/origin_specs.py` already held the authoritative origin registry and several Claude Workflow artifact rules, but it did not expose a suffix projection for the live watcher and did not distinguish coordinator streams as a complete sixth family.

Configured session parsing naturally skipped valid non-session `.json` facts. The direct configured archive route therefore needed an explicit OriginSpec-classified raw-retention pass for non-session artifacts. The live route had a separate defect: the default Claude watcher admitted JSONL only, and the live full-ingest path treated a valid fact artifact that parsed to zero sessions as failed.

### Source-tier revision semantics

`source.db` already preserves raw observations in `raw_sessions` and blobs. `raw_artifacts` is the current inventory/pointer surface. Canonical configured acquisition populated it, but direct/live raw writes could leave current pointers absent or stale. The materializer now repairs only this current inventory from the newest retained Claude raw observation, while preserving every historical `raw_sessions` row.

### Parsing and authorship

The normal Claude parser already emits session messages and material-origin values. Coordinator Workflow tool use can therefore remain a regular session event. The orchestration fact parser existed but retained too few provider fields and did not expose enough association information for robust sidecar/transcript linking or explicit parse errors.

### Generic work-evidence

The generic graph model and tables already represented runs, invocations, calls, attempts, session segments, results, claims, edges, association states, corpus snapshots, and evidence arrays. The missing substrate capability was a typed way to cite retained raw artifact revisions in addition to message/session `EvidenceRef`s. Extending the union and SQLite decoder was sufficient; a new schema or Workflow table was unnecessary.

### Production projection

A Claude Workflow projector existed, but it was prototype-level: it accepted simplified inputs, did not have a complete source/index materializer, and could not prove current raw revisions, exact archive counts, live acquisition, atomic replacement, or complete provenance. No production call site satisfied the reopened Bead. The new materializer bridges existing source and index tiers and is called by configured ingestion and daemon convergence.

## Contradictions and resolutions

1. **Prototype presence versus Bead status.** Source contained a projector, but the latest Bead note reset the work to open and history contained no matching completed implementation. Resolution: preserve useful concepts, replace the incomplete API, and add production routes/tests.
2. **Fact artifacts versus session parser.** Run snapshots, journals, sidecars, and adopts are not chat sessions. Resolution: retain them as raw source authorities, parse them as provider facts during materialization, and never synthesize sessions.
3. **Filename proximity versus evidence membership.** Attempt files share naming conventions and parent-child relationships, but the mandate forbids inferred membership. Resolution: use explicit journal/meta transcript references, attempt/agent identifiers, and sidecar evidence; missing counterparts remain gaps.
4. **Current pointer versus historical revisions.** Derived projection needs one current artifact per path, while the source tier must preserve every revision. Resolution: rank observations by acquisition time/row order to update `raw_artifacts.raw_id` and leave `raw_sessions` untouched.
5. **Per-message evidence versus raw fact evidence.** Existing `EvidenceRef` cannot identify a non-session raw JSON revision. Resolution: permit only artifact `ObjectRef`s as the second evidence type and persist both formats in the generic evidence arrays.
6. **Derived freshness versus raw durability.** Rebuilding must not destroy source history. Resolution: semantic snapshots drive an atomic delete/reinsert only for graph IDs beginning `claude-workflow:`.

## Evidence-to-graph mapping

- Run snapshot raw revision -> run identity/status/workflow/script/final-result claims and run artifact node.
- Journal raw revision and source line -> content-keyed calls, attempts, retry/result state, progress/status/model/timing/token/tool claims, and unresolved call gaps.
- Sidecar raw revision -> agent/model/status/timing/token/tool/transcript association claims.
- Transcript raw revision plus indexed session/message -> session-segment membership and generated prompt material-origin claims.
- Coordinator stream indexed event/message -> invocation nodes, task/resume/workflow/script claims, and direct prompt authorship.
- Adopt manifest raw revision -> adopt/recovery claims and artifact membership.
- Corpus snapshot -> semantic version plus current raw IDs/blob hashes, coordinator events, indexed session bindings, and prompt-origin evidence.

Every produced node and edge is rejected at construction time if its evidence-ref sequence is empty.

## Quantified reparse evidence

For the exact initial fixture, `claude_workflow_reparse_plan()` reports:

- 224 current artifacts;
- 224 retained raw revisions preserved;
- 94 raw fact artifacts read by a projector-only semantic rebuild (1 snapshot + 1 journal + 91 sidecars + 1 adopt manifest);
- 4 indexed coordinator event rows reused;
- 129 indexed transcript-session bindings reused (91 linked plus 38 excluded candidates retained in the index);
- 130 session-stream raw reads only if session parsing, authorship, or event semantics change (129 transcript streams + 1 coordinator stream);
- 1 Workflow graph atomically replaced;
- `stale = false` immediately after materialization.

After a revised run snapshot, current artifacts remain 224 and retained revisions become 225. The semantic snapshot changes, `stale` becomes true before materialization, then false after one graph-family replacement.

## Private-data boundary

No private live Claude archive, operator daemon, browser profile, secrets, deployment, or current operator worktree was accessed. The exact corpus is generated synthetically in the integration test. Reparse impact is quantified from that fixture and the public schema, not from private inventory.
