# EVIDENCE — polylogue-j2zz

## Snapshot and identity evidence

- `polylogue-overview.md` reports generation at `2026-07-17T18:09:50Z`, source `/realm/project/polylogue`, branch `master`, commit `536a53ef`, and `dirty=true`.
- `polylogue-branch-delta.md` names `origin/master` and merge base `536a53efac0cbe4a2473ad379e4db49ef3fce74d`; its diff stat and commit list are empty.
- `polylogue-manifest.json` records the all-refs bundle and working-tree archive checksums and records the branch-delta patch/file/log artifacts as zero-byte files.
- Git reconstruction resolves `HEAD`, `master`, and `origin/master` to `536a53efac0cbe4a2473ad379e4db49ef3fce74d`.

Interpretation: the snapshot marks local dirtiness, but supplies no branch patch or branch-only commit to layer. The current source archive remains authority for file contents; the deliverable patch is based on the named clean commit and was validated there.

## Bead evidence

The full `polylogue-j2zz` record says:

- priority 1, open bug, title “Lower Codex orchestration child calls into typed actions”
- newest 100-session sample: every session has nested calls, 14,004 envelopes contain child operations, 19,180 results produce zero structured paths/outcomes, and 1,444 texts contain `exit_code`
- design: retain outer transport; add a typed registry for `exec_command`, `apply_patch`, `write_stdin`, `update_plan`, `wait`, web, image, MCP, and unknown; promote structural fields only; preserve order/repeats; feed the relation owned by `polylogue-z9gh.2`
- acceptance: single/multiple fixtures, normalized commands/patch paths, structural-or-unknown outcomes, raw malformed/unknown evidence, deterministic repeated/continuation pairing, live coverage report, outer-only zero baseline, parser/action tests, and quick gate

Relevant relations/dependencies inspected:

- `polylogue-z9gh.2`: current bounded `action_pairs` / `delegation_facts` implementation and compatibility views
- `polylogue-9l5.6`: tool episodes require every use to survive as paired or unknown and explicitly reject fabricated recovery evidence
- `polylogue-2qx`: OriginSpec source-admission program notes that `j2zz` is a provider regression slice that must retain normalized provenance rather than become a surface-only parser hack

## Current source evidence

### Parser and dispatch

`polylogue/sources/parsers/codex.py` previously emitted only one outer tool-use block for a function/custom tool call and one outer result block. It normalized outer execution arguments into `command`, but had no child registry or nested operation blocks. Outer result parsing recognized exact metadata exit code in a narrow JSON shape.

`polylogue/sources/dispatch.py` owns provider detector tightness. This patch changes no detector, ordering, payload lowering, or provider admission path.

### Structural outcome contract

`AGENTS.md` calls `blocks.tool_result_is_error` and `blocks.tool_result_exit_code` the provider-reported outcome keystone and states that null means unknown, never regex-guessed from prose. The implementation follows that constraint: text token counts are diagnostics only.

### Content hash

`polylogue/pipeline/ids.py` lines 78-105 build each message hash from content blocks. A block contributes type and text plus tool name, tool id, tool input, and media type when present. Session hashes include all message payloads. Additional child blocks therefore alter affected Codex session hashes.

### Action relation and writer route

`polylogue/storage/sqlite/archive_tiers/index.py` defines:

- `blocks.tool_command` as `json_extract(tool_input, '$.command')`
- `blocks.tool_path` as `COALESCE(file_path, path)`
- materialized rebuildable `action_pairs` keyed by tool-use block id with command, path, result id/text, `is_error`, and `exit_code`
- indexes for session order, message, tool, semantic type, path, and outcome
- `actions` as a simple compatibility projection over `action_pairs`

`polylogue/storage/sqlite/action_pairs.py` ranks uses and results by message position, variant, and block position inside `(session_id, tool_id)` and joins equal rank. Null/empty ids remain unpaired.

`polylogue/storage/sqlite/archive_tiers/write.py` calls `refresh_action_pairs` after the relevant append and full-replacement block writes. Therefore parser-emitted child blocks reach the production relation without new DDL or a parallel index path.

### Claim-vs-evidence

`devtools/claim_vs_evidence.py` constructs the failure population from structured result rows and publishes the predicate `tool_result_is_error = 1 OR tool_result_exit_code != 0`. It then examines the next assistant turn. Missing nested Codex outcomes therefore undercount the eligible failure frame; lowering them expands evidence but does not perform the follow-up classification itself.

## History inspected

- `9163d0134f3d334960e4c249c96c5671919a9a06`, PR #3018, `feat(query): bound agent-facing archive reads`: introduced indexed rebuildable `action_pairs`/`delegation_facts`, compatibility views, scoped refresh, and duplicate-id pairing. It is the direct parent of the snapshot commit.
- `d42cc1497ee91fded8c46313a46e18733f9084ee`, `test(query): prove action cardinality composition`: real-route action cardinality proof for duplicate tool ids and missing results.
- `219869f660358df09946b726a0ca213a1e0c43bf`, `fix(actions): expose Codex exec payloads as commands`: established provider-neutral `command` promotion for outer Codex execution payloads and a parser-to-query test.
- `13d19ae36c2bfbc60d1197390573f5beed08e953`, `fix(actions): read legacy Codex commands without rewriting evidence`: preserved read compatibility for historical provider-native command fields without rewriting evidence.

These commits establish that child lowering belongs at parser/block normalization, while pairing and query selectivity remain owned by the existing derived relation.

## Current Codex wire evidence inspected

Official OpenAI Codex issue/source evidence shows current Code Mode calls represented as outer `functions.exec`/`exec` envelopes containing JavaScript calls such as `tools.exec_command(...)`, with tool results serialized through structured text/content items. Current protocol source also permits custom tool outputs as a plain string or an array of structured content items. That evidence motivated support for both exact JSON values and ordered content-item arrays, while leaving status/header prose unpromoted.

No live private archive or daemon was accessed. The 14,004/19,180/1,444 figures are supplied bead evidence and were not independently reproduced.

## Contradictions and resolutions

1. Root `AGENTS.md` still says `actions` is a view that directly joins tool-use/result blocks. Current source after PR #3018 has a materialized `action_pairs` table and `actions` is only a simple compatibility view. Current source/history supersede the stale architecture sentence.
2. The snapshot overview says `dirty=true`, but branch-delta patch/files/commits are empty and HEAD equals `origin/master`. The package does not invent an unavailable dirty patch; it targets the named commit and records this limitation.
3. Older prework or plans may name globally ranked action views. PR #3018/current source replaced them with bounded per-session materialization; this patch feeds that current route.
4. The mission quotes live coverage counts, but the environment contains no authorized corpus. The census implementation and fixture receipt are verified; live N/M numbers remain an integrator responsibility.

## Evidence boundary

This package proves parser behavior, occurrence pairing through the production refresh SQL, fixture census output, content-hash change, patch applicability, and Python compilation. It does not claim managed-gate success, live-scale performance, live archive coverage, or daemon/deployment verification.
