# HANDOFF — polylogue-j2zz

## Mission and result

This package implements the Codex Code Mode lowering requested by bead `polylogue-j2zz`. Modern outer `exec` / `functions.exec` calls remain stored as transport evidence, while statically recoverable nested child operations become additional typed `tool_use` / `tool_result` blocks. Those blocks flow through the existing block writer and the rebuildable `action_pairs` relation, so action queries can see both the envelope and the operations it transported.

The implementation is deliberately conservative. It parses JSON-like JavaScript literals and exact structured result values; it does not evaluate JavaScript, resolve references, or infer outcomes from prose. Dynamic and malformed argument expressions remain raw evidence with `parse_state=malformed`. Unknown tool names remain typed as `unknown` with their exact name/path and arguments.

## Snapshot identity

The supplied Chisel snapshot reports:

- source repository: `/realm/project/polylogue`
- branch: `master`
- commit: `536a53efac0cbe4a2473ad379e4db49ef3fce74d`
- generated: `2026-07-17T18:09:50Z`
- snapshot dirty flag: `true`
- branch base and merge base: `origin/master` at the same commit
- branch delta patch, changed-file list, and branch-only commit list: empty

Authority was reconstructed from `polylogue-all-refs.bundle` and `polylogue-working-tree.tar.gz`. The current working-tree source was inspected for the relevant routes. Because the supplied branch-delta artifacts contain no patch to preserve, `PATCH.diff` is intentionally based on the clean named commit above. Apply validation was performed in a fresh detached worktree at that exact commit.

## Inspected authority

The implementation followed these production routes and constraints rather than inventing a parallel model:

- full bead `polylogue-j2zz`, including dependencies on/relations to `polylogue-z9gh.2`, `polylogue-9l5.6`, `polylogue-2qx`, and `polylogue-2qx.1.2`
- root `AGENTS.md`, especially structural outcome discipline, semantic content hashing, source detector tightness, and derived-tier rebuild rules
- `polylogue/sources/parsers/codex.py` and Codex provider records
- `polylogue/sources/dispatch.py` detector ordering; it is unchanged
- `polylogue/pipeline/ids.py` block/session hashing
- `polylogue/storage/sqlite/archive_tiers/index.py` block generated columns, `action_pairs`, indexes, and `actions` compatibility view
- `polylogue/storage/sqlite/action_pairs.py` session-scoped Nth-use/Nth-result materialization
- `polylogue/storage/sqlite/archive_tiers/write.py` refresh call sites after append and full replacement
- action query lowering and the existing parser-to-`ArchiveStore.query_actions` Codex command test
- `devtools/claim_vs_evidence.py` structured-failure frame and predicate
- history for PR #3018 and the earlier Codex command compatibility changes

## Mechanism

### Envelope discovery and preserved transport

`_parse_records` now pre-scans the complete record sequence for modern Code Mode call/output records. A call is eligible only when the outer tool name is exactly `exec` or `functions.exec`. The ordinary outer tool message is still emitted first and unchanged in role: its `tool_use` block remains block position zero and its outer result remains the first `tool_result` block.

For eligible calls, the pre-scan extracts nested operations from either:

1. a structured child collection such as `calls`, `tool_calls`, `children`, `operations`, `actions`, or `invocations`; or
2. a JavaScript source-bearing field such as `source`, `code`, `script`, `javascript`, `js`, `command`, `arguments`, or `input`.

The JavaScript scanner recognizes `tools.*` and `functions.*` member calls, including quoted bracket members. It skips strings and comments, records source spans, and parses only JSON/JavaScript literals. References, spreads, interpolation, shorthand objects, and unbalanced calls are retained as raw malformed arguments rather than evaluated.

### Registry and field promotion

| Child registry type | Stored child tool name | Call-side promotion | Result-side promotion | Outcome rule |
| --- | --- | --- | --- | --- |
| `exec_command` | stable `exec_command` | Preserve exact arguments. Normalize `command`, `cmd`, a literal string, or a string argv list into `command`; argv uses shell-safe `shlex.join`. Preserve exact structural path/byte fields when present. | Preserve exact result text, structural paths, and byte count. Result-derived path/byte fields are copied into the child use input so existing generated `tool_path`/action columns can index them; provenance identifies them as result-derived. | Exact integer `exit_code` and exact boolean `is_error`; nonzero exit derives `is_error=true`, zero derives `false` only when `is_error` was absent. |
| `apply_patch` | stable `apply_patch` | Preserve patch text and expose it as normalized `command`. Extract touched paths only from patch structure: `*** Add/Update/Delete File`, move markers, and unified-diff headers. | Preserve structural paths and byte count. | Same exact structural rule. |
| `write_stdin` | stable `write_stdin` | Preserve exact continuation/session/input mapping; no command is invented. | Preserve structural paths/bytes if the provider emits them. | Same exact structural rule; status prose or status-only fields remain unknown. |
| `update_plan` | stable `update_plan` | Preserve exact plan structure. | Preserve structural paths/bytes if present. | Same exact structural rule. |
| `wait` | stable `wait` | Preserve exact wait/cell/process identifiers. | Preserve structural paths/bytes if present. | Same exact structural rule; completion status alone is not promoted to success. |
| `web` | stable `web` | Preserve exact query/request arguments. | Preserve structural paths/bytes if present. | Same exact structural rule. |
| `image` | stable `image` | Preserve exact image arguments and structural paths. | Preserve structural image/file paths and byte count. | Same exact structural rule. |
| `mcp` | exact provider child path, for example `mcp.repo_memory.search` | Preserve exact tool path/name and arguments; provenance supplies registry type `mcp`. | Preserve structural paths/bytes if present. | Same exact structural rule. |
| `unknown` | exact provider child name, or `unknown` when absent | Preserve exact decoded arguments plus `raw_arguments`; malformed values retain source text and `parse_state=malformed`. No command is guessed. | Preserve exact result evidence and structural fields where present. | Unknown unless an exact structural outcome field is present. |

Structural path keys are `path`, `file_path`, `paths`, `file_paths`, and `image_path`. Structural byte keys are `bytes`, `byte_count`, `bytes_written`, `size_bytes`, and `written_bytes`. Exact outcomes are read from the result mapping itself or direct `metadata`, `result`, or `output` mappings. Text containing strings such as `exit_code=7` never changes `is_error` or `exit_code`.

Every child use carries `_polylogue` provenance with:

- `kind=codex.functions_exec_child`
- registry type and parse state
- exact raw tool path
- child ordinal
- parent transport provider message id, tool id, tool name, and block position
- JavaScript source span when available
- exact result-derived paths/byte count when those were projected onto the use block

### Ordering and pairing

Ordering is stable and evidence-preserving:

1. outer transport block
2. child use blocks in JavaScript/structured collection order
3. outer transport result block
4. child result blocks in emitted structured-result order

With transport id `T`, child index `i` receives `T::polylogue-child::i`. Repeated outer records may legitimately reuse `T`; the existing `action_pairs` materializer ranks uses and results by transcript order within `(session_id, tool_id)`, so occurrence N pairs with occurrence N rather than forming a cross product.

The parser independently ranks call and output records by outer call id before enriching them. This preserves deterministic pairing even when source records place a result before its call. Within one envelope, child index N pairs only with structured result item N. Missing result items produce no child result block, leaving the child action explicitly unpaired. A `write_stdin` continuation is represented as its own ordered child action and result; the parser does not label it as recovery or infer success from a later message.

### `action_pairs` integration

No new storage API, DDL, durable field, or index schema version is introduced. The child blocks use the existing `ParsedContentBlock` contract. The current writer already writes block outcomes, derives `tool_command` from `tool_input.command`, derives `tool_path` from `tool_input.file_path`/`path`, and calls `refresh_action_pairs` after both append and full-replacement writes. Therefore the new blocks automatically create bounded `action_pairs` rows and remain visible through the simple `actions` compatibility view.

Two existing semantic aliases were completed so the new stable names are useful through current read models:

- `apply_patch` classifies as `file_edit`
- `wait` classifies as `agent`

### Content-hash and re-ingest impact

This lowering **does alter Codex session content hashes**. `pipeline/ids.py` hashes message content blocks, including block type, text, tool name, tool id, and tool input. Adding child blocks and provenance therefore changes the hash for every session containing a lowerable Code Mode envelope. The included test proves the lowered and outer-only projections hash differently.

Expected operational effect on the next ingest:

- affected Codex sessions are not skipped as hash-identical; they are reparsed and rewritten
- each affected envelope adds one child use per recognized operation and up to one child result per exact emitted result item
- `action_pairs` is refreshed through the normal session-scoped writer route
- dependent rebuildable insights may be recomputed under the normal convergence route
- source/user durable tiers require no migration or rewrite
- Codex sessions without lowerable envelopes retain their prior parsed block shape and should remain hash-stable

The cost is proportional to the number and size of affected Codex sessions plus the added block/action cardinality. The supplied live counts establish 14,004 candidate envelopes, but the exact child multiplier and resulting write cost remain unknown until the integrator runs the census against the authorized corpus.

## Claim-vs-evidence interaction

`devtools/claim_vs_evidence.py` anchors its failure frame on structured tool results using the predicate `tool_result_is_error = 1 OR tool_result_exit_code != 0`. Before this change, nested Codex failures represented only inside Code Mode result items do not enter that frame because the archive has only the outer transport result with unknown outcome.

This patch fixes the **undercount of structurally evidenced nested Codex failures** and the resulting Codex/provider/model/tool composition bias in that frame. It expands the eligible evidence population with exact child `is_error`/`exit_code` fields. It does not infer failures from prose, decide whether the next assistant turn acknowledged or recovered from the failure, prove tool utility or intent, or retroactively change any published report. Existing reports change only after semantic re-ingest and regeneration of the analysis.

## Census script

Run the report against one or more Codex JSON/JSONL/NDJSON files, gzip variants, or directories:

```bash
nix develop --command python devtools/codex_exec_child_census.py \
  /path/to/codex/session-corpus \
  --strict \
  --output /tmp/codex-exec-child-census.json
```

The report contains:

- discovered/parsed file counts and parse errors
- outer transport action/result counts
- typed children by registry and parse state
- paired, unpaired, and orphan child result counts
- command/path/byte coverage
- exact structured outcome/exit-code coverage
- diagnostic-only counts of result texts containing the token `exit_code`
- an explicit outer-only counterfactual with zero typed child/path/outcome coverage

Fixture command used here:

```bash
PYTHONPATH=/mnt/data/work-j2zz/stubs:$PWD /opt/pyvenv/bin/python \
  devtools/codex_exec_child_census.py \
  tests/data/codex_event_stream/functions_exec_single.jsonl \
  tests/data/codex_event_stream/functions_exec_multiple.jsonl
```

Fixture result: 2 transport actions, 11 typed children, 11 paired child results, 3 child paths, 3 structured outcomes, 8 unknown outcomes, 100% pairing coverage, and 27.2727% path/outcome coverage. The outer-only counterfactual reports zero typed children, paths, and structured outcomes.

The authorized integrator should run the same script against the raw live session corpus and record the requested `14,004 envelopes -> N typed children` and `0 -> M structured outcomes` receipt. That live execution was not available in this environment.

## Changed files

| File | Change |
| --- | --- |
| `polylogue/sources/parsers/codex.py` | Conservative JS/structured child registry, provenance, ordering, structural result parsing, deterministic envelope pairing, and child block emission. |
| `polylogue/archive/viewport/tools.py` | Query semantic categories for `apply_patch` and `wait`. |
| `devtools/codex_exec_child_census.py` | Snapshot-runnable corpus census and outer-only counterfactual. |
| `tests/data/codex_event_stream/functions_exec_single.jsonl` | Single-child transport/result fixture. |
| `tests/data/codex_event_stream/functions_exec_multiple.jsonl` | All registry types, paths/bytes, unknown/malformed, and prose-token fixture. |
| `tests/unit/sources/test_codex_event_stream_contract.py` | Parser, pairing, action materialization, unknown outcome, result-order, and hash tests. |
| `tests/unit/devtools/test_codex_exec_child_census.py` | Census exact-output/anti-vacuity test. |
| `tests/unit/cli/test_query_expression.py` | Existing real parser-to-archive action query now requires both transport and child command rows. |
| `tests/unit/sources/test_tool_aliases.py` | Stable semantic category checks. |

No existing test/helper is deleted. No dominated deletion is proposed. No `FILES/` replacements are needed because the unified diff is unambiguous.

## Acceptance matrix

| Acceptance requirement | Implementation/proof | Status |
| --- | --- | --- |
| Retain outer transport and lower single/multiple ordered children | Two fixtures plus exact block-name/id/order assertions | Implemented; focused tests passed |
| Registry covers exec, patch, stdin, plan, wait, web, image, MCP, unknown | Multiple fixture asserts every registry type and stable/exact names | Implemented; passed |
| Normalize commands and patch paths | String/argv command normalization and structural patch-header parser | Implemented; passed |
| Outcomes are structural or unknown; no prose guessing | Exact mapping parser; prose `exit_code=7/9` assertions remain null | Implemented; passed |
| Malformed/unknown retain evidence | Raw arguments, raw tool path, source span, typed parse state | Implemented; passed |
| Repeated calls and continuations pair deterministically | Reused transport-id test executes production `refresh_action_pairs`; continuation is an ordered child; missing result stays unpaired | Implemented; passed |
| Result-before-use does not invent recovery | Explicit output-before-call fixture | Implemented; passed |
| Children feed bounded `action_pairs` and actions view | Existing block/write/materialization route; direct production SQL test and real archive query assertion | Implemented; SQL test passed; full ArchiveStore test unverified locally |
| Content-hash impact explicit | Hash-difference test and statement above | Implemented; passed |
| Census reports before/after coverage | Standalone script and exact fixture report | Implemented; passed |
| Live 100-session receipt | Script supplied for integrator | Unverified: live archive/corpus unavailable |
| Quick verification gate | Exact command listed below | Unverified: locked development dependencies unavailable in container |

## Apply and verification order

```bash
git checkout 536a53efac0cbe4a2473ad379e4db49ef3fce74d
git apply --check PATCH.diff
git apply PATCH.diff

nix develop --command devtools test \
  tests/unit/sources/test_codex_event_stream_contract.py \
  tests/unit/sources/test_tool_aliases.py \
  tests/unit/devtools/test_codex_exec_child_census.py \
  tests/unit/cli/test_query_expression.py::TestBooleanQueryExpression::test_codex_exec_freeform_arguments_are_queryable_as_commands

nix develop --command devtools verify --quick
```

Then run the census on the authorized raw Codex corpus, ingest/reprocess affected Codex sessions through the normal writer/convergence route, and verify representative action queries for child tool, path, and nonzero outcome. No durable migration command is required. If the deployment replaces the rebuildable index tier instead of incrementally re-ingesting, use its canonical index rebuild route; do not add a migration chain for this patch.

## Risks and boundaries

- The JavaScript parser intentionally supports literals, not arbitrary JavaScript evaluation. Dynamic variables and computed expressions are retained as malformed raw evidence; this favors false negatives over fabricated semantics.
- Unrecognized multi-part namespaces are typed as MCP except first-class `web` and `image` namespaces. Live census registry distribution should reveal provider shapes that deserve a later explicit alias.
- Child result pairing assumes Code Mode emits child results in child-call order. This matches the inspected current evidence and the bead contract; the parser does not correlate by prose or guessed command text.
- A missing exact result item stays unpaired. Later success/failure text is not treated as recovery.
- Child IDs are deterministic from transport id and child index. When transport ids repeat, correctness depends on the existing `action_pairs` occurrence ranking, which is directly exercised.
- Parent transports without a provider tool id still produce child uses with full parent provenance, but their child result ids remain null and therefore intentionally unpaired under current action semantics.
- Semantic re-ingest may be material on the live corpus because every added block creates derived action/index work.

## Verification performed and remaining

Performed against a fresh apply of `PATCH.diff` to the exact snapshot commit:

- `git apply --check PATCH.diff`: passed
- `git apply PATCH.diff`: passed
- `git diff --check`: passed
- Python compile/compileall for all changed Python files: passed
- line-length audit for all changed Python files: zero lines over 120 characters
- focused parser/alias/action-pair/census suite: **34 passed**, with three warnings caused by disabled optional pytest plugins (`asyncio_mode`, `timeout`, `timeout_method`)
- fixture census: completed with no parse errors and the exact counts documented above

The focused run used Python 3.13.5 and import-only shims outside the patch for container-missing dependencies. The tested behavior does not call those shim implementations. The full managed environment could not be reconstructed because `uv` could not obtain locked packages and no Nix command was available. Consequently these remain unverified here:

- `devtools verify --quick` (format, Ruff, strict mypy, render/topology/layering/policy gates)
- the modified full `ArchiveStore` query test, whose local collection/runtime was blocked by absent `hypothesis` and `sqlite_vec`
- live archive ingest/rebuild, daemon behavior, authorized archive queries, and the 100-session census

A further iteration is likely a small repair if the managed quick gate exposes only formatting/type issues. A substantial second implementation pass is justified only if the live census shows materially different Code Mode call/result shapes, namespace conventions, or result ordering that the conservative registry does not yet recognize.
