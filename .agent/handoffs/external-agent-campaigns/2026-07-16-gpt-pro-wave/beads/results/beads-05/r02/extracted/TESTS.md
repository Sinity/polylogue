# Test design and execution record

## Test strategy

The tests are designed around production dependencies rather than presentation snapshots alone. Each behavior names the implementation seam that establishes it and a representative mutation or removal that makes the test fail.

| Behavior | Production dependency exercised | Anti-vacuity mutation/removal |
|---|---|---|
| Complete persisted-family policy | `semantic_type_policy_documents()` over `SemanticBlockType` and `_SEMANTIC_CARD_KIND`/fallback policy | Remove one enum row or one policy mapping; exhaustive set/card-kind assertions fail. |
| Complete executable-origin policy | `origin_policy_documents()` over `Origin` | Remove one `OriginPolicy`; the test comparing policy origins with `set(Origin)` fails. |
| Exact alias evidence | `_TOOL_MAPPINGS` and committed evidence paths | Delete/rename an alias or point evidence to a file that does not contain the tool name; registry evidence test fails. |
| Structural MCP classification | `parse_mcp_tool_name()` and `classify_tool()` precedence | Parse by substring, omit the server/tool split, or move MCP below provider aliases; MCP helper/fixture tests fail. |
| All semantic card kinds | builders in `semantic_cards.py` and the 23-case corpus | Remove a builder or redirect one kind to fallback; golden equality and all-kinds coverage fail. |
| Structural outcomes | `_outcome_from_result()` using `is_error`/`exit_code` | Infer failure from output prose or default missing fields to success; hostile success/failure/unknown fixtures fail. |
| Result serialized before use | `_pair_tool_results()` pre-indexes uses/results and records paired result coordinates | Return to a one-pass “seen use” map or fail to mark the earlier result as paired; `result-before-use` emits a duplicate orphan and fails. |
| Mixed result/context message | placement computes `paired_result_ids - independent_message_ids` | Suppress every paired result message regardless of independent prose/context; mixed-row fixture and placement assertions fail. |
| Empty-thinking compaction | typed `_THINKING_BLOCK_TYPES`, `.strip()`, pending run, explicit flush boundaries | Treat whitespace as content, compact by provider/role, or fail to flush on a typed non-thinking row; ordered notice tests fail. |
| Non-empty thinking retained | `_prose_for_block()` path for typed thinking/reasoning with content | Drop all thinking blocks or convert them to absence notices; fixture entry sequence fails. |
| Exact authoredness/variant metadata | `RenderableMessage`, `TranscriptProse`, notice sources, `SemanticCardSource` | Remove material origin, parent, variant, active-path/leaf, duration, or inherited-prefix projection; `typed-prose-metadata` and source assertions fail. |
| Metadata-backed runtime blocks | `_block_attribute_mapping()` plus metadata fallbacks in `_coerce_mapping_block()` | Remove metadata fallback for language/URL/name/media type; `test_attribute_block_metadata_retains_persisted_display_fields` fails. |
| Message-envelope attachments | `_coerce_renderable_attachments()` and `_build_attachment_envelope_card()` | Render only typed attachment blocks or remove envelope projection; attachment-envelope fixture fails. |
| Exact archive inherited-prefix evidence | `ArchiveMessageRow.source_session_id` and envelope inheritance/branch point | Remove source-session propagation or edge facts; added lineage normalization assertions fail. |
| Honest bounded lineage | session/archive descriptor builders and `_build_lineage_card()` | Invent a root/completeness value from a session row or drop archive truncation reason; lineage tests fail. |
| CLI/web structural parity | both surfaces consume `SemanticTranscript`/entry documents | Reintroduce a web-side tool classifier or attach entries only to the full-detail route; visual contract source and actual-route test fail. |
| Browser authoritative empty array | `_polySemanticEntriesForMessage()` checks `Array.isArray`, not length | Treat `[]` as “missing” and fall back to raw role/text; DOM source-contract assertion fails and paired rows regain duplicate rendering. |
| Public contract validity | both JSON Schemas plus fixture/route validation | Add an undeclared field, omit required entry discriminator, or emit an unknown kind; JSON Schema validation fails. |

## Focused automated tests executed

### Shared renderer and contract suite

Command:

```text
/opt/pyvenv/bin/python -m pytest -q -o addopts='' \
  tests/unit/rendering/test_semantic_cards.py \
  --confcutdir=tests/unit/rendering
```

Result:

```text
96 passed, 2 warnings in 2.29s
```

The warnings are `PytestConfigWarning` reports for unavailable `timeout` and `timeout_method` plugin options in this reduced runtime. They are not test failures.

This suite covers golden cards, ordered transcript entries, public schemas, exhaustive origin/family policy, structural outcomes, result-before-use pairing, mixed rows, empty-thinking runs, typed prose metadata, attachments, MCP, lineage descriptors, placement/suppression, Markdown output, and generator idempotence.

### Frozen fixture generator

Command:

```text
/opt/pyvenv/bin/python -m devtools.render_semantic_card_fixtures --check
```

Result:

```text
semantic-card fixtures: verified 23 case(s)
```

### Generated registry

Command:

```text
/opt/pyvenv/bin/python -m devtools.render_semantic_card_registry --check
```

Result:

```text
semantic-card registry: verified 2 surface(s)
```

### Topology inventories

Commands:

```text
/opt/pyvenv/bin/python -m devtools render topology-projection --check
/opt/pyvenv/bin/python -m devtools render topology-status --check
```

Result: both commands exited successfully. The projection reports 1,008 rows and 9 existing TBD ownership rows.

### Independent schema validation

A direct Python check used `jsonschema.Draft202012Validator.check_schema` on both public schemas, rebuilt each fixture transcript through the production renderer, validated every transcript, and separately validated every nested card against `semantic-card.v1`.

Result:

```text
validated 23 transcript fixtures and 22 card entries
```

### Shared MCP identity tests

The two fixture-free functions in `tests/unit/core/test_tool_identity.py` were executed directly with `runpy`, avoiding unrelated repository conftest imports.

Result:

```text
executed 2 standalone core identity tests
```

### Python compilation and diff hygiene

All changed Python paths, including new files, were compiled with `py_compile`.

Result:

```text
compiled 21 changed Python files
```

`git diff --check` exited successfully.

### Browser JavaScript syntax

The literal `READER_JS`, `SEMANTIC_CARD_JS`, and the script block from the assembled `WEB_SHELL_HTML` template were extracted with Python AST parsing and passed individually to Node 22.16.0:

```text
node --check /tmp/beads05-reader_js.js
node --check /tmp/beads05-semantic_card_js.js
node --check /tmp/beads05-web-shell-0.js
```

Result: all three scripts reported syntax OK.

### Patch certification

`PATCH.diff` was generated from a temporary Git index rooted at the named snapshot so tracked edits, new files, deletion/rename state, and generated text are all represented without altering the repository index.

Checks performed:

```text
git -C /mnt/data/beads05-work/base-repo apply --check PATCH.diff
git clone --no-hardlinks /mnt/data/beads05-work/base-repo <fresh clone>
git -C <fresh clone> checkout f654480cadb7cc4c194704e24dfd483199547b35
git -C <fresh clone> apply PATCH.diff
git -C <fresh clone> diff --check
```

Every changed path in the applied clone was then compared byte-for-byte with the implementation workspace.

Result:

```text
git apply --check: OK
byte-for-byte apply comparison OK for 50 changed paths
actual patch apply: OK
```

Patch properties:

```text
275,889 bytes
7,265 lines
49 diff records / 50 paths (one rename)
4,119 insertions / 536 deletions
```

The patch contains no `GIT binary patch`/`Binary files` marker, no supplied archive path, and no implementation placeholder marker.

## Tests added or strengthened but not executed in this runtime

`tests/unit/cli/test_messages.py` now supplies a realistic bounded session row to the CLI renderer. The existing structural shell test exercises the same shared renderer, but normal module import is blocked here by missing `dateparser` in the repository dependency chain.

`tests/unit/storage/test_lineage_normalization.py::test_prefix_sharing_child_stores_only_tail_and_composes` now asserts exact `lineage_inheritance`, branch-point identity, and per-message `source_session_id`. Collection reaches unrelated source-package imports and is blocked by missing dependencies (`ijson`, then `tenacity`; the normal suite also requires `aiosqlite`).

`tests/visual/test_reader_semantic_cards.py` now targets the browser’s real paginated `/api/sessions/{id}/messages` route, validates the reconstructed ordered transcript against `semantic-transcript.v1`, validates nested cards against `semantic-card.v1`, and pins the authoritative DOM hook. The daemon visual harness was not executable because the complete locked runtime dependencies are absent.

The existing ChatGPT parser regression from PR #2629, which reparses recipient-addressed tool calls as `TOOL_USE`, was inspected but not re-executed for the same dependency reason.

## Repository-wide gates attempted or unavailable

`/opt/pyvenv/bin/python -m devtools render all --check` was attempted. It stopped before running surfaces because importing the generated-surface catalog reached `polylogue/sources/decoders.py`, where `ijson` is unavailable. This is an environment dependency failure, not a reported generated drift result.

The normal root/unit pytest harness is also incomplete in this runtime: collection encounters missing packages including `hypothesis`, `ijson`, `aiosqlite`, `tenacity`, and `dateparser` depending on the selected path.

Ruff 0.15.20 and mypy are declared by the repository lock/config but no executable or cached wheel is available offline here. No Ruff or mypy pass is claimed.

## Manual and deployment checks remaining

The following checks require the operator environment and remain unverified:

- locked `devtools verify` or at minimum `devtools verify --quick` plus affected testmon selection;
- Ruff format/lint and strict mypy on all changed production/test files;
- added CLI, storage, and visual tests through the normal managed harness;
- browser inspection of a real ChatGPT export containing multiple unavailable reasoning/thinking blocks and at least one non-empty typed thinking block;
- daemon DB-backed and archive-backed route comparison on a prefix-sharing session;
- performance measurement for the paginated DB route’s whole-session placement;
- live Nix/deployment, secrets, private archive, and browser-capture behavior.
