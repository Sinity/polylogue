# TESTS — polylogue-j2zz

## Test strategy

The tests pin the production parser and the existing bounded action relation rather than reproducing the lowering in a test helper. Each test below names the production dependency and an implementation mutation that must make it fail.

| Test | Production dependency exercised | Anti-vacuity mutation/removal |
| --- | --- | --- |
| `test_single_child_retains_transport_and_promotes_exact_structural_result` | Codex envelope pre-scan, JS literal parser, child block emission, exact outcome/byte promotion, transport provenance | Remove `_code_mode_exec_envelopes`, child block append, provenance, or exact result decoding; child/action assertions collapse to outer-only |
| `test_multiple_children_preserve_order_registry_paths_and_unknown_states` | Registry classification, JavaScript source order, argv normalization, patch path parsing, MCP/unknown naming, malformed raw evidence, prose discipline | Reorder registry, regex-guess prose, omit raw arguments, or remove patch structure parser; exact arrays/fields fail |
| `test_repeated_transport_ids_pair_children_by_occurrence_rank` | Parser occurrence pairing and production `polylogue.storage.sqlite.action_pairs.refresh_action_pairs` SQL | Replace ranked pairing with equality join or make child ids occurrence-unique; expected two ranked rows fail or cross-product |
| `test_structured_content_item_result_collection_expands_in_child_order` | Current Code Mode content-item decoding and ordered result collection expansion | Treat the outer content array as prose or only parse the first item; two child results fail |
| `test_missing_structural_child_result_keeps_use_unpaired` | Header/prose rejection and no invented child result | Manufacture a result from script-status text; expected empty child result list fails |
| `test_single_status_mapping_pairs_with_unknown_outcome` | Exact result retention with nullable outcome | Require an exit code before retaining a result, or map `status=completed` to success; pairing/null assertions fail |
| `test_result_before_use_pairs_by_envelope_occurrence_without_recovery_guess` | Sequence pre-scan independent of record order | Pair only forward/adjacent records; early exact result is lost |
| `test_lowered_blocks_change_semantic_content_hash` | Real `pipeline.ids.session_content_hash` block hashing | Omit child blocks/tool input from lowering/hash projection; hashes become equal |
| `test_census_reports_outer_only_counterfactual_and_lowered_coverage` | Census entrypoint calling the production parser | Remove lowering; all typed/path/outcome counts become zero while expected lowered counts fail |
| `test_codex_code_mode_child_aliases_have_queryable_categories` | Existing viewport semantic classification | Remove `apply_patch` or `wait` aliases; category assertions fail |
| existing `test_codex_exec_freeform_arguments_are_queryable_as_commands` | Parser -> block writer -> generated columns -> `action_pairs` refresh -> `ArchiveStore.query_actions` | Remove child emission or writer refresh; expected `exec_command` action row disappears |

The fixtures are provider-wire JSONL, not prebuilt `ParsedSession` objects. The repeated-id test inserts the parser's real blocks into a minimal SQLite shape and calls the production refresh function; it does not duplicate the rank join.

## Commands and results

### Fresh patch application

```bash
git worktree add --detach /mnt/data/work-j2zz/applycheck \
  536a53efac0cbe4a2473ad379e4db49ef3fce74d
git -C /mnt/data/work-j2zz/applycheck apply --check \
  /mnt/data/work-j2zz/package-r01/PATCH.diff
git -C /mnt/data/work-j2zz/applycheck apply \
  /mnt/data/work-j2zz/package-r01/PATCH.diff
git -C /mnt/data/work-j2zz/applycheck diff --check
```

Result: passed; nine changed/added files appeared in the clean detached worktree.

### Focused production-route suite

```bash
PYTHONPATH=/mnt/data/work-j2zz/stubs:/mnt/data/work-j2zz/applycheck \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/opt/pyvenv/bin/pytest -q -o addopts='' \
  --confcutdir=tests/unit/sources \
  tests/unit/sources/test_codex_event_stream_contract.py \
  tests/unit/sources/test_tool_aliases.py \
  tests/unit/devtools/test_codex_exec_child_census.py
```

Result:

```text
34 passed, 3 warnings in 2.03s
```

The warnings are pytest configuration keys owned by optional plugins that were deliberately not autoloaded in the partial environment: `asyncio_mode`, `timeout`, and `timeout_method`.

### Compilation and formatting floor

```bash
PYTHONPATH=/mnt/data/work-j2zz/stubs:$PWD /opt/pyvenv/bin/python -m compileall -q \
  polylogue/sources/parsers/codex.py \
  polylogue/archive/viewport/tools.py \
  devtools/codex_exec_child_census.py \
  tests/unit/sources/test_codex_event_stream_contract.py \
  tests/unit/sources/test_tool_aliases.py \
  tests/unit/devtools/test_codex_exec_child_census.py \
  tests/unit/cli/test_query_expression.py

git diff --check
```

Result: passed. A direct line audit found zero changed Python lines over the repository's 120-character limit.

### Census fixture receipt

```bash
PYTHONPATH=/mnt/data/work-j2zz/stubs:$PWD /opt/pyvenv/bin/python \
  devtools/codex_exec_child_census.py \
  tests/data/codex_event_stream/functions_exec_single.jsonl \
  tests/data/codex_event_stream/functions_exec_multiple.jsonl \
  --output /mnt/data/work-j2zz/census-fixtures.json
```

Result: two files/sessions parsed, no errors, two transport actions, eleven typed child actions, eleven paired results, three path-bearing children, three structured outcomes, eight unknown outcomes, and the explicit zero-child/path/outcome outer-only counterfactual.

## Environment limits and unexecuted checks

The container's Python is 3.13.5. The project lock expects dependencies absent from the runtime. `uv` could not acquire them from the package index, and no `nix` executable was available. Import-only shims for `ijson`, `tenacity`, `aiosqlite`, and `dateparser` were placed under `/mnt/data/work-j2zz/stubs`; they are outside the repository and ZIP. The focused tests do not exercise those implementations.

The following checks were attempted or traced but not completed:

- The modified real `ArchiveStore` query test could not be collected through the repository root because `hypothesis` is absent. A manual archive route reached embeddings initialization and then stopped because `sqlite_vec` is absent.
- Ruff and strict mypy are declared project dependencies but are not installed in the container.
- `devtools verify --quick` could not run without the managed environment.
- No live daemon, configured archive, secrets, NixOS deployment, or authorized 100-session corpus was available.

Run these in the managed project environment after applying:

```bash
nix develop --command devtools test \
  tests/unit/sources/test_codex_event_stream_contract.py \
  tests/unit/sources/test_tool_aliases.py \
  tests/unit/devtools/test_codex_exec_child_census.py \
  tests/unit/cli/test_query_expression.py::TestBooleanQueryExpression::test_codex_exec_freeform_arguments_are_queryable_as_commands

nix develop --command devtools verify --quick
```

Then run the census and representative child command/path/failure action queries against the authorized live corpus after semantic re-ingest.
