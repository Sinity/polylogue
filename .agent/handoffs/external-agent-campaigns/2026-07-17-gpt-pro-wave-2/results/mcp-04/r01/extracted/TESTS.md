# Test design and execution record

## Production dependencies exercised

The central invariant exercises production code rather than a copied parser:

- `compile_expression` in `polylogue/archive/query/expression.py` for compact and Boolean session expressions.
- `parse_unit_source_expression` in the same module for terminal sources, session-scoped terminal sources, and pipeline stages.
- `_split_pipeline_stages` and `_apply_pipeline_stage` transitively through the public terminal parser.
- `query_unit_descriptor(s)` and terminal-source metadata in `polylogue/archive/query/metadata.py`.
- Public Pydantic row payload models in `polylogue/surfaces/payloads.py`.
- Existing completion serialization in `polylogue/archive/query/completions.py`.
- Click root-help rendering and generated CLI reference.
- FastMCP prompt/resource registration and the normal MCP JSON envelope/budget path.
- Existing MCP declaration enum and target algebra.
- Registered devtools generated-surface machinery and topology projection.

## Anti-vacuity and representative mutations

The tests are designed so that representative implementation regressions fail for a named reason:

- Delete or change a Lark terminal, field binding, structural predicate, or terminal-source parser branch: one or more of the 106 positive rows fails in `test_every_positive_example_parses_through_the_real_production_route`.
- Remove the quote/parenthesis-aware pipeline splitter or terminal stage application: scoped/pipeline corpus rows fail through `parse_unit_source_expression`.
- Accept a formerly rejected expression or alter an error's class/text/field: the corresponding one of 18 pinned negative tests fails.
- Change a public exhaustive/aggregate row payload field without updating declarations: the projection-model test fails.
- Remove a required unit, semantics class, cost class, or enough examples: corpus shape/count tests fail.
- Introduce a provider name, local home path, email address, or multi-sentence answer into the corpus: provider/privacy/shape tests fail.
- Bypass safe parameter rendering in a cookbook prompt: quote/space/newline test inputs make the emitted query fail production parsing.
- Hand-edit the generated discovery block, CLI reference, devtools reference, or topology projection: the corresponding renderer `--check` fails.
- Break CLI corpus-key substitution: root help exposes a marker or a changed query; the CLI help test fails.
- Remove the MCP six-class mapping or exceed the response budget: the query capability resource contract fails.
- Break docs extraction so no examples are collected: `test_docs_audit_is_not_vacuous` fails because both durable counts must remain at least 40.
- Change a concrete search-guide or generated CLI query to a rejected form: its parameterized test reports the source file and line.

## Successful commands in the development tree

Generation and compilation:

```bash
PYTHONPATH=. /opt/pyvenv/bin/python -m devtools.render_query_discovery --check
PYTHONPATH=. /opt/pyvenv/bin/python -m devtools.render_cli_reference --check
PYTHONPATH=. /opt/pyvenv/bin/python -m devtools.render_devtools_reference --check
PYTHONPATH=. /opt/pyvenv/bin/python -m devtools.render_topology_status --check
PYTHONPATH=. /opt/pyvenv/bin/python -m compileall -q polylogue devtools \
  tests/unit/archive/query/test_discovery.py \
  tests/unit/cli/test_query_discovery_help.py \
  tests/unit/devtools/test_query_teaching_surfaces.py \
  tests/unit/devtools/test_render_query_discovery.py
```

Result: all checks passed.

Static checks over all changed Python files:

```bash
/opt/pyvenv/bin/ruff format --check <all changed Python files>
/opt/pyvenv/bin/ruff check <all changed Python files>
git diff --check
```

Result: 15 Python files already formatted; Ruff passed; whitespace check passed.

Focused parser/metadata/CLI/docs suite:

```bash
PYTHONPATH=. /opt/pyvenv/bin/pytest -o addopts='' -q \
  tests/unit/archive/query/test_discovery.py \
  tests/unit/archive/query/test_transaction.py \
  tests/unit/archive/test_query_metadata.py \
  tests/unit/cli/test_query_expression.py \
  tests/unit/cli/test_completions_contract.py \
  tests/unit/cli/test_query_discovery_help.py \
  tests/unit/devtools/test_generated_surfaces.py \
  tests/unit/devtools/test_query_teaching_surfaces.py \
  tests/unit/devtools/test_render_query_discovery.py
```

Result: **678 passed, 1 skipped**.

Affected MCP contracts:

```bash
PYTHONPATH=. /opt/pyvenv/bin/pytest -o addopts='' -q \
  tests/unit/mcp/test_envelope_contracts.py::test_query_capability_resource_exposes_mcp_algebra_and_valid_terminal_forms \
  tests/unit/mcp/test_server_surfaces.py::TestPromptSurfaces::test_cookbook_prompt_query_strings_parse_through_production_routes \
  tests/unit/mcp/test_server_surfaces.py::TestPromptSurfaces::test_cookbook_prompts_prefill_repo_context \
  tests/unit/mcp/test_server_surfaces.py::TestPromptSurfaces::test_cookbook_prompts_render_tool_recipes
```

Result: **9 passed**.

## Fresh-apply certification

A detached worktree was created at `536a53efac0cbe4a2473ad379e4db49ef3fce74d`. The package patch then passed:

```bash
git apply --check PATCH.diff
git apply PATCH.diff
git diff --check
```

After application, all four renderer checks, Ruff format/lint, and the same 678+1 and 9-test commands above passed again. Fresh-worktree results:

- **678 passed, 1 skipped in 14.11s**
- **9 passed in 3.56s**

This proves the results do not depend on unrecorded files in the implementation worktree.

## Honest incomplete or failed environment checks

Project-lock bootstrap:

- The lock requested `types_dateparser-1.4.1.20260617`.
- The configured package mirror returned HTTP 403 for that future-dated wheel.
- A complete lock-faithful environment was therefore not available.
- Verification used `/opt/pyvenv`, `pip install -e .`, and the missing runtime test tools (`hypothesis`, `ruff`, `pytest-xdist`, `pytest-timeout`, and `pytest-benchmark`).

Full MCP file run:

```bash
PYTHONPATH=. /opt/pyvenv/bin/pytest -o addopts='' -q \
  tests/unit/mcp/test_envelope_contracts.py \
  tests/unit/mcp/test_server_surfaces.py
```

Result: the command showed no failure through 61% but exceeded the 300-second command ceiling. It is recorded as **unverified**, not passed. The nine directly affected tests complete successfully as shown above.

Not performed:

- Full repository-wide pytest suite.
- Full `devtools verify`, mypy, or Nix-based verification.
- Live daemon/archive/browser/MCP client execution.
- Installed shared-skill parity or cold-model trials.
- Current-coverage/freshness/catalog cardinality tests, because those production features are not part of this patch.
