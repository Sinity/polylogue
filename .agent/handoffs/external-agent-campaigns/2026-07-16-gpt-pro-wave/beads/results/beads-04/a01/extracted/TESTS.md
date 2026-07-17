# Test design and execution ledger

## Environment

- Python 3.13.5
- pytest 9.1.1
- Ruff 0.15.20
- MyPy 1.17.1
- Linux 4.4.0 x86_64 container
- Repository base: `f654480cadb7cc4c194704e24dfd483199547b35`

The tests use repository fixtures and isolated temporary archives/databases. No operator live daemon, browser, production archive, secret, or deployed receiver was accessed.

## New production-route tests and anti-vacuity witnesses

### Declaration-owned role-scoped registration

Test: `test_declarations_drive_role_scoped_registration`

Production dependencies exercised:

- `polylogue.mcp.server_tools.register_tools`
- `polylogue.mcp.declarations.registration.MCPDeclarationRegistrar`
- `polylogue.mcp.server_support.role_allows`
- real FastMCP registration for read, write, review, and admin roles

The expected subset is independently calculated from declaration roles and compared with the real registered server. Representative changes that make it fail are restoring family-level write/review/admin registration, changing a declaration role, bypassing the registrar for one handler, or registering a privileged tool on the read server.

### Exact production handler and signature equivalence

Test: `test_declarations_bind_exact_production_handlers_and_input_contracts`

Production dependencies exercised:

- all 41 registered FastMCP handler functions;
- exact `__module__` ownership;
- async function identity;
- concrete signatures, minimal-invocation parameters, and fail-closed defaults.

Replacing a handler with a wrapper, moving it to a parallel adapter, deleting it, changing a minimal argument name, or dropping a required `confirm: bool = False` / `dry_run: bool = False` default makes the test and startup registrar fail.

### Trust, injection, and immutable annotation provenance

Test: `test_registry_preserves_trust_injection_and_annotation_provenance`

Production dependencies exercised:

- candidate capture declaration;
- annotation batch import declaration;
- review-only judgment declarations;
- registry fail-closed validation.

Promoting agent-authored capture/import to active or injectable state, weakening judgment from review to write/read, dropping `author_ref`/`author_kind=agent`, or removing any of `batch_id`, `schema_id`, `schema_version`, `source_result_ref`, `actor_ref`, `model_ref`, or `prompt_ref` makes the test fail.

### Generated equivalence drift

Test: `test_generated_equivalence_artifact_matches_executable_registry`

Production dependencies exercised:

- deterministic declaration ordering;
- canonical JSON rendering;
- committed `docs/generated/mcp-context-assertion-equivalence.json`.

Editing a declaration without regenerating the artifact, editing the artifact by hand, changing ordering, or returning non-canonical mutable values makes byte equality fail.

### No invented protocol alternatives

Test: `test_declarations_do_not_claim_unimplemented_protocol_routes`

Production dependencies exercised:

- current MCP resource registration (none for this family);
- current MCP prompt registration;
- the existing read-only `resume_context` prompt.

Adding an invented resource URI, naming a prompt that is not actually registered, or removing the real `resume_context` prompt makes the test fail. This prevents the equivalence artifact from presenting future protocol design as landed behavior.

### Discovery inventory ownership

Test: `test_expected_tool_inventory_uses_declared_names`

Production dependencies exercised:

- `tests.infra.mcp.EXPECTED_TOOL_NAMES`;
- the 41-name declaration registry union.

Removing the union, restoring a duplicate literal as a competing authority, or omitting one migrated tool makes the expected inventory diverge.

### Per-handler confirmation before backend access

Test: `test_declared_confirmation_contracts_fail_closed_before_backend_access`

Production dependencies exercised:

- eleven distinct handlers: the ten `polylogue-jn40` tools plus pre-existing `delete_session`;
- FastMCP registered functions;
- each handler's local confirmation check;
- typed `confirmation_required` error envelope.

The server facade seam is patched only as a tripwire. Every unconfirmed call must return before that seam is touched. Removing any guard, moving it after session resolution/facade access/planner construction, changing a declaration's confirmation class, or routing tools through one generic dispatcher makes this test fail.

### Real candidate write, review, and readback

Test: `test_real_mcp_candidate_write_review_and_readback_preserves_authority`

Production dependencies exercised:

- real registered write/read/review MCP functions;
- `Polylogue` facade;
- real isolated archive session;
- real durable `user.db` assertion and judgment storage;
- candidate list and active-claim read paths;
- context injection policy.

Sequence:

1. `capture_assertion_candidate` stores an agent candidate.
2. `list_assertion_candidates` reads it back as candidate with `inject=false`.
3. `judge_assertion_candidate` accepts it under `actor_ref=user:operator` and requests injection.
4. `list_assertion_claims` reads the resulting active claim with `inject=true`.

Replacing persistence with a mock-only route, auto-promoting the candidate, letting write role judge, dropping reviewer actor context, disconnecting readback, or failing to apply the accepted injection decision makes the test fail.

### Maintenance dry-run versus apply authorization

Test: `test_maintenance_execute_dry_run_does_not_require_confirmation`

Production dependencies exercised:

- distinct `maintenance_execute` registered handler;
- real handler argument routing;
- `polylogue.maintenance.planner.execute_backfill` seam;
- shared maintenance envelope.

The test proves `dry_run=True` reaches the planner with `dry_run=True` and returns an MCP execute envelope without confirmation. Requiring confirmation for safe planning or removing the non-dry confirmation gate violates the paired tests.

## Existing contract tests updated

No existing test or helper was deleted. Existing success/idempotency tests now send `confirm=True` only when they intend to exercise the post-authorization branch. Required-argument sets do not mark `confirm` as required because the public safety contract is an optional argument with a fail-closed default.

`test_personal_state_tools_depend_on_parent_composition_seam` was strengthened: suppressing the extracted personal-state registrar now must make server startup fail with missing declared handlers, rather than quietly producing a reduced server.

## Final test commands and results

### All MCP unit tests, partitioned by file

`pytest --collect-only -q tests/unit/mcp` reported **713 tests in 36 files**. Every file passed in isolated partitions. The newly changed declaration suite was kept in its own process so its result is independently attributable:

```text
pytest -q \
  tests/unit/mcp/test_agent_coordination.py \
  tests/unit/mcp/test_annotation_import_tool.py \
  tests/unit/mcp/test_assertion_judgment_tools.py \
  tests/unit/mcp/test_blackboard_tools.py \
  tests/unit/mcp/test_candidate_capture_tool.py \
  tests/unit/mcp/test_compose_context_preamble.py \
  tests/unit/mcp/test_context_image.py \
  tests/unit/mcp/test_distilled_bundle_tools.py
```

Result: **45 passed in 7.84s**.

```text
pytest -q tests/unit/mcp/test_context_assertion_declarations.py
```

Result: **11 passed in 2.86s**.

```text
pytest -q \
  tests/unit/mcp/test_mcp_call_log.py \
  tests/unit/mcp/test_mcp_edge_cases.py \
  tests/unit/mcp/test_server_runtime.py \
  tests/unit/mcp/test_tool_error_isolation.py \
  tests/unit/mcp/test_envelope_contracts.py \
  tests/unit/mcp/test_contract_evidence.py
```

Result: **102 passed in 18.08s**.

```text
pytest -q \
  tests/unit/mcp/test_server_surfaces.py \
  tests/unit/mcp/test_tool_discovery.py
```

Result: **151 passed in 29.33s**.

```text
pytest -q -n 8 \
  tests/unit/mcp/test_aggregate_sessions.py \
  tests/unit/mcp/test_analysis_primitives_facade_parity.py \
  tests/unit/mcp/test_annotation_join_tool.py \
  tests/unit/mcp/test_cli.py \
  tests/unit/mcp/test_correlate_session.py \
  tests/unit/mcp/test_cost_outlook_tool.py \
  tests/unit/mcp/test_embedding_retrieval_not_ready.py \
  tests/unit/mcp/test_embedding_status_tool.py \
  tests/unit/mcp/test_facets_tool_contract.py \
  tests/unit/mcp/test_insight_shape_tools.py \
  tests/unit/mcp/test_lineage_completeness_payload.py \
  tests/unit/mcp/test_logical_session_tool.py \
  tests/unit/mcp/test_query_tool_schema_derivation.py \
  tests/unit/mcp/test_session_analysis_primitives.py \
  tests/unit/mcp/test_session_tool_timing.py
```

Result: **78 passed in 31.92s**.

```text
pytest -q -n 4 tests/unit/mcp/test_per_tool_contracts.py
```

Result: **199 passed in 23.16s**.

```text
pytest -q -n 4 tests/unit/mcp/test_tool_contracts.py
```

Result: **98 passed in 22.33s**.

```text
pytest -q -n 4 \
  tests/unit/mcp/test_user_state_tools.py \
  tests/unit/mcp/test_tag_idempotency.py
```

Result: **29 passed in 14.91s**.

The partition totals are exactly **713 passed**, matching collection. This covers every file under `tests/unit/mcp`.

### Focused maintenance contracts

```text
pytest -q \
  tests/unit/maintenance/test_envelope_contracts.py \
  tests/unit/maintenance/test_scope_filter.py \
  tests/unit/maintenance/test_scope_filter_envelope_contract.py
```

Result: **49 passed in 2.90s**.

### Independent patch application proof

A separate detached worktree was reset to the exact base commit, cleaned, and checked:

```text
git apply --check PATCH.diff
git apply PATCH.diff
git diff --check
```

Result: all passed.

Applied-tree smoke:

```text
PYTHONPATH=<applied-worktree> python - <<'PY'
from polylogue.mcp.declarations.registry import MCP_CONTEXT_ASSERTION_DECLARATIONS
from polylogue.mcp.server import build_server
from tests.infra.mcp import EXPECTED_TOOL_NAMES
assert len(MCP_CONTEXT_ASSERTION_DECLARATIONS) == 41
assert set(build_server(role="admin")._tool_manager._tools) == EXPECTED_TOOL_NAMES
PY
```

Result: **41 declarations; 104 exact admin tools**.

Applied-tree test:

```text
pytest -q tests/unit/mcp/test_context_assertion_declarations.py
```

Result: **11 passed in 3.59s**.

### Static and generated checks

Changed-file format and lint:

```text
ruff format --check <16 changed Python files>
ruff check <16 changed Python files>
```

Result: **passed; 16 files formatted; no lint findings**.

Strict production typing:

```text
mypy \
  polylogue/mcp/declarations/__init__.py \
  polylogue/mcp/declarations/models.py \
  polylogue/mcp/declarations/registration.py \
  polylogue/mcp/declarations/registry.py \
  polylogue/mcp/server_context_tools.py \
  polylogue/mcp/server_maintenance_tools.py \
  polylogue/mcp/server_mutation_tools.py \
  polylogue/mcp/server_personal_state_tools.py \
  polylogue/mcp/server_tools.py
```

Result: **success; no issues in 9 source files**. Repository configuration sets `strict = true`.

Generated topology:

```text
python -m devtools render topology-projection
python -m devtools render topology-status
python -m devtools render topology-projection --check
python -m devtools render topology-status --check
```

Result: **passed; 1,011 realized and declared Python modules; 9 pre-existing non-blocking topology assignments remain**.

All generated surfaces:

```text
python -m devtools render all --check
```

Result: **passed** for CLI reference, CLI schemas, OpenAPI, devtools reference, demo corpus datasheet, quality reference, product workflows, docs surface, MCP tool index, topology status, and Pages/local links.

Focused repository verifiers:

```text
python -m devtools verify layering
python -m devtools verify topology
python -m devtools verify test-infra-currency
```

Results:

- layering: **no violations**;
- topology: **realized=1011, declared=1011, blocking=False**;
- test-infra currency: **89 schema tables, 8 helper references, 0 unmatched**.

Patch integrity:

```text
git diff --check
```

Result: **passed**.

The final patch changes exactly 19 repository paths. It contains no supplied archive/bundle names or `/mnt/data`/snapshot paths. A scan found no placeholder-like added lines and no standalone code-body ellipsis. The only `TBD` text anywhere in the unified diff is unchanged context from the pre-existing generated topology dashboard/assignments; the patch adds no placeholder.

## Incomplete aggregate commands

Two aggregate forms did not complete in this container even though every constituent file later passed:

1. `pytest -q tests/unit/mcp` stalled after partial progress. An xdist aggregate also stalled after partial progress. No assertion failure was emitted. Running all 36 files in isolated partitions produced 713/713 passes. The cause appears to be process-order/global-resource interaction; whether it reproduces on the operator runner is unverified.
2. `PATH="$PWD/.venv/bin:$PATH" python -m devtools verify --quick` passed Ruff format and Ruff lint, then did not complete the repository-wide MyPy phase within the container execution. The changed production modules passed strict MyPy independently, and `render all --check` passed independently. The first attempt without the virtual environment on `PATH` failed immediately because the harness could not locate `ruff`; that invocation is superseded by the corrected one but is recorded for completeness.

No claim is made that the complete repository test suite, live daemon integration, browser integration, real archive migration, NixOS packaging, or deployed MCP authentication was verified.
