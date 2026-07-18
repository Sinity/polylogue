# TESTS — terminal continuity gate

## Test design

The tests are organized around production ownership rather than around the historical symptom list. The public-route tests exercise MCP discovery, JSON-RPC framing, the shared `query_units` transaction, opaque continuation state, SQLite lowering, exact totals, and evidence projection. The terminal tests then consume provider-native Claude artifacts, material-origin parsing, the generic work graph, independent effect receipts, reconciliation, repository persistence/traversal, and the selective action planner.

Expected answers are planted in `tests/data/continuity/catalog.json` before route execution and independently checked by direct fixture/SQLite census. The incident's later route arguments are not parameterized from those expected values; they are materialized from public discovery observations.

## Production dependency and anti-vacuity matrix

| Test / assertion | Production dependency | Representative mutation/removal that must fail |
|---|---|---|
| `test_all_scenarios_pass_through_official_mcp_stdio_json_rpc` | MCP SDK stdio transport, installed read server, tool discovery, public handlers, query DSL, SQLite, continuation transaction | Replace stdio with in-process fake; remove tool/schema discovery; erase continuation state; cap totals; duplicate pages; change query/result identity. |
| Incident binding assertions | Public discovery message plus `ContinuityBindingProjection` | Put run/coordinator fixture IDs back into authored member/topology expressions or remove the discovery markers. |
| Six-page exact membership assertions | `query_units`, exact count probe, page envelopes, item identity, query/result refs | Return capped pseudo-total, replay page one, omit offsets, duplicate an identity, or stop at an arbitrary semantic cap. |
| Restart/resume assertions | `ContinuityRoute.restart`, opaque continuation, archive epoch validation | Store continuation only in server process, mutate tool discovery after restart, or produce a non-progressing token. |
| Named t8t curriculum | Discovery/formulation/execution/projection/reasoning classifiers | Lost request state, hidden grammar, missing coverage, unreasonable query, transport failure, oversized physical request. |
| Claude artifact coverage | `artifact_rule_for_path`, `inventory_claude_orchestration_artifacts`, orchestration parser | Remove one metadata sidecar or an OriginSpec admission rule. |
| Attempt representation | `project_claude_workflow_evidence` admitted-artifact map and journal paths | Remove `metaPath` or `transcriptPath`; do not fabricate a one-to-one link. |
| Material provenance | Claude Code parser `MaterialOrigin` assignment | Mark a generated worker prompt as human-authored or collapse all `role=user` rows. |
| Work graph census | Generic run/invocation/call/attempt/result/claim nodes and resume edges | Infer Workflow membership from all parent children, drop calls/results, merge resumed run, or omit unresolved state. |
| Effect reference check | Final structured result vs independent git/PR/Beads receipts | Change a direct effect ref without changing the independent receipt. |
| Claim/effect separation | `reconcile_work_effects` and typed `ObjectRef` identity | Reuse the commit effect identity for the PR or promote a self-report directly into an observed effect. |
| Persisted traversal | `SessionRepository.replace_work_evidence_graph` and `traverse_work_evidence` | Drop the `claimed` edge, snapshot, effect node, or persistence adapter. |
| Selective SQL terminal canary | `_action_relation_for_query`, shared query runner, aggregate lowering | Return `("", "actions", [])` for an exact-session predicate. Expected: `selective_sql_plan_amplification` / `plan`. |
| Live authorization | `_validate_lane_contract` | Allow live lane with only CLI flag, only environment value, or short/no salt. |
| Live redaction | `redact_live_continuity_report` | Preserve raw path/ID/text/evidence refs or redact arguments field-by-field instead of HMACing the canonical object. |
| Cold-model receipt | `_verify_cold_model_receipt` | Accept a mismatched prompt hash, hidden fixture input class, undeclared plan, or evidence-free facts. |

## Exact successful commands and results

### Changed-file static checks

```bash
BASE=3a23389823b9a78fe03f497ee719ac9af670d815
mapfile -t PYFILES < <({ git diff --name-only "$BASE"; git ls-files --others --exclude-standard; } | sort -u | grep -E '\.py$')
ruff format --check "${PYFILES[@]}"
ruff check "${PYFILES[@]}"
mypy "${PYFILES[@]}"
python -m compileall -q "${PYFILES[@]}"
git diff --check "$BASE"
```

Result:

```text
16 files already formatted
All checks passed!
Success: no issues found in 16 source files
```

`compileall` and `git diff --check` completed with no output/errors.

### Generated surfaces

```bash
python -m devtools render all --check
```

Result: exit 0. CLI reference, CLI output schemas, OpenAPI, WebUI design system/client, devtools reference, demo corpus datasheet, quality reference, product workflows, docs surface, MCP equivalence/tool index, topology status, and pages all reported in sync; site sources and local links resolved.

### Focused dependency/unit batch

```bash
pytest -q -n 0 -p no:randomly -p no:random-order \
  tests/unit/mcp/test_prompt_query_parity.py \
  tests/unit/product/test_continuity_scenarios.py \
  tests/unit/insights/test_claude_workflow_evidence.py \
  tests/unit/insights/test_work_reconciliation.py \
  tests/unit/insights/test_work_evidence.py \
  tests/unit/sources/test_parsers_claude_code_artifacts.py \
  tests/unit/devtools/test_continuity_replay_lane.py \
  tests/unit/storage/test_archive_tiers_archive.py::test_exact_session_action_count_bounds_pairing_before_global_ranking \
  tests/unit/storage/test_archive_tiers_archive.py::test_c03_exact_session_actions_uses_real_provider_pipeline_and_planted_facts \
  tests/unit/archive/query/test_execution_control.py::test_exact_session_multi_aggregate_work_is_not_amplified_by_irrelevant_growth
```

Result:

```text
45 passed, 8 warnings in 5.81s
```

The warnings are Python 3.13's existing `multiprocessing` fork-from-multithreaded-process deprecation emitted by the provider-pipeline fixture.

### Terminal mutation integration

```bash
pytest -q -n 0 -p no:randomly -p no:random-order \
  tests/integration/test_terminal_continuity_gate.py
```

Result:

```text
6 passed in 5.19s
```

### Real MCP continuity integration

```bash
pytest -q -n 0 -p no:randomly -p no:random-order \
  tests/integration/test_continuity_replay.py
```

Result:

```text
10 passed in 18.73s
```

This includes the all-eight-scenario official stdio walk, six named t8t mutations, a dropped-workflow-filter mutation, and an independent-oracle mutation.

### Standalone official stdio replay

Fixture creation:

```bash
python - <<'PY'
from pathlib import Path
from tests.infra.continuity import load_continuity_catalog, seed_continuity_archive
root = Path('/tmp/mandate03-smoke/archive')
seed = seed_continuity_archive(root, catalog=load_continuity_catalog())
print(len(seed.evidence_paths))
PY
```

Result: `189` terminal evidence paths.

Replay:

```bash
python devtools/continuity_replay.py \
  --archive-root /tmp/mandate03-smoke/archive \
  --oracle tests/data/continuity/catalog.json \
  --transport stdio \
  --output /tmp/mandate03-smoke/report.json
```

Observed receipt summary:

```text
status=pass
transport=mcp-stdio-json-rpc
scenario_count=8
passed=8
failed=0
incident_page_count=6
incident_restart_count=1
incident_exact_enumeration_verified=true
terminal_status=pass
calls=50
attempts=91
results=65
completed_call_keys=49
unresolved_call_keys=1
final_results=1
excluded_unrelated_children=38
unrelated_children_admitted_to_graph=0
selective_relation=bounded_actions
selective_relation_parameter_count=3
cold_model.status=unavailable
```

Measured incident guidance:

```text
observed_calls=27 / max_calls=32
largest_page_bytes=23958 / max_page_bytes=25000
total_response_bytes=281027 / max_total_bytes=650000
scenario_elapsed_ms=5056.612 / max_elapsed_ms=45000
process_peak_rss_bytes=372809728 / max_memory_bytes=536870912
restart_resume_elapsed_ms=4514.813 / max_cancel_grace_ms=1000
advisories=[max_cancel_grace_ms_guidance_exceeded]
advisory_only=true
```

The restart grace miss is intentionally reported and does not invalidate a complete exact result.

### Authorized redaction smoke over sanitized data

```bash
POLYLOGUE_LIVE_CONTINUITY_AUTHORIZATION=I_AUTHORIZE_POLYLOGUE_REDACTED_LIVE_SCALE \
POLYLOGUE_LIVE_REDACTION_SALT=terminal-continuity-redaction-salt-0001 \
python devtools/continuity_replay.py \
  --archive-root /tmp/mandate03-smoke/archive \
  --oracle tests/data/continuity/catalog.json \
  --scenario parallel-claude-incident \
  --transport registered \
  --lane live \
  --authorize-live-redacted \
  --output /tmp/mandate03-smoke/live-redacted-report.json
```

Result: exit 0, report status `pass`, algorithm `HMAC-SHA256`. Scans found no raw archive path, `wf_synthetic_841`, coordinator session ID, fixture ID, or `repo:polylogue` text. This verifies the authorization/redaction mechanism only; it is not a live-scale archive run.

### Fresh-baseline patch application

```bash
git worktree add --detach /tmp/mandate03-applycheck \
  3a23389823b9a78fe03f497ee719ac9af670d815
cd /tmp/mandate03-applycheck
git apply --check /tmp/mandate03-package/PATCH.diff
git apply /tmp/mandate03-package/PATCH.diff
git diff --check 3a23389823b9a78fe03f497ee719ac9af670d815
```

Then all 17 patch members were SHA-256 compared with the implementation tree, and the changed-file Ruff/Mypy/compile checks were rerun in the applied worktree.

Result:

```text
patch_members=17
content_mismatches=[]
16 files already formatted
All checks passed!
Success: no issues found in 16 source files
```

## Honest failed or environment-limited checks

### Initial raw pytest plugin failure

An initial focused invocation with the repository's ambient randomization plugins failed before product execution with:

```text
ValueError: Seed must be between 0 and 2**32 - 1
```

The repository's deterministic plugin exclusions (`-p no:randomly -p no:random-order`) were then used for all reported focused results.

### Combined pytest process

One combined command containing the fork-based provider-pipeline fixture followed by stdio subprocess integration tests reached 58 passing dots and then exceeded a 300-second command cap. No product failure was emitted. The same selections pass in the isolated batches reported above (45 + 6 + 10). This is recorded as an orchestration/fixture interaction, not counted as a successful check.

### Managed `devtools test`

```bash
python -m devtools test tests/integration/test_terminal_continuity_gate.py
```

The wrapper refused to start pytest because this container exposes only 64 MiB free in `/dev/shm`:

```text
verify: only 64 MiB free in /dev/shm; refusing disk-backed pytest
```

Focused raw pytest was therefore used.

### Repository-wide quick verification

```bash
python -m devtools verify --quick
```

Ruff format and Ruff lint passed. The repository-wide no-argument Mypy stage did not finish within the local 600-second command cap. Focused strict Mypy over every changed Python file passed, and generated surfaces were checked separately and passed. Repository-wide Mypy is therefore `unverified`, not claimed green.

## Unverified external checks

- External cold-model execution from MCP schemas/errors/catalog evidence only.
- Explicitly authorized live-scale run against the operator archive.
- Private daemon, browser, secrets, NixOS deployment, or 4.85-million-block corpus.
- Real GitHub PR checks/status API evidence.
- Current live Beads database, complete Beads history reconciliation, or tracker updates.
- Full repository test suite and repository-wide Mypy completion.
