# Exact next actions: census → capture → transition → cutover → regeneration → clients

## 0. Freeze authority and reconcile tracker state

1. Record old anchor `536a53efac0cbe4a2473ad379e4db49ef3fce74d` and hash `docs/generated/mcp-equivalence.json`.
2. Update `polylogue-t46.8.1`: foundation landed in `ed44be18f448`, but runtime resource/prompt parity and cold-model AC remain open or move explicitly to `.2/.3`; do not close by count alone.
3. Record the 104-vs-103 correction and the six-vs-seven `graph` decision in the Bead notes.

Acceptance: a reviewer can reproduce the exact old inventory from the anchor and no plan still claims 103 current tools.

## 1. Run the live usage census before source deletion

1. Add `devtools mcp-parity census` to materialize the declaration inventory and run the SQL in `EVIDENCE.md` read-only against `<archive-root>/ops.db`.
2. Emit `tests/golden/mcp_parity/v1/usage-census.json` or a reviewed non-sensitive committed summary containing query version, archive identity hash, horizon min/max, captured timestamp, row counts, outbox debt, per-tool counts/sessions/days/failures/latency, recommended disposition, and artifact SHA-256.
3. Promote every mapping-only row with any live call. Treat unresolved outbox pending/quarantine or an unexplained short horizon as “usage unknown,” which promotes rather than demotes.

Acceptance: every 104-row disposition cites the census artifact or an explicit criticality override; counts add to 104.

## 2. Build fixtures and capture the old surface

1. Implement `devtools/mcp_parity.py`, command catalog registration, fixture builder, normalizer, and tests.
2. Create `tests/fixtures/mcp_parity/v1/manifest.json`, `cases.jsonl`, source seeds, and expected public state.
3. Convert existing `TOOL_CONTRACT` and per-tool happy/invalid cases into parity case references.
4. Run the exact `build-fixture` and `capture-old` commands in `DECISIONS.md` **from the transition checkout against the detached old anchor before deleting a handler**.
5. Review goldens for secrets and over-normalization; commit old payloads and capture manifest.

Acceptance: every parity-golden row has required happy/boundary/error/continuation or authority cases; mapping-only rows validate target plan and census; no empty/missing/unowned case.

## 3. Implement the six/four target surface without deleting old test authority

Owning areas:

- declarations/models/registry: six read transactions, four privileged, resources/prompts, exact role dependencies;
- MCP adapters: canonical request/response models, plan lowering, resource resolver, role gating;
- shared query transaction: paging/cursor/result ref/cancellation/cleanup, no MCP-local parser or executor;
- context/assertion/maintenance owners: policy, candidate state, authorization, idempotency, receipts;
- telemetry: canonical transaction dimensions and shadow diff records;
- tests: exact discovery, schemas, lifecycle, resource/prompt role validity.

Keep legacy handlers callable only by the test/shadow harness, not by new client profiles. During this phase, production branch discovery may temporarily expose old surface until the atomic deletion commit; do not publish a mixed long-term surface.

Acceptance: new route tests pass; read adapters make bounded first progress; cancellation/disconnect releases cursors/readers/temp/leases; write dry-run receipts compare without dual mutation.

## 4. Shadow and cold-model gates

1. Run deterministic compare until zero unexplained mismatches.
2. Exercise incident-scale list/search/topology cases with repeated concurrency, cancellation, and disconnect; record bounded RSS/PSS/swap/temp bytes and return to steady state.
3. Run live transition shadow sampling for actually used read tools against one archive epoch; persist only fingerprints/diff classes.
4. Run blind tasks for phrase search, exact ref, action/path, topology/Workflow reconstruction, continuation, failure recovery, context resume, judgment authorization, and maintenance preview. Prompts must not contain expected tool names.

Acceptance: no capability/authority/lifecycle mismatch, no non-progressing continuation, no unbounded adapter state, and cold models choose valid first routes from six-tool discovery.

## 5. Prepare clients before the atomic switch

1. In Sinnix `flake/data/mcp-registry.nix`, update expected counts/tool allowlists while retaining command and role arguments.
2. Update every `client-profile` to 6/8/9/10 exact qualified names.
3. Regenerate the Polylogue skill/manual from declarations.
4. Rewrite recipe prompts and fix the read/write mismatch in `unacknowledged_failures`.
5. Change SessionStart to `context(policy="session-start", successor_session_id=...)`.
6. Search agent docs/configs for all old names and qualified prefixes; label historical analytics examples instead of rewriting them.
7. Stage package + config revisions so they can switch together; document client restart/cache invalidation.

Acceptance: a staged client pointed at the transition build completes discovery and all blind tasks using only the new names.

## 6. Atomic cutover commit

In one source commit:

1. Remove 104 legacy tool registrations/handlers or make old implementation test-only where required for frozen comparison.
2. Expose exactly six read and four role-gated transactions; no runtime aliases.
3. Wire canonical resources and retained/rewritten prompts with exact role scopes.
4. Replace runtime test baselines with 6/8/9/10; move old inventory/contracts to parity artifacts.
5. Update telemetry family mapping and operation-level Python parity declarations.
6. Update internal docs and agent manual inputs.

Do not split handler deletion from client-profile switch in a deployed channel.

## 7. Exact regeneration command order

From repository root after source/tests/docs edits:

```bash
set -euo pipefail

# Required only when files/modules were added, moved, or removed.
devtools render topology-projection

# Make the current MCP contract visible before the full registry render.
devtools render mcp-equivalence
devtools render mcp-tool-index

# Canonical registry order also regenerates CLI output schemas, OpenAPI,
# devtools docs, docs surface, topology status, and pages.
devtools render all

devtools render all --check
```

`devtools render all` currently runs: CLI reference → CLI output schemas → OpenAPI → devtools reference → demo corpus datasheet → quality reference → product workflows → docs surface → MCP equivalence → MCP tool index → topology status → pages. Review the OpenAPI/CLI schema diffs even when the intended change is MCP-only.

If topology did not change, omit only the first command; do not omit `render all`.

## 8. Exact focused verification order

```bash
set -euo pipefail

POLYLOGUE_PYTEST_WORKERS=3 devtools test \
  tests/unit/declarations/test_registry.py \
  tests/unit/mcp/test_tool_declarations.py \
  tests/unit/mcp/test_tool_discovery.py \
  tests/unit/mcp/test_tool_contracts.py \
  tests/unit/mcp/test_per_tool_contracts.py \
  tests/unit/mcp/test_envelope_contracts.py \
  tests/unit/mcp/test_server_runtime.py \
  tests/unit/mcp/test_server_surfaces.py \
  tests/parity/mcp \
  -x

devtools verify docs-coverage
devtools verify --quick
```

Add protocol smoke tests for each role and resource/prompt discovery. `devtools verify --quick` already checks format, lint, mypy, generated-surface drift, topology, layering, closure matrix, schemas, manifests, workflows, doc commands, docs coverage, and test-infrastructure policies.

Acceptance: exact role counts; no legacy runtime name; old goldens still reopen; parity compare zero unexplained; generated files clean; focused tests and quick verification green.

## 9. Deploy client switch and observe

1. Deploy new Polylogue package and Sinnix profiles together.
2. Restart MCP servers/clients and clear discovery caches.
3. Run SessionStart, query, exact get, recursive get, status, one write dry-run/apply, one judge denial/authorized case, and maintenance preview/status from the correct roles.
4. Monitor canonical transaction/operation telemetry, unknown-tool errors, shadow/cold-model mismatch metrics, call-log delivery debt, resource usage, and cancellation cleanup.
5. Keep a rollback to the old package+old profiles as one unit. Do not roll back only the server or only the client allowlist.

Acceptance: no sustained old-name calls, no role escalation, no continuation leaks, and used-tool workflows retain or improve success.

## 10. Close/reconcile Beads only with evidence

- `.2.1`: close after aliases are absent from discovery and unified filter/shape tests pass.
- `.2`: close after six-tool read equivalence, bounded incident replay, telemetry, and cold-model gates pass.
- `.3`: close after context/assertion/judgment/run/maintenance lifecycle and role-negative tests pass.
- `s1kr`: update/close only after generated operation-level Python parity includes the new transaction bindings/intentional absences.
- Parent `t46.8`: close only with final current-surface/token-cost/usage report and every retained exception justified.
