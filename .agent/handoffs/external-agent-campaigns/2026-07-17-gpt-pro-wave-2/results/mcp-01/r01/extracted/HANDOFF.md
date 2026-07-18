# Six-tool-era standing agent manual and continuity kit

## Mission and result

This package is an apply-ready implementation draft for the Polylogue MCP cutover from the compatibility surface to the six default read transactions `query`, `read`, `get`, `explain`, `context`, and `status`, with role-gated `write`, `judge`, `run`, and `operate`. It regenerates the standing manual from typed declarations, teaches the real query grammar and continuation protocol, retains the already-delivered beads-06 native installation architecture, adds deterministic drift checks, and exposes staged manual resources without falsely claiming that the cutover has landed.

The implementation deliberately fails closed at the activation boundary. The checked snapshot still registers 104 compatibility tools and does not contain cutover-final FastMCP signatures. Therefore native installation and automatic server-instruction injection require two independent conditions before teaching the staged calls as live: exact role-scoped target tool names and a `live-verified` contract-schema marker. The generated assets remain inspectable and packageable while those gates are false.

## Snapshot identity and authority

| Field | Value |
| --- | --- |
| Snapshot source | `/realm/project/polylogue`, as recorded by `polylogue-manifest.json` |
| Snapshot generated | `2026-07-17T18:09:50Z` |
| Recorded branch | `master` |
| Commit | `536a53efac0cbe4a2473ad379e4db49ef3fce74d` |
| Commit subject | `fix(repair): harden raw authority convergence (#3046)` |
| Recorded working-tree state | dirty |
| Branch delta against remote default | empty log, empty file list, empty patch |
| Patch base | exact commit above |

The all-refs bundle was checked out detached at the recorded commit to create the patch. The packaged working-tree source was compared with all 3,864 tracked paths at that commit. Every packaged tracked file matched byte-for-byte; the 1,300 omitted tracked paths were confined to `.agent`, `.beads`, `.claude`, `flake.lock`, and `uv.lock`. Consequently the production source used here is coherent with the named commit, but the snapshot does not contain enough information to reconstruct the operator's exact original dirty delta in those omitted local-state areas. `PATCH.diff` does not modify any of them except the separately reconstructed checkout's tracked `flake.nix`; it does not modify either lockfile.

## Evidence inspected

The implementation followed production dependencies beyond the requested manual files:

- MCP declaration authority in `polylogue/mcp/declarations/models.py`, `registry.py`, and registrar/runtime inventory paths.
- Current FastMCP server composition in `polylogue/mcp/server.py`, `server_resources.py`, the tool-family registration modules, and `tests/infra/mcp.py`.
- The real parser in `polylogue/archive/query/expression.py`, strict command-floor routing in `polylogue/cli/query_group.py` and `root_request.py`, and parser/CLI fixtures.
- The real opaque continuation implementation in `polylogue/archive/query/transaction.py` and its `q1` request/result-ref binding.
- Mutation authority intent in `polylogue/operations`, action contracts, and Bead `polylogue-t46.9`.
- The authoritative `Origin` enum in `polylogue/core/enums.py` and provider/source mappings.
- Current packaging, generated-surface, documentation topology, CLI registration, MCP resource, and server-surface tests.
- Beads `polylogue-3gd.2`, `polylogue-3gd.3`, `polylogue-t46.8`, `polylogue-t46.8.1`, `polylogue-t46.8.2`, `polylogue-t46.8.3`, and `polylogue-t46.9`.
- All-ref history, especially `7d918eb08 chore(campaign): intake latest external reports (#3047)`, which contains the beads-06 report package on a non-base ref.
- The extracted beads-06 `HANDOFF.md`, `PATCH.diff`, `TESTS.md`, and `EVIDENCE.md`, plus a temporary application of that patch for architectural comparison.

## Beads-06 reuse adjudication

Verdict: reuse, not replacement.

The implementation keeps the beads-06 system boundaries and installation semantics:

- a typed integration specification as the canonical manual input;
- generated, versioned manual/reference/recipe/manifest assets;
- an asset digest and cache key rather than mutable hand-written client copies;
- an ownership ledger for native client configuration;
- idempotent install, status, doctor, upgrade, drift detection, and lossless uninstall;
- role-scoped MCP manifests;
- Claude Code SessionStart delivery and the closest persistent native mechanism for Codex, Gemini CLI, and Hermes;
- a Home Manager module that owns agent integration only and does not start or configure daemon ingestion;
- production-route verification rather than prose-only assertions.

`nix/agent-integration-module.nix` is byte-identical to the beads-06 implementation. `polylogue/agent_integration/installer.py` retains its behavior and structure; changes are limited to current-package imports, direct use of the repository's `MCPRole`, strict typing, formatting, and preservation of its public conflict exception.

The content and reconciliation layer changed because beads-06 targeted the old compatibility surface. Its manual declarations, generated examples, and verifier were replaced with the six-tool-era contract. The new layer normalizes the in-flight separate `graph` declaration into `read`, maps current `maintenance` authority to the target public name `operate`, uses the production parser and continuation codec, reports the current eleven Origin tokens, and adds the dual activation guard. No second installer, ownership database, manual hierarchy, or client configuration framework was introduced.

## Implemented mechanism

### Typed contract and generated assets

`polylogue/agent_integration/spec.py` declares:

- asset version `2026-07-17.6tool-r01`;
- explicit schema state `cutover-parameterized`;
- ten target transaction contracts;
- six default-read tools and cumulative role visibility (`read`, `write`, `review`, `admin`);
- source-declaration mappings, including `read <- (read, graph)` and `operate <- maintenance` for this snapshot;
- typed argument specifications, normal calls, resumability, result semantics, resource/prompt links, role and capability requirements, and confirmation behavior;
- ten parser-backed DSL examples and strict command-floor examples;
- four six-tool-only continuity recipes;
- eleven enum-checked Origin meanings;
- native delivery declarations for Claude Code, Codex, Gemini CLI, and Hermes.

`devtools/render_agent_manual.py` deterministically renders six package assets and the two public documentation copies:

- `standing-manual.md`;
- `deep-reference.md`;
- `recipes.json`;
- `integration-spec.json`;
- `tool-contracts.json`;
- `integration-manifest.json`;
- `docs/agent-manual.md`;
- `docs/agent-integration-reference.md`.

The resulting asset digest is `9635a669c7510574702c579f3b99924146235d129a93157b36f6cd41d97b709e`; the cache key is `polylogue-agent-2026-07-17.6tool-r01-9635a669c7510574`. The standing manual is 23,331 bytes and the deep reference is 29,757 bytes.

### Continuation, limits, refs, and citations

The manual's resume example is generated by the production `QueryContinuation` implementation. It is a real deterministic `q1` token that binds the original operation, expression/projection arguments, page size, stable ordering, offset 20, and `result:0123456789abcdef01234567`. Resumption sends only:

```json
{"continuation":"<opaque token returned by the preceding response>"}
```

The compiler rejects mixed initial/resume shapes. Every contract whose result semantics are resumable must expose `continuation`; the verifier caught and the implementation corrected an initial omission on `run`. The manual distinguishes physical response limits from logical completeness and teaches narrower projection/filtering, opaque continuation, durable result refs, explicit top-k/sample/aggregate semantics, and citation of refs rather than re-created prose.

### Query language teaching

All ten DSL examples pass through `parse_expression_ast` and serialize/parse again. The standing section teaches the three intent signals used by the strict command floor:

1. leading `find`;
2. a quoted expression;
3. recognized field syntax.

A bare root word is intentionally refused rather than silently becoming an archive query. The examples were selected from or cross-checked against repository parser and CLI tests; there is no manual-only grammar.

### Role and safety ladder

The target role surface is cumulative:

| Role | Visible target transactions |
| --- | --- |
| `read` | the six default tools |
| `write` | read tools plus `write` and `run` |
| `review` | write surface plus `judge` |
| `admin` | review surface plus `operate` |

`write` handles governed reversible mutation; `judge` records adjudication with conflict semantics; `run` executes declared recipe/saved-query refs and is resumable; `operate` handles administrative preview/token/execute lifecycle. Destructive execution is documented against `polylogue-t46.9`: a fresh confirmation token must bind actor, archive identity, operation/spec version, expiry, and exact target/preview digest. The patch does not invent an MCP-local authorization engine.

### Runtime activation and MCP delivery

`polylogue/agent_integration/manifest.py` exposes the current and target surfaces side by side. `target_surface_is_registered(role)` is true only when:

1. the role-scoped runtime name set equals the target set; and
2. generated contracts have been rebound to final live schemas and marked `live-verified`.

`polylogue/mcp/server.py` injects the complete standing manual into server instructions only after both gates pass. Before then it emits compatibility guidance and directs clients to live discovery. `polylogue/mcp/server_resources.py` exposes staged, inspectable resources at:

- `polylogue://agent/manual`;
- `polylogue://agent/reference`;
- `polylogue://agent/manifest/{role}`.

The staged manual itself prominently reports that the final schemas are absent, so resource inspection cannot be mistaken for cutover completion.

### CLI and native installation

The root CLI now registers `polylogue agent` with:

- `manual`;
- `manifest`;
- `install`;
- `status`;
- `doctor`;
- `uninstall`;
- hidden `session-start` for Claude Code.

`install` fails closed until the dual cutover gate is complete. The lower-level ownership-aware manager remains testable and stageable in isolated homes.

Per-client delivery deltas are:

| Client | Six-tool-era change | Untouched beads-06 mechanism |
| --- | --- | --- |
| Claude Code | SessionStart emits the complete six-tool standing manual and reconciled role manifest. | Native hook merge, ownership ledger, MCP entry, drift handling, exact uninstall. |
| Codex | Managed standing block contains the generated six-tool manual; reference and MCP profile use the same digest/role. | Effective global `AGENTS.md` override/base strategy and marked ownership. |
| Gemini CLI | Managed `GEMINI.md` content is regenerated from the six-tool contract. | Settings merge, marked block, reference file, ownership behavior. |
| Hermes | Project-owned productivity skill teaches six-tool calls and continuity recipes. | Native skill directory, config merge, reference path, ownership behavior. |

### Verification lanes

`devtools/verify_agent_integration.py` defines eight lanes:

1. generated asset drift and digest;
2. manual compilation against typed target/source declarations;
3. query-parser and strict-floor round trip;
4. continuation binding and request-shape validation;
5. target/current declaration reconciliation;
6. native installer round trip across all four clients;
7. package/Home Manager asset ownership;
8. exact live FastMCP signatures.

The first seven pass. The eighth is intentionally `unverified`, not skipped as success, because the snapshot still exposes 104 compatibility tools. `--require-live` returns exit status 1 until the cutover-final server is present.

## Current-source contradictions and decisions

| Authority conflict | Current evidence | Decision |
| --- | --- | --- |
| Mission describes a 103-tool surface. | Runtime declaration verification reports 104 tools. | Report 104; do not preserve a stale count. |
| Mission names six default reads. | In-flight declaration registry has seven rows because `graph` remains separate. | Generate target `read` from current `read` plus `graph`; live verifier must prove the final merged signature. |
| Mission names privileged `operate`. | Current registry names `maintenance`. | Map source `maintenance` to public target `operate`; require final argument reconciliation. |
| Mission requests ten Origin tokens. | `Origin` contains eleven, including `beads-issue`. | Generate all eleven from enum-checked declarations and state why the old count is stale. |
| Mission requires exact examples. | Final FastMCP signatures are absent. | Mark schemas `cutover-parameterized`, compile statically, and block live activation until exact signature parity. |
| Mutation confirmation is in flight. | `polylogue-t46.9` owns executable preview-bound confirmation. | Teach its declared lifecycle but do not implement competing MCP authorization. |

## Changed files

### Canonical specification, rendering, verification, and package assets

- `polylogue/agent_integration/__init__.py`
- `polylogue/agent_integration/assets.py`
- `polylogue/agent_integration/spec.py`
- `polylogue/agent_integration/manifest.py`
- `polylogue/agent_integration/installer.py`
- `polylogue/agent_integration/data/__init__.py`
- `polylogue/agent_integration/data/standing-manual.md`
- `polylogue/agent_integration/data/deep-reference.md`
- `polylogue/agent_integration/data/recipes.json`
- `polylogue/agent_integration/data/integration-spec.json`
- `polylogue/agent_integration/data/tool-contracts.json`
- `polylogue/agent_integration/data/integration-manifest.json`
- `devtools/render_agent_manual.py`
- `devtools/verify_agent_integration.py`

### CLI, MCP, and installation delivery

- `polylogue/cli/commands/agent.py`
- `polylogue/cli/click_command_registration.py`
- `polylogue/mcp/server.py`
- `polylogue/mcp/server_resources.py`
- `nix/agent-integration-module.nix`
- `flake.nix`

### Generated documentation and repository discovery surfaces

- `docs/agent-manual.md`
- `docs/agent-integration-reference.md`
- `docs/README.md`
- `docs/cli-reference.md`
- `docs/devtools.md`
- `docs/plans/topology-target.yaml`
- `docs/topology-status.md`
- `devtools/command_catalog.py`
- `devtools/docs_surface.py`
- `devtools/generated_surfaces.py`

### Tests and fixtures

- `tests/unit/agent_integration/test_assets_and_cli.py`
- `tests/unit/agent_integration/test_installer.py`
- `tests/unit/agent_integration/test_manual_contract.py`
- `tests/unit/mcp/test_server_surfaces.py`
- `tests/infra/mcp.py`
- `tests/unit/cli/__snapshots__/test_help_snapshots.ambr`

The patch contains 36 files, 8,121 insertions, and 8 deletions. Generated topology files are included as binary-safe Git patches because repository attributes mark them non-textual.

## Acceptance matrix

| Requirement | Result | Evidence |
| --- | --- | --- |
| Complete six-tool standing manual generated from declarations | pass | generated-assets and manual-compilation lanes |
| Normal invocation of all six tools | pass, statically compiled | typed calls in standing/deep manuals; live signatures blocked |
| Exact continuation request, limits, and recovery | pass | production `q1` codec, two request-shape fixtures |
| Result-ref citation discipline | pass | generated manual/reference and typed result semantics |
| Role ladder and confirmation gates | pass at declaration/manual layer | cumulative manifests; t46.9-owned lifecycle |
| URI resource addressing and prompts | pass | typed target resources/prompts; MCP manual resources |
| Source coverage | pass against current enum | all eleven authoritative Origin tokens |
| Four six-tool-only continuity workflows | pass | recipe compiler verifies every call/tool/argument |
| Parser-backed DSL and strict command floor | pass | ten round trips; all three intent signals and bare-word refusal |
| Reuse beads-06 installation architecture | pass | byte-identical Nix module; retained ownership-aware installer |
| Claude/Codex/Gemini/Hermes native round trips | pass in isolated temporary homes | installer verification lane and focused tests |
| Deterministic regeneration and `--check` drift failure | pass | renderer and generated-surface lanes |
| Exact final FastMCP schema parity | unverified by design | cutover not registered; `--require-live` fails closed |
| Nix parse/evaluation | unverified | Nix executable unavailable in the container |
| Live daemon/archive/client binary smoke | unverified | no operator daemon, archive, secrets, deployment, or client binaries were used |
| Cold-agent blind trials/ablation | unverified | requires cutover-final surface and client runtimes |

## Apply order

1. Apply `PATCH.diff` to commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d` or rebase it carefully if the cutover branch has moved.
2. Run `python devtools/render_agent_manual.py --check` and `python -m devtools.render_all --check` before editing generated files.
3. Run `python -m devtools.verify_agent_integration --json`; expect seven passes and one unverified lane on the base snapshot.
4. Run the focused tests listed in `TESTS.md`.
5. Build wheel and source distribution; verify all six data assets and the installer/CLI modules are present.
6. Do not run `polylogue agent install` against users while schemas remain `cutover-parameterized`; the CLI correctly refuses.

## Exact post-cutover regeneration checklist

After `polylogue-t46.8.2`, `polylogue-t46.8.3`, and the relevant `polylogue-t46.9` declarations land:

1. Rebase the patch and inspect the final `polylogue/mcp/declarations/` transaction definitions and FastMCP registration output.
2. Replace parameterized argument names/types/defaults in `TOOL_CONTRACTS` with values mechanically derived from the final signatures. In particular verify:
   - `query` initial versus continuation-only shape;
   - `read` graph/topology selector and projection/view fields;
   - `get` ref kinds and exact evidence resolution fields;
   - `context` result-ref and receipt inputs;
   - `run` continuation and saved-query/recipe ref fields;
   - `operate` preview, bound token, execution, status, and receipt fields.
3. Reconcile role/capability declarations and t46.9 confirmation binding. Remove the temporary `maintenance -> operate` mapping when `operate` is authoritative.
4. Run `python -m devtools.verify_agent_integration --require-live`. Fix every exact FastMCP name/required/default mismatch.
5. Only after that lane reaches parity, set `TARGET_SCHEMA_STATUS = "live-verified"` and ensure every generated `ToolContract.schema_status` is live-verified.
6. Regenerate with `python devtools/render_agent_manual.py`, then run the same command with `--check`.
7. Run `python -m devtools.render_all --check`, the focused tests, complete MCP registration tests, and topology verification.
8. Build wheel/sdist/Nix outputs and run clean-home install/status/doctor/upgrade/uninstall for all four clients.
9. Run live invocation fixtures against a public/demo archive, including continuation, stale result refs, incomplete coverage, no-daemon, wrong-archive, and unconverged-index states.
10. Run cold-agent blind trials and ablation before removing compatibility aliases or enabling automatic standing instructions in production.

## Risks, limitations, and value of another iteration

The principal risk is schema timing, not the generated/manual architecture. Exact cutover signatures, graph ownership, and `operate` fields do not exist in this snapshot. The dual gate prevents those unknowns from becoming active instructions, but a post-cutover reconciliation pass is mandatory.

The package does not prove Nix evaluation, live daemon behavior, real client process integration, or cold-model task success. It proves generated content, parser/continuation validity, static declaration reconciliation, native file/config ownership semantics, packaging, and server/CLI composition in the available snapshot.

Before cutover, another iteration would add only a small repair or independent review: wording refinements, broader static mutation tests, or more documentation assertions. After final t46.8/t46.9 schemas land, a second pass has substantial value because it can replace all parameterized fields with executable signatures, turn the eighth lane green, run live client/archive workflows, and produce the evidence required to retire compatibility tools.

## Verification summary

- 114 agent-integration and MCP server-surface tests passed.
- 364 CLI help/contract tests passed; one approval snapshot passed.
- 45 command-catalog/generated-surface/render/topology tests passed.
- Total focused pytest acceptance set: 523 passed, 0 failed.
- Strict Ruff and MyPy checks passed for the changed implementation and focused tests.
- Deterministic manual regeneration and full generated-surface checks passed.
- Agent verifier: 7 pass, 0 fail, 1 intentionally unverified.
- `--require-live` returned exit 1 as designed on the compatibility server.
- Fresh-checkout `git apply --check`, application, `git diff --check`, renderer check, and verifier reproduction passed.
- Wheel and source distribution built after the final contract changes. The wheel contains all 13 expected Python modules/assets; the source distribution contains those plus the Home Manager module.
- Installed-wheel smoke imported from the installation target, loaded all six assets, reproduced the asset digest, observed 66 read-role compatibility declarations, and reported both activation gates false.
- `uv.lock` remained byte-stable at SHA-256 `de6874fc1d719617f02349280dff9dce6b3cd35c6f12a5e28c3610e8af90a727`.
- `git diff --check` passed; no copied snapshot/archive or environment-specific package index change is in the patch.

# Complete rendered standing manual

The following is the exact generated `docs/agent-manual.md` included by `PATCH.diff`.

<!-- Generated by `devtools render agent-manual` from Polylogue declarations. Content version: 2026-07-17.6tool-r01. -->
# Polylogue standing agent manual

Polylogue is the local evidence system for prior AI work. Use it whenever the task depends on what was tried, decided, changed, observed, paid for, or left unfinished. Do not wait for the operator to say “search the archive.” First establish archive authority, then retrieve evidence, then cite stable refs. Do not use Polylogue for facts that the current repository or live system can answer more directly.

This manual targets the six-tool default read surface: `query`, `read`, `get`, `explain`, `context`, and `status`. The current source snapshot still carries an in-flight seven-row declaration (`graph` is separate) and calls the administrative transaction `maintenance`; this generated contract folds graph into `read` and names the administrative transaction `operate`. Final FastMCP signatures remain a post-cutover verification gate.

## Cold-start decision route

1. Call `status` before any broad, freshness-sensitive, or source-completeness claim.
2. Call `explain` when grammar, fields, values, result semantics, refs, or recovery are uncertain. Never guess a field or tool name.
3. Call `query` to discover sets or aggregates. Preserve its `query_ref` and `result_ref`.
4. Call `read` for bounded context around a stable ref or retained result set; call `get` for one exact object.
5. Call `context` only after discovery when a bounded resume/postmortem/prior-art packet is useful. Its receipt states what was included and omitted.
6. Continue any exhaustive result until the response has no continuation when the claim requires logical completeness.

## Source coverage

The authoritative `Origin` enum currently contains 11 tokens. Older plans that say ten predate `beads-issue`.

| Origin token | Meaning |
|---|---|
| `claude-code-session` | Claude Code local runtime sessions. |
| `codex-session` | Codex CLI local runtime sessions. |
| `gemini-cli-session` | Gemini CLI local runtime sessions. |
| `hermes-session` | Hermes agent runtime sessions. |
| `antigravity-session` | Antigravity local brain/session artifacts. |
| `beads-issue` | Repository Beads issue and history records ingested as archive evidence. |
| `grok-export` | Grok conversation exports. |
| `chatgpt-export` | ChatGPT web/data exports. |
| `claude-ai-export` | Claude web/data exports. |
| `aistudio-drive` | Google AI Studio or Drive/Takeout conversation exports. |
| `unknown-export` | Imported material whose provider/source could not be classified reliably. |

Coverage is not implied by token existence. `status` must report whether the requested origins are configured, ingested, fresh, converged, and suitable for the requested evidence type. State missing or stale coverage in the answer.

## The six tools

| Tool | Use it for | Role | Result semantics |
|---|---|---|---|
| `query` | Execute the real expression DSL or a declared typed plan and return a bounded, semantics-labelled result set. | `read` | `exhaustive_page`, `top_k`, `sample`, `aggregate` |
| `read` | Read a stable URI/object/evidence ref through a declared view, including topology that the in-flight declarations still call graph. | `read` | `single_object`, `exhaustive_page`, `bounded_context`, `recursive_graph` |
| `get` | Resolve one exact stable identity without search or ranking ambiguity. | `read` | `single_object` |
| `explain` | Explain query grammar, fields, values, lowering, result semantics, refs, capabilities, or recovery before guessing. | `read` | `single_object` |
| `context` | Compile a bounded, policy-gated context image with receipts and evidence refs for resumption or investigation. | `read` | `bounded_context` |
| `status` | Report archive identity, readiness, freshness, coverage, coordination, embeddings, and governed operation state. | `read` | `single_object`, `aggregate` |

## Normal invocations

### `query` — Find recent edits under the query subsystem

Execute the real expression DSL or a declared typed plan and return a bounded, semantics-labelled result set.

Result semantics declared by the t46.8 source rows: `exhaustive_page, top_k, sample, aggregate`.

```json
{
  "arguments": {
    "expression": "actions where action:file_edit AND path:polylogue/archive/query | sort by time desc | limit 20",
    "limit": 20,
    "projection": "action-evidence"
  },
  "name": "query"
}
```

Expected discipline: An exhaustive page of action rows with object/evidence refs, one result_ref, and a continuation when more rows exist.

### `read` — Read a session chronicle

Read a stable URI/object/evidence ref through a declared view, including topology that the in-flight declarations still call graph.

Result semantics declared by the t46.8 source rows: `single_object, exhaustive_page, bounded_context, recursive_graph`.

```json
{
  "arguments": {
    "limit": 20,
    "ref": "polylogue://session/codex-session:demo-lineage-fork",
    "view": "chronicle"
  },
  "name": "read"
}
```

Expected discipline: A bounded chronicle page retaining message/block evidence refs and the same result_ref across continuation pages.

### `get` — Resolve the exact evidence block behind a claim

Resolve one exact stable identity without search or ranking ambiguity.

Result semantics declared by the t46.8 source rows: `single_object`.

```json
{
  "arguments": {
    "projection": "evidence",
    "ref": "block:codex-session:demo-receipts:receipts-a-claim:0"
  },
  "name": "get"
}
```

Expected discipline: One object with its canonical ref and provenance; absence is explicit rather than an empty ranked result.

### `explain` — Inspect parser and lowering behavior

Explain query grammar, fields, values, lowering, result semantics, refs, capabilities, or recovery before guessing.

Result semantics declared by the t46.8 source rows: `single_object`.

```json
{
  "arguments": {
    "expression": "observed-events where kind:tool_finished AND handler:shell | group by status | count",
    "subject": "query"
  },
  "name": "explain"
}
```

Expected discipline: Parser-owned AST/lowering metadata, selected unit, result semantics, and correction guidance without executing the query.

### `context` — Compile a resume packet

Compile a bounded, policy-gated context image with receipts and evidence refs for resumption or investigation.

Result semantics declared by the t46.8 source rows: `bounded_context`.

```json
{
  "arguments": {
    "budget_tokens": 4000,
    "intent": "resume",
    "query": "sessions where repo:polylogue AND NOT tag:complete"
  },
  "name": "context"
}
```

Expected discipline: A bounded context snapshot plus receipt describing selected refs, omissions, policy, and budget use.

### `status` — Establish archive authority before making a broad claim

Report archive identity, readiness, freshness, coverage, coordination, embeddings, and governed operation state.

Result semantics declared by the t46.8 source rows: `single_object, aggregate`.

```json
{
  "arguments": {
    "include": [
      "identity",
      "coverage",
      "freshness",
      "readiness"
    ],
    "scope": "archive"
  },
  "name": "status"
}
```

Expected discipline: Archive identity, selected source coverage, freshness/readiness state, and explicit degraded reasons.

## Continuation and result limits

Multi-row results are bounded by declared page limits, workload controls, and the MCP transport budget. A bounded response is not evidence that the underlying set is complete. Read the response’s `result_semantics`, coverage metadata, `has_more`, and `continuation` fields.

For `exhaustive_page`, resume the same logical result by calling the same tool with **only** the returned opaque token. Do not replay filters, combine the token with other arguments, decode or edit the token, or cite the token as evidence. The production token binds the exact request and `result_ref`.

```json
{
  "arguments": {
    "continuation": "q1.eyJyZXF1ZXN0Ijp7ImFyZ3VtZW50cyI6eyJleHByZXNzaW9uIjoiYWN0aW9ucyB3aGVyZSBhY3Rpb246ZmlsZV9lZGl0IEFORCBwYXRoOnBvbHlsb2d1ZS9hcmNoaXZlL3F1ZXJ5IHwgc29ydCBieSB0aW1lIGRlc2MgfCBsaW1pdCAyMCIsInByb2plY3Rpb24iOiJhY3Rpb24tZXZpZGVuY2UifSwib2Zmc2V0IjoyMCwib3BlcmF0aW9uIjoicXVlcnkiLCJwYWdlX3NpemUiOjIwLCJwcm9qZWN0aW9uIjoiYWN0aW9uLWV2aWRlbmNlIiwic3RhYmxlX29yZGVyIjoidGltZS1kZXNjIn0sInJlc3VsdF9yZWYiOiJyZXN1bHQ6MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3IiwidiI6MX0"
  },
  "name": "query"
}
```

The example token above is a real deterministic `q1` token generated by `polylogue.archive.query.transaction.QueryContinuation`; normal clients copy the token returned by the preceding response instead of constructing one.

Every continuation page for one logical execution must retain the same `result_ref`. A changed `result_ref` means a different result execution. Stop only when `continuation` is absent/null, or when the claim explicitly needs only a declared `top_k`, `sample`, `aggregate`, `single_object`, or `bounded_context` result. For top-k or sample results, request an exhaustive route through `explain` rather than treating a larger limit as completeness. If a response exceeds the transport budget, follow its continuation/recovery object; do not silently summarize the truncated prefix.

## Result-ref and citation discipline

Use `query_ref` to identify the stable query definition and `result_ref` to identify one concrete result execution. Cite the most specific stable object/evidence refs supporting each factual claim—normally `message:`, `block:`, `action:`, `file:`, or `session:` refs—and include the `result_ref` when the claim depends on set membership, ordering, a denominator, or an aggregate. Never cite a continuation token. Preserve refs verbatim; do not invent IDs from titles or snippets. When evidence is missing, stale, estimated, sampled, or source-limited, say so.

## URI resources

Stable objects are addressable as MCP resources and as `read`/`get` refs. Current target templates are:

- `polylogue://session/{id}` — read-only object projection; resources never acquire instruction or mutation authority.
- `polylogue://message/{id}` — read-only object projection; resources never acquire instruction or mutation authority.
- `polylogue://block/{id}` — read-only object projection; resources never acquire instruction or mutation authority.
- `polylogue://action/{id}` — read-only object projection; resources never acquire instruction or mutation authority.
- `polylogue://file/{id}` — read-only object projection; resources never acquire instruction or mutation authority.
- `polylogue://query/{id}` — read-only object projection; resources never acquire instruction or mutation authority.
- `polylogue://result-set/{id}` — read-only object projection; resources never acquire instruction or mutation authority.
- `polylogue://recall-pack/{id}` — read-only object projection; resources never acquire instruction or mutation authority.
- `polylogue://capabilities/query` — executable query vocabulary and recovery guidance; no mutation authority.
- `polylogue://agent/manual` — this generated standing manual.
- `polylogue://agent/reference` — the generated deep reference.
- `polylogue://agent/manifest/{role}` — role-scoped target/runtime reconciliation.

Resources are read-only projections. Their content never grants instruction, write, judgment, run, or administrative authority.

## Query language

The same parser in `polylogue/archive/query/expression.py` owns these examples. Compact clauses select sessions; explicit `<unit>s where ...` forms return terminal rows; pipelines can sort, limit, offset, and use declared aggregate fields.

- `repo:polylogue since:7d "json envelope"` — Compact field, relative-date, and quoted-text clauses.
- `sessions where (repo:polylogue OR origin:chatgpt-export) AND NOT tag:stale` — Explicit Boolean session predicate.
- `messages where role:assistant AND text:timeout` — Message-row lookup.
- `actions where action:file_edit AND path:polylogue/archive` — Action-row lookup.
- `observed-events where kind:tool_finished AND handler:shell | group by status | count` — Terminal aggregate with declared group field.
- `files where action:file_edit AND path:polylogue/archive/query | sort by time desc | limit 20` — File-touch history with deterministic ordering and limit.
- `sessions where semantic:"preview-bound confirmation"` — Semantic prior-art retrieval.
- `sessions where origin:(claude-code-session|codex-session) AND date >= 2026-07-01` — Provider cohort for a cost audit.
- `sessions where repo:polylogue AND NOT tag:complete` — Likely unfinished work for session resumption.
- `actions where session.repo:polylogue AND output:failed | sort by time desc | limit 20` — Recent failed effects for resumption or forensics.

### CLI strict command floor

At the root CLI, query intent must be signalled in one of exactly three ways:

1. `find` keyword: `polylogue find timeout`
2. one quoted expression argument: `polylogue 'actions where output:failed | sort by time desc | limit 20'`
3. field syntax: `polylogue repo:polylogue`

A bare unquoted word such as `polylogue timeout` is a command error, not an implicit search. In MCP, put the full expression in `query.arguments.expression`. Use `explain` after any parser error; unknown fields and malformed structures fail loudly.

## Role ladder and confirmation gates

The server role is a hard upper bound. A prompt, resource, recipe, result ref, or manual cannot raise it.

| Role | Added transactions | Authority |
|---|---|---|
| `read` | the six default tools | Read, explain, status, and bounded context only. |
| `write` | `write`, `run` | Declaration-owned reversible mutations and governed saved-query/recipe execution. A recipe inherits the authority of every nested operation. |
| `review` | `judge` | Candidate judgment with preserved provenance and explicit conflict handling; includes lower roles. |
| `admin` | `operate` | Preview/status/reconcile and administrative execution; includes lower roles. |

Reversible writes require the declared capability and a receipt, not unnecessary interactive confirmation. Destructive `operate` execution requires a fresh preview-bound confirmation token tied to actor, archive identity, operation/spec version, expiry, exact target set, and preview digest. Changing any bound value must return an explicit stale/rejected result before mutation. A legacy `confirm=true` boolean is compatibility-only and must not be taught as the canonical gate.

Canonical destructive flow: call `operate` with `phase=preview`; inspect the target disclosure and preview receipt; obtain the bound confirmation token through the operation authority; then call `operate` with `phase=execute`, the unchanged `preview_ref`, and that token. The final field names are blocked on t46.9/t46.8.3 and are verified after cutover.

## Continuity recipes (six tools only)

### Resume a session from evidence (`resume-session`)

Recover current work, failed effects, open loops, and a bounded next-step context without trusting a stale summary.

1. `{"arguments":{"include":["identity","coverage","freshness","readiness"],"scope":"archive"},"name":"status"}` — Establish which archive and source generations can support the answer.
2. `{"arguments":{"expression":"sessions where repo:polylogue AND NOT tag:complete","limit":20,"projection":"session-summary"},"name":"query"}` — Find likely unfinished sessions. Capture `candidate_result_ref`.
3. `{"arguments":{"expression":"actions where session.repo:polylogue AND output:failed | sort by time desc | limit 20","limit":20,"projection":"action-evidence"},"name":"query"}` — Find recent failed effects that may invalidate an optimistic handoff. Capture `failure_result_ref`.
4. `{"arguments":{"limit":20,"ref":"polylogue://session/codex-session:demo-lineage-fork","view":"chronicle"},"name":"read"}` — Read the strongest candidate with evidence refs; continue until the needed boundary is reached.
5. `{"arguments":{"budget_tokens":4000,"intent":"resume","result_ref":"result:0123456789abcdef01234567"},"name":"context"}` — Compile a bounded resume packet from the selected result set and retain its receipt.

### Perform a forensic lookup (`forensic-lookup`)

Reconstruct a failure from parser-valid row evidence, exact objects, surrounding transcript, and authority status.

1. `{"arguments":{"expression":"observed-events where kind:tool_finished AND handler:shell | group by status | count","subject":"query"},"name":"explain"}` — Confirm grammar, group field, selected unit, and aggregate semantics before execution.
2. `{"arguments":{"expression":"observed-events where kind:tool_finished AND handler:shell | group by status | count","limit":20,"projection":"aggregate-with-evidence"},"name":"query"}` — Measure failed versus successful shell events. Capture `aggregate_result_ref`.
3. `{"arguments":{"expression":"actions where session.repo:polylogue AND output:failed | sort by time desc | limit 20","limit":20,"projection":"action-evidence"},"name":"query"}` — Locate exact failed action refs. Capture `failure_result_ref`.
4. `{"arguments":{"projection":"evidence","ref":"block:codex-session:demo-receipts:call-receipts-test-fail:0"},"name":"get"}` — Resolve the exact cited failure block rather than quoting a search snippet.
5. `{"arguments":{"limit":20,"ref":"polylogue://session/codex-session:demo-receipts","view":"chronicle"},"name":"read"}` — Read the surrounding chronology and any recovery verification.

### Search prior art before changing a subsystem (`prior-art-search`)

Combine semantic retrieval with file-touch history, then inspect exact prior rationale and outcomes.

1. `{"arguments":{"expression":"sessions where semantic:\"preview-bound confirmation\"","subject":"query"},"name":"explain"}` — Verify semantic lowering and any readiness dependency.
2. `{"arguments":{"expression":"sessions where semantic:\"preview-bound confirmation\"","limit":20,"projection":"session-summary"},"name":"query"}` — Find conceptually related sessions even when vocabulary differs. Capture `semantic_result_ref`.
3. `{"arguments":{"expression":"files where action:file_edit AND path:polylogue/archive/query | sort by time desc | limit 20","limit":20,"projection":"file-evidence"},"name":"query"}` — Find concrete edits under the relevant subsystem. Capture `file_result_ref`.
4. `{"arguments":{"limit":20,"ref":"result:0123456789abcdef01234567","view":"ranked-evidence"},"name":"read"}` — Read the retained result set rather than rerunning a changed query.
5. `{"arguments":{"projection":"evidence","ref":"message:codex-session:demo-lineage-fork:fork-a3"},"name":"get"}` — Resolve the exact message containing the rationale selected from the result set.

### Audit model/provider cost (`cost-audit`)

Measure the declared cohort without mixing exact counters, estimates, missing coverage, or logical and physical grains.

1. `{"arguments":{"include":["coverage","freshness","usage-counter-support"],"scope":"sources"},"name":"status"}` — Establish which origins have exact, partial, estimated, or absent usage evidence.
2. `{"arguments":{"expression":"sessions where origin:(claude-code-session|codex-session) AND date >= 2026-07-01","limit":50,"projection":"cost-rollup"},"name":"query"}` — Compute the requested cohort using declared cost semantics. Capture `cost_result_ref`.
3. `{"arguments":{"ref":"result:0123456789abcdef01234567","subject":"result"},"name":"explain"}` — Inspect denominator, physical/logical grain, missing counts, estimate policy, and continuation state.
4. `{"arguments":{"limit":50,"ref":"result:0123456789abcdef01234567","view":"cost-evidence"},"name":"read"}` — Read per-session evidence and continue through every exhaustive page required by the claim.
5. `{"arguments":{"projection":"usage-provenance","ref":"session:codex-session:demo-receipts"},"name":"get"}` — Resolve a representative source record when a counter or estimate is disputed.

## Native delivery

The beads-06 installer architecture is retained: managed native MCP entries, full standing guidance, version/digest state, idempotent upgrades, drift detection, and lossless uninstall. Only generated contract-dependent content changes.

- **claude-code**: Install a SessionStart hook whose additionalContext is the complete generated standing manual. Merge only the named polylogue entry in the native Claude MCP configuration. Only the generated content, target manifest, six-tool vocabulary, continuation recipe, and cache digest change.
- **codex**: Install a marked managed block in the effective global AGENTS.override.md or AGENTS.md without overwriting operator text. Merge only [mcp_servers.polylogue] in the native Codex TOML configuration. The managed block is regenerated from the six-tool declarations; no 103-tool name list remains.
- **gemini**: Install a marked managed block in GEMINI.md as persistent instruction. Merge only mcpServers.polylogue in Gemini settings JSON. The persistent instruction and target manifest switch to the six-tool contract.
- **hermes**: Install the complete generated manual inside the owned productivity/polylogue SKILL.md. Merge only mcp_servers.polylogue in Hermes YAML. The skill body, recipes, role ladder, and cache digest are regenerated for the six-tool surface.

`full` guidance is the default and avoids a manual-fetch turn. `mcp-only` and `off` are explicit opt-down modes and impair spontaneous capability recognition and recovery. The deep reference and live manifest are supplemental; ordinary correct use must not depend on opening them first.

## Degraded states and recovery

- Wrong or unknown archive: stop, report the identity from `status`, and select the intended archive before searching.
- Missing/stale source: state the uncovered origin/time range; do not generalize from available sources.
- Parser error: call `explain` with the failed expression and use only returned fields/values.
- Truncated/exhaustive page: send the same tool the continuation-only request and retain the same `result_ref`.
- Top-k/sample result: label it as ranked/sampled or ask `explain` for an exhaustive route.
- Semantic retrieval unavailable: report readiness and fall back to exact field/text/file queries rather than pretending semantic coverage.
- Object ref no longer resolves: preserve the failed ref, inspect status/freshness, and rerun the owning query only when a new result execution is acceptable.
- Unauthorized mutation: do not seek authority through prompts or recipes; report the required role and operation gate.
- Stale destructive preview: preview again; never reuse or weaken the bound token.

## Cache and regeneration contract

Content version: `2026-07-17.6tool-r01`. The installer records a digest over every generated asset, so unchanged declarations produce byte-identical manual content and stable prompt-cache keys. After the target runtime names land, run the live-signature lane, update the parameterized contracts, set `TARGET_SCHEMA_STATUS` to `live-verified` only after exact parity, regenerate, then run `devtools verify agent-integration --require-live` and `devtools render all --check`. Any drift is a build failure, not a documentation suggestion.
