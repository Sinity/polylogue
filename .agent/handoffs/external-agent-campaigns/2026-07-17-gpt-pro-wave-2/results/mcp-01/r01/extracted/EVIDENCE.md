# Source, Beads, history, and contradiction evidence

## Snapshot provenance

The supplied project-state archive records:

- source: `/realm/project/polylogue`;
- generated: `2026-07-17T180950Z`;
- branch: `master`;
- commit: `536a53efac0cbe4a2473ad379e4db49ef3fce74d`;
- dirty: `true`.

The commit is `fix(repair): harden raw authority convergence (#3046)`, authored at `2026-07-17T18:55:47+02:00`. The snapshot's branch-delta log, file list, and patch against the remote default branch are empty.

The all-refs bundle was used to reconstruct a detached checkout at the exact commit. A byte comparison of the packaged working-tree source against every tracked blob at that commit found:

- 3,864 tracked paths in the commit;
- 2,564 packaged tracked paths present and byte-identical;
- 1,300 tracked paths omitted by snapshot packaging;
- omitted paths: 1,290 under `.agent`, six under `.beads`, two under `.claude`, plus `flake.lock` and `uv.lock`;
- zero byte differences among packaged tracked files.

This supports using the named commit as product-source authority. It does not reconstruct the exact dirty local-state patch, so the handoff preserves that limitation.

## All-ref history and beads-06 intake

The base commit's ancestry does not contain the external beads-06 implementation. A later all-ref commit exists:

```text
7d918eb08 chore(campaign): intake latest external reports (#3047)
```

That ref adds the external report package under the campaign handoff area. The extracted `beads-06` result contains `HANDOFF.md`, `PATCH.diff`, `TESTS.md`, and `EVIDENCE.md`. Applying it to a temporary checkout showed a complete typed spec/generator/installer/CLI/Nix architecture for the 103-tool-era surface.

Architectural comparison established:

- `nix/agent-integration-module.nix` in this result is byte-identical to beads-06;
- the ownership-aware installer retains the beads-06 data model, transaction behavior, native adapters, conflict rules, state schema, and lifecycle commands;
- installer changes are current-source imports, direct `MCPRole` use, strict typing, formatting, and a lint annotation preserving the public exception name;
- the old generated content and old verifier were not reused verbatim because they advertise compatibility-surface tools and therefore cannot satisfy the six-tool mission.

This is reuse of the existing manual system, not a competing implementation.

## Beads authority

### `polylogue-3gd.2` — comprehensive standing manual

The Bead requires project-owned comprehensive guidance in standing context, without a preliminary fetch turn. It requires examples, MCP names, expressions, fields, result claims, and role claims to compile or execute against production declarations; it also calls out stale invalid expressions, nonexistent tool names, archive-root confusion, response-budget continuation, unconverged archives, incomplete coverage, and orchestration reconstruction as regression fixtures.

Result alignment:

- canonical typed declarations and generated package assets;
- complete standing manual plus deep reference;
- parser/continuation/declaration compilation lanes;
- no activation while schemas are staged;
- explicit degraded-state recovery;
- no hand-maintained client tool lists.

### `polylogue-3gd.3` — package and native client delivery

The Bead requires wheel/sdist/Nix inclusion, `agent install/status/doctor/uninstall`, isolated-home idempotence and drift handling, Home Manager options, daemon separation, clean-home smoke, upgrade preservation, standing-guidance size/digest reporting, and explicit opt-down disclosure.

Result alignment:

- beads-06 installer retained;
- all four native client declarations retained and regenerated;
- generated asset digest/cache key exposed;
- package assets verified in wheel/sdist;
- Home Manager module retained and flake-exported;
- daemon lifecycle remains outside the module;
- native configuration behavior verified in temporary homes;
- live client binary and Nix evaluation remain unverified in this container.

### `polylogue-t46.8` — protocol-native algebra

The Bead states that the tool-count problem is a choice-architecture failure, not merely stale counting or token cost. It requires a small expressive algebra, URI resources, prompts, role-scoped discovery, bounded resumability, semantic equivalence, and cold-model success before old tools are removed. Its design explicitly says MCP must remain a leaf adapter over shared declarations/parser/execution rather than create parallel policy.

Result alignment:

- target algebra is six default-read transactions plus four privileged transactions;
- stable resources and prompts remain typed declaration references;
- compatibility inventory is reported, not hidden;
- no old tool is removed by this patch;
- activation is blocked until registration and signatures match;
- parser, continuation, role, and result semantics use production owners.

### `polylogue-t46.8.1` — declaration/equivalence inventory

The Bead identifies `polylogue/mcp/declarations/models.py` and `registry.py` as the domain declaration pilot, with fields for role, result semantics, invocation, continuation/query/result refs, resources/prompts, workflow coverage, and compatibility route. It permits up to fifteen default transactions and explicitly retains old implementations during this slice.

Current source evidence:

- `polylogue/mcp/declarations/registry.py:3376` begins `TARGET_DEFAULT_READ_ALGEBRA`;
- entries are `query` at 3378, `read` at 3392, `graph` at 3414, `context` at 3432, and `status` at 3441, plus `get`/`explain` in the same block;
- `PRIVILEGED_ALGEBRA` starts at 3451 with `write`, `judge`, `run`, and `maintenance`;
- the current registry still treats graph as a separate transaction and uses `maintenance` rather than `operate`.

Result alignment:

- the generated target contract is derived from these declarations where possible;
- source mappings are recorded in generated JSON and verifier output;
- current `read` and `graph` are normalized into target `read`;
- current `maintenance` is normalized into target `operate`;
- final signatures remain explicitly parameterized.

### `polylogue-t46.8.2` — read migration

The Bead owns migration to `query/read/get/explain` and stable URI resources over the shared bounded query transaction. It requires physical paging without logical truncation, stable refs/cursors, explicit top-k/sample/aggregate semantics, cancellation, and no adapter-side accumulation.

Result alignment:

- manual terminology distinguishes physical page limits from logical completeness;
- continuation examples use the shared query transaction token;
- result semantics are encoded per contract;
- URI resource patterns and ref citation are taught;
- runtime execution/cancellation equivalence remains blocked on this Bead landing.

### `polylogue-t46.8.3` — privileged/context migration

The Bead requires context, assertion, judgment, recipe/run, coordination, and maintenance capabilities to map to typed verbs/resources/prompts without weakening authority. It explicitly says prompts/resources cannot acquire write authority and that candidate/judged state, dry-run/authorization, receipts, recovery, and role isolation must survive.

Result alignment:

- `context`, `write`, `judge`, `run`, and `operate` are declared with role/capability/result semantics;
- prompts are references/workflows, not authority;
- install manifests are role scoped;
- final runtime equivalence is not claimed.

### `polylogue-t46.9` — mutation authority

The Bead makes `OperationSpec`/`OperationExecutor` the single executable mutation authority. It requires destructive confirmation tokens bound to actor, archive identity, operation version, expiry, and exact target/preview digest; stale bindings must return an explicit rejection before mutation. Legacy confirm booleans are interim adapters only.

Result alignment:

- the manual teaches preview/bound-token/execute/receipt behavior;
- `operate` is not implemented as a generic ungoverned mutation endpoint;
- the patch does not add a parallel MCP authorization policy;
- final field names and execution binding remain a live-schema blocker.

## Production-source anchors

### Current role and declaration model

- `polylogue/mcp/declarations/models.py:11` defines `MCPRole = Literal["read", "write", "review", "admin"]`.
- `polylogue/mcp/declarations/registry.py:3376-3520` declares the in-flight target transaction inventory.
- `tests/infra/mcp.py` and the declaration registrar provide expected/runtime inventory authority.
- The verifier observes 104 runtime tools across the full role surface and 66 in the read role.

### Real continuation protocol

`polylogue/archive/query/transaction.py:79-126` defines `QueryContinuation`:

- it serializes operation, arguments, page size, offset, projection, stable order, and result ref;
- canonical JSON is URL-safe base64 encoded;
- the public prefix is `q1.`;
- decode rejects the wrong version, malformed body, and a result ref without the `result:` prefix.

`QueryResultPage` begins at line 129 and derives the next advancing offset from actual returned rows. The generated manual token is created by this implementation rather than copied text.

### Real expression parser and strict floor

- `polylogue/archive/query/expression.py:2478` exposes `parse_expression_ast`.
- `polylogue/cli/query_group.py:94-105` recognizes quoted/multiword and field-syntax query intent.
- `polylogue/cli/query_group.py:298-305` refuses an unsignalled bare root and points users toward `find` or quoting.
- `polylogue/cli/root_request.py:24-30` validates field prefixes against the real expression field registry rather than treating any colon as DSL.
- `polylogue/cli/root_request.py:33-58` recognizes structured quoted expressions such as terminal `where` clauses and compact field filters.

The generated verifier checks leading `find`, quoted expression, registered field syntax, and bare-word refusal against those production routes.

### Authoritative Origin vocabulary

`polylogue/core/enums.py:42-55` defines eleven tokens:

1. `claude-code-session`
2. `codex-session`
3. `gemini-cli-session`
4. `hermes-session`
5. `antigravity-session`
6. `beads-issue`
7. `grok-export`
8. `chatgpt-export`
9. `claude-ai-export`
10. `aistudio-drive`
11. `unknown-export`

The generated spec has an import-time equality check between its meanings and the enum order. A stale ten-token manual cannot render successfully.

### Activation guard

`polylogue/agent_integration/manifest.py:42-64` separates:

- exact target/runtime name equality;
- explicit live-schema verification;
- final activation requiring both.

`build_live_manifest` at lines 67-112 reports current tools, target tools, missing target tools, compatibility tools remaining, resources, prompts, counts, and both gate values. This prevents a future names-only migration from activating parameterized calls.

### Native installation ownership

`polylogue/agent_integration/installer.py` retains the external implementation's:

- state schema and self-digest;
- atomic transaction/rollback behavior;
- JSON, YAML, TOML, marked-block, and managed-file ownership records;
- separate native layouts for Claude Code, Codex, Gemini CLI, and Hermes;
- no-rewrite idempotence;
- drift/conflict detection;
- exact owned-operation uninstall and empty-directory cleanup.

The Home Manager module remains separate from daemon modules and only invokes `polylogue agent install` for selected clients/options.

## Contradictions adjudicated

| Statement | Falsifying/current evidence | Implemented resolution |
| --- | --- | --- |
| Compatibility surface has 103 tools. | Runtime declaration inventory is 104. | Generated manifest and handoff report 104. |
| Default read surface is already exactly six tools. | Registry has separate `graph`, producing seven in-flight rows. | Target has six; `read` derives from current `read` and `graph`; final signature lane remains blocked. |
| Privileged public transaction is already `operate`. | Registry declares `maintenance`. | Target public name is `operate`, mapped from current `maintenance` until cutover. |
| There are ten Origin tokens. | Current enum contains eleven because of `beads-issue`. | Manual includes all eleven and explicitly identifies the stale count. |
| Generated examples can be proven against final schemas now. | Final FastMCP target signatures are absent and compatibility names remain. | Static compilation passes; live verification is unverified; installation/instruction activation fails closed. |
| A manual resource alone satisfies no-fetch standing guidance. | Beads 3gd requires standing native context. | Resources are supplemental; SessionStart/persistent native delivery remains the default after live activation. |
| MCP can define its own confirmation semantics. | t46.9 assigns authority to shared `OperationSpec`/executor. | Manual follows shared preview-token intent; no parallel policy is added. |

## Generated and runtime measurements

Final generated measurements:

- asset version: `2026-07-17.6tool-r01`;
- asset digest: `9635a669c7510574702c579f3b99924146235d129a93157b36f6cd41d97b709e`;
- cache key: `polylogue-agent-2026-07-17.6tool-r01-9635a669c7510574`;
- standing manual: 23,331 bytes;
- deep reference: 29,757 bytes;
- packaged data assets: six;
- target transactions: ten;
- target read transactions: six;
- continuity recipes: four;
- parser examples: ten;
- Origin tokens: eleven;
- full compatibility runtime tools: 104;
- read-role compatibility runtime tools: 66;
- live verifier: seven pass, zero fail, one unverified.

## Evidence that would falsify the design

The implementation should be rejected or revised if any of the following is observed after cutover:

- final FastMCP argument names/defaults cannot be represented by the typed contracts without hand-written exceptions;
- `read` does not actually own graph/topology semantics;
- `operate` bypasses shared OperationSpec preview/token/receipt authority;
- continuation accepts mixed initial fields or does not bind the original request/result ref;
- a supported client cannot receive the complete standing manual through its documented persistent mechanism;
- installer upgrade or uninstall modifies operator-authored content;
- wheel/sdist omit generated assets or produce a different digest from source;
- a parser example only succeeds in the verifier because it bypasses production parsing;
- cold-agent trials select removed compatibility routes or make unsupported evidence claims despite the standing manual;
- the six-tool surface loses exhaustive logical access or disguises top-k/sample/aggregate limits.

Those checks are the high-value targets for the mandatory post-cutover pass.
