# Test design and execution record

## Test philosophy

The acceptance design observes production dependencies rather than validating copied prose. Generated calls resolve through typed contracts sourced from the MCP declaration registry; query examples use the real expression parser; continuation examples use the real transaction codec; server tests build the real FastMCP composition; installer tests exercise native configuration and ownership in isolated homes; package checks inspect the built wheel and source distribution.

The cutover-final FastMCP schema lane is intentionally separate from static compilation. It cannot pass merely because a staged manual is internally consistent. It requires the runtime role-scoped tool names and exact argument names, required fields, and defaults to match the generated contracts, followed by an explicit `live-verified` schema state.

## Verification lanes and anti-vacuity mutations

| Lane | Production dependency exercised | Representative mutation/removal that must fail it |
| --- | --- | --- |
| generated-assets | `polylogue.agent_integration.spec`, renderer output, package-resource digest, public docs copies | Edit a generated asset, reorder a declaration without regeneration, remove one package asset, or change the digest algorithm input order. |
| manual-compilation | target contracts, current declaration mappings, every generated call, role/result/resource/prompt metadata, authoritative Origin enum | Rename/drop an argument, remove `graph` from the `read` source mapping while it remains separate, change a result class, delete a resource/prompt, or remove `beads-issue` from the manual while it remains in `Origin`. |
| query-parser-roundtrip | `polylogue.archive.query.expression.parse_expression_ast`, AST serialization, strict command-floor predicates | Remove a grammar field/operator, change quoted-expression routing, accept an unmarked bare word, or stop recognizing leading `find`/field syntax. |
| continuation-contract | `polylogue.archive.query.transaction.QueryContinuation`, opaque token decode, request/result-ref binding, target contracts | Change the expression, projection, offset, page size, stable order, or result ref after token creation; mix initial fields with `continuation`; mark a tool resumable without declaring `continuation`; remove `run` continuation. |
| target-declaration-reconciliation | current MCP declaration registry, target source mappings, runtime inventories, dual activation gates | Claim names-only cutover readiness, map `operate` to no source, expose target names before the final runtime set, or set the live marker without exact FastMCP parity. |
| native-installer-roundtrip | beads-06 ownership ledger and native adapters for Claude Code, Codex, Gemini CLI, Hermes | Overwrite operator content, rewrite an unchanged install, lose ownership on upgrade, hide drift, remove another client's state, grant a broader role, or leave owned files after exact uninstall. |
| packaging-and-home-manager | package data, source distribution, flake export, Home Manager module separation | Remove any of the six generated assets, omit installer/CLI code, drop the flake export, introduce daemon ownership into the agent module, or emit secret-bearing data. |
| live-fastmcp-signatures | actual role-scoped FastMCP tool registration and inspected input schemas | Add/drop/rename a target field, change required/default status, retain a compatibility tool, separate graph after the target says `read`, or expose a different `operate` preview/token/execute contract. |

## Focused unit and integration tests

### Agent assets, CLI, installer, and MCP composition

Command:

```bash
.venv/bin/pytest -q \
  tests/unit/agent_integration \
  tests/unit/mcp/test_server_surfaces.py
```

Result: `114 passed in 18.89s`.

Coverage includes:

- package resource reads, metadata, digest, and CLI manual/manifest output;
- fail-closed `agent install` behavior before live cutover;
- role-scoped target/current manifests and dual-gate activation;
- all four native clients in temporary homes;
- idempotence, upgrade, role change, drift, operator-owned conflicts, and exact uninstall;
- generated manual/reference/recipe/contract invariants;
- real `q1` continuation binding and resume-only shape;
- MCP instructions before/after a simulated complete gate;
- all three `polylogue://agent/*` resources and role validation.

Anti-vacuity examples in the suite include asserting that a names-only registration still cannot activate the staged schema, that every resumable tool exposes `continuation`, and that direct unconditional standing-manual injection changes the server-surface expectations.

### CLI discovery and command contract

Command:

```bash
.venv/bin/pytest -q \
  tests/unit/cli/test_help_snapshots.py \
  tests/unit/cli/test_help_contract.py \
  tests/unit/cli/test_click_app.py \
  tests/unit/cli/test_cli_action_contracts.py
```

Result: `364 passed in 4.33s`; `1 snapshot passed`.

These tests exercise real root command registration, help rendering, the strict command floor, explicit query intent, and action-contract discovery. Removing `agent` registration or changing its generated help row fails this group. Changing bare-word/quoted/find/field routing fails existing strict-floor tests.

### Generated repository surfaces and topology

Command:

```bash
.venv/bin/pytest -q \
  tests/unit/devtools/test_command_catalog.py \
  tests/unit/devtools/test_generated_surfaces.py \
  tests/unit/devtools/test_render_all.py \
  tests/unit/devtools/test_render_cli_reference.py \
  tests/unit/devtools/test_render_devtools_reference.py \
  tests/unit/devtools/test_render_docs_surface.py \
  tests/unit/devtools/test_render_topology_status.py \
  tests/unit/devtools/test_topology_gates.py
```

Result: `45 passed in 13.46s`.

This group proves that the new generator/verifier are discoverable through the repository command catalog, that the manual assets are owned generated surfaces, that documentation indexes and CLI/devtools references remain synchronized, and that topology generation accepts the new module relationships. Removing a generated-surface declaration or editing generated docs without regeneration fails it.

A prior combined shell invocation completed the first two groups and then hit the execution harness timeout shortly after the third group began. The third group was rerun by itself and passed completely; no test assertion failed.

### Focused pytest total

`523 passed, 0 failed` across the three completed groups.

## Renderer and verifier commands

Commands:

```bash
.venv/bin/python devtools/render_agent_manual.py --check
.venv/bin/python -m devtools.verify_agent_integration --json
.venv/bin/python -m devtools.render_all --check
.venv/bin/python -m devtools.verify_topology --json
```

Results:

- manual renderer: deterministic, no drift;
- integration verifier: `7 pass`, `0 fail`, `1 unverified`;
- all repository generated surfaces: synchronized;
- topology verification: non-blocking, with no orphan, missing, conflict, or kernel-rule failure; nine pre-existing target-storage locations remain marked TBD in the repository's topology model.

The live requirement was checked separately:

```bash
.venv/bin/python -m devtools.verify_agent_integration --require-live
```

Result: exit status `1`, expected. The blocker is the current 104-tool compatibility registration. The lane reports the exact missing checks: final argument names and required/default states, continuation exclusivity, graph ownership under `read`, and `operate` preview/token/execute fields.

## Static quality checks

Command:

```bash
.venv/bin/ruff check \
  polylogue/agent_integration \
  polylogue/cli/commands/agent.py \
  polylogue/mcp/server.py \
  polylogue/mcp/server_resources.py \
  devtools/render_agent_manual.py \
  devtools/verify_agent_integration.py \
  tests/unit/agent_integration \
  tests/unit/mcp/test_server_surfaces.py
```

Result: `All checks passed!`

Command:

```bash
.venv/bin/mypy --strict \
  polylogue/agent_integration \
  polylogue/cli/commands/agent.py \
  polylogue/mcp/server.py \
  polylogue/mcp/server_resources.py \
  devtools/render_agent_manual.py \
  devtools/verify_agent_integration.py \
  tests/unit/agent_integration \
  tests/unit/mcp/test_server_surfaces.py
```

Result: `Success: no issues found in 15 source files`.

`git diff --check` also passed.

## Fresh-application proof

`PATCH.diff` was applied to a new detached worktree at `536a53efac0cbe4a2473ad379e4db49ef3fce74d`.

Commands:

```bash
git apply --check PATCH.diff
git apply PATCH.diff
git diff --check
PYTHONPATH="$fresh_checkout" python devtools/render_agent_manual.py --check
PYTHONPATH="$fresh_checkout" python -m devtools.verify_agent_integration --json
```

Results:

- patch applicability: pass;
- binary-safe topology patches: applied;
- whitespace/error check: pass;
- deterministic renderer: pass;
- verifier reproduced `7 pass`, `0 fail`, `1 unverified`.

## Distribution build and installed-wheel smoke

Build command:

```bash
uv build --wheel --sdist --out-dir /tmp/dist-agent-final
```

Results after the final contract changes:

| Artifact | Bytes | SHA-256 | Inventory |
| --- | ---: | --- | --- |
| `polylogue-0.2.0-py3-none-any.whl` | 4,241,414 | `a0fff9fd65c35e77cf84ea2e8c3c407d92efdea46ffb884b0515ae01c47592e3` | all 13 expected Python code/data paths present |
| `polylogue-0.2.0.tar.gz` | 33,993,445 | `a1b23a72a0ab02efff5b9d56870d02a3cf21170cde85f3b67b59bc7e9191239e` | all 14 expected paths present, including `nix/agent-integration-module.nix` |

The wheel was installed with `--no-deps` into a clean target and imported from that target outside the source checkout. Results:

- six generated assets loaded through `importlib.resources`;
- digest matched `9635a669c7510574702c579f3b99924146235d129a93157b36f6cd41d97b709e`;
- read-role compatibility inventory was 66 tools;
- `tool_names_registered = false`;
- `contract_schemas_verified = false`;
- `cutover_ready = false`.

The build did not modify `uv.lock`; its SHA-256 remained `de6874fc1d719617f02349280dff9dce6b3cd35c6f12a5e28c3610e8af90a727`.

## Checks not executable in this environment

- Nix parse/evaluation and flake checks: neither `nix` nor `nix-instantiate` is installed.
- Live FastMCP target signatures: cutover-final registration is absent.
- Live daemon/archive behavior: no operator daemon, archive, secrets, browser, or deployment was accessed.
- Native client process launch: Claude Code, Codex, Gemini CLI, and Hermes binaries were not invoked; native file/config behavior was tested in isolated homes.
- Clean public/demo archive invocation and failure-mode drills: require a live cutover server.
- Cold-agent blind trials and ablation: require final discovery schemas and supported client runtimes.

These remain explicit acceptance work, not implicit passes.
