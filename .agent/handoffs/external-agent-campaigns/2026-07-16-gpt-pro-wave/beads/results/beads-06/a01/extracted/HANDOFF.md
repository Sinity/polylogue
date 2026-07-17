# Polylogue executable agent manual and installation kit — handoff r04

## Mission and outcome

This package completes the implementation and delivery work for job `beads-06`, primarily covering Beads `polylogue-3gd.2` and `polylogue-3gd.3`. It adds a project-owned, executable cold-start manual; a deeper local reference; typed recipe and capability declarations; live role-scoped MCP manifests; user-scoped native installation for Claude Code, Codex, Gemini CLI, and Hermes; a separate Home Manager module; and verification that compiles or resolves the generated routes against current production declarations.

The implementation is supplied as one apply-ready `PATCH.diff`. `FILES/` is intentionally omitted because the unified diff contains every new file in full and applies without ambiguity to the named commit.

The strongest self-contained portion is complete: a clean clone of the exact snapshot commit accepted the patch, reproduced all 29 changed files byte-for-byte, passed the 22 focused agent-integration tests, passed generated-manual and topology checks, passed all six executable integration-verifier lanes under an isolated registration/import harness, and built matching wheel/sdist artifacts whose five packaged agent assets were byte-identical. External-runtime claims remain explicitly unverified where this shell lacked the real dependency closure, Nix/Home Manager tooling, live clients, or a live Polylogue archive/daemon.

## Snapshot identity and authority

The attached project-state archive is the code authority. Its manifest reports:

- Project: `polylogue`
- Branch: `master`
- Commit: `f654480cadb7cc4c194704e24dfd483199547b35`
- Commit subject: `chore(beads): file archive-insight benchmark findings wofr/vhjs/r7p6/1wtm`
- Commit author date: `2026-07-17T03:45:52+02:00`
- Snapshot generated: `2026-07-17T043202Z`
- Snapshot metadata dirty flag: `true`

The bundled `polylogue-branch-delta.patch`, branch file list, and branch log are all empty and have the SHA-256 of an empty file. The exact bundled Git commit is therefore the patch base. The working-tree tar export omitted some tracked repository-local surfaces, including portions of `.agent`, `.beads`, `.claude`, `flake.lock`, `uv.lock`, and other metadata. Those archive-packaging omissions were not interpreted as source deletions. The clean base was reconstructed from the bundled all-refs Git object store, and `PATCH.diff` is based on the exact commit above.

## Evidence inspected

The implementation followed dependencies beyond the obvious manual files. The inspection covered:

- Repository rules in `CLAUDE.md`, including substrate-first semantics, lazy CLI registration, MCP surface ownership, generated-surface discipline, and the requirement to regenerate topology when adding any `polylogue/` module.
- Packaging and release contracts in `pyproject.toml`, `flake.nix`, `nix/`, and Hatchling build behavior.
- CLI command registration, root dispatch, daemon entry points, config/path resolution, demo commands, status/readiness commands, import surfaces, and query command parsing.
- MCP server construction, role gating, tool/resource/prompt registration, payload serialization, response budgets, and the existing MCP test manifest.
- Existing documentation generation and topology verification infrastructure.
- Beads export records for `polylogue-3gd.2` and `polylogue-3gd.3`, including their later operator-contract notes that supersede pointer-only or consumer-owned guidance designs.
- Relevant history for `polylogue/mcp/server.py`, `polylogue/mcp/server_resources.py`, `polylogue/cli/click_command_registration.py`, and `flake.nix`.
- Current official native-client contracts for Claude Code hooks/MCP, Codex MCP and global instruction precedence, Gemini CLI settings/MCP, and Hermes configuration/skills.
- Existing and newly added unit-test seams, distribution checks, generated-surface checks, and a direct MCP resource-route smoke.

Detailed source, Bead, history, provider-contract, and contradiction findings are in `EVIDENCE.md`.

## Implemented mechanism

### 1. One packaged authority for agent behavior

`polylogue/agent_integration/` is the new production package. It contains:

- `spec.py`: typed client, role, capability-family, and executable-recipe declarations.
- `assets.py`: package-resource reads, measured byte counts, aggregate digest, and cache key.
- `manifest.py`: role-scoped manifest generation from actual FastMCP registration.
- `installer.py`: transactional, ownership-aware native integration management.
- `data/standing-manual.md`: the comprehensive standing cold-start instruction.
- `data/deep-reference.md`: exhaustive local follow-up reference.
- `data/recipes.json`: machine-readable executable workflows.
- `data/integration-spec.json`: generated capability/client authority.
- `data/integration-manifest.json`: static install/content manifest.

The content version is `2026-07-17.4`. Measured packaged content is:

- Standing manual: 24,232 UTF-8 bytes.
- Deeper reference: 24,272 UTF-8 bytes.
- Capability families: 15.
- Recipes: 13.
- Aggregate agent-asset digest: `8a9beca255dfcaa6e34bc97f88cbab7f0a5220cc785513cc3aedbf68d6d7148a`.
- Cache key: `polylogue-agent-2026-07-17.4-8a9beca255dfcaa6`.

The implementation reports actual size and digest; it does not impose an arbitrary prompt-size ceiling. Opt-down controls are explicit and described as behavior-reducing choices.

### 2. Standing context, local reference, and live MCP catalog

The MCP server now carries the complete standing manual in its server instructions, together with a hard role-boundary reminder. The server also exposes:

- `polylogue://agent/manual`
- `polylogue://agent/reference`
- `polylogue://agent/manifest/{role}`

The manifest resource is generated from actual FastMCP registration. A final direct route smoke caught a plain-dictionary/Pydantic serialization mismatch; the resource now emits deterministic JSON directly. Under the registration harness, live counts are:

| Role | Tools | Resources | Resource templates | Prompts |
| --- | ---: | ---: | ---: | ---: |
| `read` | 66 | 7 | 7 | 12 |
| `write` | 95 | 7 | 7 | 12 |
| `review` | 97 | 7 | 7 | 12 |
| `admin` | 104 | 7 | 7 | 12 |

The standing manual defaults agents to the read role and names only surfaces valid for that role. It explicitly rejects stale fictional names such as `get_session` and `get_recovery_report`.

### 3. Lazy CLI integration

A new lazily registered `polylogue agent` command group provides:

- `manual`: render the standing manual or deeper reference, plain or JSON.
- `manifest`: enumerate the live role-scoped MCP surface.
- `install`: install selected native clients transactionally.
- `status`: inspect ownership and native state without mutation.
- `doctor`: run blocking integrity, syntax, executable, identity, drift, and precedence checks.
- `uninstall`: remove only exact operations recorded as Polylogue-owned.
- Hidden `session-start`: emit the Claude Code `SessionStart` hook payload containing the complete manual.

The command group follows the repository’s lazy command-registration pattern and is included in generated command/surface catalogs.

### 4. Ownership-safe native installer

The installer writes user-scoped native configuration and records exact ownership operations in:

`$XDG_STATE_HOME/polylogue/agent-integrations.json`, or `~/.local/state/polylogue/agent-integrations.json` when `XDG_STATE_HOME` is unset.

The state is canonicalized and self-digested. A malformed, unsupported, or tampered state file makes status/doctor blocking and prevents uninstall from deleting native content.

Safety properties implemented and tested include:

- Atomic writes and process-level file locking.
- Multi-client transaction rollback on conflict.
- Fail-closed handling for malformed JSON/TOML/YAML and symlinked managed targets.
- No-rewrite idempotence when desired state is already present.
- Exact preservation of unrelated configuration and operator-authored text.
- Values that merely coincide with Polylogue’s desired value are treated as satisfied but unowned; they are never upgraded or removed.
- Owned-content drift is reported and retained rather than overwritten or deleted.
- Exact role, archive-root, config-path, command, guidance, and reference reconciliation.
- Clean-home uninstall removes installer-created empty directories, including shared parents, but preserves operator additions.
- `--replace-clients` reconciles the selected set while still respecting ownership and drift.

### 5. Native client delivery

Claude Code:

- MCP entry: `~/.claude.json`, or `$CLAUDE_CONFIG_DIR/.claude.json` when the alternate root is active.
- Hook settings: `~/.claude/settings.json` or `$CLAUDE_CONFIG_DIR/settings.json`.
- Guidance: one managed `SessionStart` command hook invoking `polylogue agent session-start --client claude-code`.
- Hook output: complete manual in `hookSpecificOutput.additionalContext` with `hookEventName: SessionStart`.
- Existing unrelated hooks are retained. A non-list `hooks.SessionStart` shape is a conflict and is not clobbered.

Codex:

- Root: `CODEX_HOME` or `~/.codex`.
- MCP: one marked `[mcp_servers.polylogue]` block in `config.toml`.
- Guidance: one marked block in the effective global instruction file. A non-empty `AGENTS.override.md` takes precedence; otherwise `AGENTS.md` is used.
- If an override becomes active after installation, reinstall relocates the owned block transactionally. `doctor` reports the shadowed state before relocation.
- Exact surrounding TOML and Markdown text is preserved.

Gemini CLI:

- Root: `$GEMINI_CLI_HOME/.gemini` or `~/.gemini`.
- MCP: `mcpServers.polylogue` in `settings.json`.
- Guidance: one marked block in `GEMINI.md`.
- Reference: `polylogue-reference.md` when enabled.

Hermes:

- Root: `HERMES_HOME` or `~/.hermes`.
- MCP: `mcp_servers.polylogue` in `config.yaml`.
- Guidance: `skills/productivity/polylogue/SKILL.md`.
- Reference: `skills/productivity/polylogue/references/reference.md`.
- `SOUL.md` is intentionally untouched because the Polylogue material is a procedural capability skill, not identity/personality state.

Every MCP entry includes an explicit command, hard role argument, and optional `POLYLOGUE_ARCHIVE_ROOT` / `POLYLOGUE_CONFIG` identity. No secret value is embedded by the installer.

### 6. Independent Home Manager module

`nix/agent-integration-module.nix` adds `programs.polylogueAgent` with typed options for:

- package
- clients
- MCP role
- guidance mode
- reference visibility
- MCP enablement
- archive root
- config path
- replacement of removed clients

The module installs the selected package and invokes the same production `polylogue agent install` path during Home Manager activation. It does not start the daemon, run ingestion, or grant write authority. `flake.nix` exports it as `homeManagerModules.agentIntegration`.

The module source and export are checked statically. Nix parsing, option evaluation, and activation were not run because `nix`, `nix-instantiate`, and `home-manager` are unavailable in this environment.

### 7. Generated documentation and executable verifier

`devtools/render_agent_manual.py` copies packaged authority to the two checked-in document mirrors and supports a drift-check mode. It is registered as a generated surface.

`devtools/verify_agent_integration.py` checks six lanes:

1. Generated assets, mirrors, byte counts, digest, families, and recipes.
2. Query catalog extraction and compilation with correct session/terminal separation and deliberate negative boundaries.
3. Every inline `polylogue`, `polylogued`, and `polylogue-mcp` route in both documents through the actual Click command tree.
4. Static and live MCP role surfaces, all documented MCP identifiers, deliberate stale-name negatives, and recipe minimum roles.
5. Native installer install/status/doctor/upgrade/opt-down/uninstall round trips.
6. Distribution declarations and Home Manager separation from daemon ownership.

Current extraction covers 20 unique executable expressions, including 10 session expressions, 9 terminal-unit expressions, and 2 deliberate negative boundaries; 9 notation-only forms are tracked but not executed. It parses 30 documented CLI/daemon routes and resolves 61 real documented MCP identifiers.

## Key design decisions

1. The default MCP role is `read`. Higher authority is an explicit installation choice, and the manual does not quietly teach write/admin calls under the read default.
2. The full standing manual is delivered where the client supports standing context. A pointer-only bootstrap was rejected because the later Beads operator contract explicitly requires capability breadth, invocation policy, evidence limits, and recovery before a lookup turn.
3. The deep reference and live manifest supplement the standing manual rather than replacing it.
4. Typed declarations and production registration are authority. The JSON assets, document mirrors, MCP prompts/resources, and verifier are consumers.
5. Native configuration is merged at the smallest client-owned seam. Existing operator text or unrelated configuration is never replaced wholesale.
6. Ownership means “created or adopted by a prior Polylogue-owned operation,” not “currently equals Polylogue’s desired value.” This prevents uninstall from deleting coincidentally equal operator state.
7. Drift is evidence. It is retained and reported rather than force-reconciled or deleted.
8. Hermes receives a skill, not an identity mutation. Codex guidance follows actual global precedence, including `AGENTS.override.md`.
9. The Home Manager integration remains separate from daemon lifecycle and delegates to the same tested installer instead of implementing a second merge engine.
10. The patch includes regenerated topology because new modules were added under `polylogue/`.

## Changed files

### Production package and assets

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
- `polylogue/agent_integration/data/integration-manifest.json`

### CLI and MCP surfaces

- `polylogue/cli/click_command_registration.py`
- `polylogue/cli/commands/agent.py`
- `polylogue/mcp/server.py`
- `polylogue/mcp/server_resources.py`

### Documentation and developer tooling

- `docs/agent-manual.md`
- `docs/agent-integration-reference.md`
- `devtools/render_agent_manual.py`
- `devtools/verify_agent_integration.py`
- `devtools/command_catalog.py`
- `devtools/generated_surfaces.py`

### Nix and topology

- `nix/agent-integration-module.nix`
- `flake.nix`
- `docs/plans/topology-target.yaml`
- `docs/topology-status.md`

### Tests

- `tests/unit/agent_integration/test_assets_and_cli.py`
- `tests/unit/agent_integration/test_installer.py`
- `tests/infra/mcp.py`
- `tests/unit/mcp/test_server_surfaces.py`

## Acceptance matrix

Legend: **PASS** means the stated self-contained behavior was executed in this environment. **PARTIAL** means the implementation is present and a meaningful subset was executed, but an external or broader proof remains. **UNVERIFIED** means no truthful execution claim is made.

### `polylogue-3gd.2`

| Criterion | Status | Evidence |
| --- | --- | --- |
| Canonical guidance ships in package artifacts | PASS for wheel/sdist; PARTIAL for Nix | Five versioned assets are present and byte-identical in wheel, sdist, and a wheel rebuilt from the sdist. Nix source/export is present; Nix build unavailable. |
| Supported agents receive full standing context | PASS for generated native files/payload; UNVERIFIED in live client processes | Complete Claude hook payload and persistent Codex/Gemini/Hermes files are tested in isolated homes. No client executable consumed them here. |
| Broad capability map and automatic invocation policy | PASS | Standing manual contains 15 capability families, cold-start policy, evidence limits, recovery, and non-obvious workflow triggers. |
| Embedded commands/names/expressions/roles cannot silently drift | PARTIAL | All extracted inline routes parse, 20 query expressions compile under the registration harness, 61 MCP names resolve, and live role counts are enumerated. Demo-backed execution of every result/field claim was not possible without the locked runtime/archive dependencies. |
| Required stale-name/query/root/continuation/recovery fixtures | PARTIAL | Stale names, query-surface separation, archive identity, role boundaries, response continuation guidance, and degraded-state text are statically/executably checked. Full demo archive failure-state fixtures were not run. |
| Blind realistic agent trials | UNVERIFIED | No live agent was available. |
| Ablation tests and behavioral economics | UNVERIFIED | Size/cache reporting exists; no controlled section-ablation campaign was run. |
| Deep reference/live catalog reachable without being mandatory | PASS | Native reference files and MCP manual/reference/manifest resources are present; ordinary manual delivery does not require a lookup turn. |

### `polylogue-3gd.3`

| Criterion | Status | Evidence |
| --- | --- | --- |
| Wheel/sdist/Nix contain versioned assets and adapters | PASS for wheel/sdist; PARTIAL for Nix | Offline wheel/sdist builds passed and assets matched. Nix source/export checked, not built. |
| Four-client install/status/doctor/uninstall, idempotence, drift, exact removal | PASS | 22 focused tests plus verifier round trip in isolated temporary homes. |
| Typed Home Manager options and evaluations | PARTIAL | Typed module and flake export are present; Nix parsing/evaluation/activation unavailable. |
| Agent integration does not own daemon/ingestion/write authority | PASS by source contract | Module only installs package and calls `polylogue agent install`; default role is read; no daemon activation or ingestion call. |
| Sinnix consumes upstream artifacts and parity check | UNVERIFIED | Sinnix was not present in the supplied authority and was not accessed. |
| Clean-HOME receives manual and reaches realistic demo/recovery | PARTIAL | Clean-HOME install/uninstall and complete manual delivery passed. Live demo/archive/client behavior was unavailable. |
| Upgrade across two fixture versions preserves additions | PARTIAL | Role/archive/content/opt-down reconciliation and operator-addition preservation are tested. Two released Polylogue package fixtures were not available. |
| Report size, digest, coverage, and opt-down impairment | PASS | CLI/status metadata exposes measured bytes, digest/cache key, recipe/family counts; docs describe opt-down impact. |

### Mission-level cold-start route

| Requirement | Status | Evidence |
| --- | --- | --- |
| Identify correct installation path | PASS | Native path resolution and profile-root overrides are tested for all four clients. |
| Seed or import privacy-safe data | PARTIAL | Current production `demo seed`, `demo verify`, and import/help routes are documented and parsed; live seeding was dependency-blocked. |
| Verify daemon/archive | PARTIAL | Production status/readiness/doctor routes are documented and parsed; no live daemon or archive was accessed. |
| Connect MCP | PASS for generated native configuration; UNVERIFIED live handshake | Per-client entries and roles are tested; no provider process or network MCP handshake ran. |
| Run representative queries | PASS for grammar/route compilation; UNVERIFIED against live archive | 20 unique expressions are checked; no production archive query execution here. |
| Diagnose failure without tribal knowledge | PASS for manual/doctor contract; PARTIAL for live states | Manual contains explicit identity, readiness, stale index, incomplete coverage, response-budget, and recovery guidance; live failure modes were not exercised. |

## Apply order

From a clean checkout of the exact base commit:

```bash
git checkout f654480cadb7cc4c194704e24dfd483199547b35
git apply --check PATCH.diff
git apply PATCH.diff
```

Install the project’s locked development dependency closure using the repository-supported environment, then run:

```bash
python -m pytest -o addopts='' --confcutdir=tests/unit/agent_integration -q tests/unit/agent_integration
python devtools/render_agent_manual.py --check
python -m devtools.render_topology_status --check
python -m devtools.verify_topology --json
python devtools/verify_agent_integration.py --json
```

With real Nix tooling, add:

```bash
nix flake check
```

Then evaluate at least one Home Manager configuration for each of: one read client, multiple clients, write role, guidance off, reference off, custom archive/config paths, and removal/reconciliation of a formerly selected client.

For a local client smoke, use a disposable home/profile and run the generated `polylogue agent install`, `status`, `doctor`, and `uninstall` commands before testing a real provider process. Do not point a first smoke at an operator’s primary native configuration.

## Verification performed

All commands below were executed against the final implementation; the focused and verifier commands were repeated after applying `PATCH.diff` to a fresh clone.

1. Focused tests:

```text
22 passed, 2 warnings in 0.48s
fresh-applied tree: 22 passed, 2 warnings in 0.60s
```

The warnings are only unknown `pytest-timeout` configuration keys because that plugin is absent in the shell.

2. Python syntax/import compilation:

```text
python -m compileall ...: PASS
```

3. Generated documents and topology:

```text
render_agent_manual --check: PASS
render_topology_status --check: PASS
verify_topology: blocking=false; orphans=0; missing=0; conflicts=0; kernel_rule=0
```

The topology verifier still reports nine advisory `tbd` storage paths that predate this implementation; none is a new blocking topology finding.

4. Native dependency environment verifier:

```text
fail=0, pass=3, unverified=3, ok=true
```

Generated assets, installer round trip, and packaging/Home Manager static contracts passed. Query, some lazy CLI routes, and live MCP registration were marked unverified because `dateparser`, `ijson`, `aiosqlite`, and the MCP SDK are absent.

5. Isolated registration/import compatibility harness:

```text
fail=0, pass=6, unverified=0, complete=true
```

The harness supplied minimal import/registration-only stand-ins for missing dependencies. It did not open a database, stream data, or serve MCP traffic. Under that bounded purpose it compiled 20 unique query expressions, parsed 30 routes, resolved 61 documented MCP names, checked 13 recipes, and enumerated the four role surfaces shown above.

6. Direct MCP resource-route smoke under the same registration harness:

```text
manual_bytes=24232
reference_bytes=24272
read_tools=66
invalid_role_code=invalid_request
```

7. Distribution build and resource reproduction:

```text
wheel: polylogue-0.2.0-py3-none-any.whl
bytes: 4,096,599
sha256: ff878bc16ea7c521867092267f7838e031fb045a990758b0d34f8d0f08e21773

sdist: polylogue-0.2.0.tar.gz
bytes: 32,936,161
sha256: 21234e105c752b61d4265c7e7a0191254f658eddb8534892d3742b350192c4bf
```

All five assets were present in both. Rebuilding a wheel from the sdist produced the same wheel SHA-256 and the same aggregate agent-asset digest.

8. Patch proof:

```text
PATCH.diff bytes: 260,939
PATCH.diff sha256: 549a1e3e41d901de08066ce22fda69020cbd06c0f8a76055b2ab38edbff7ff88
git apply --check against exact commit: PASS
git apply against fresh clone: PASS
changed files compared: 29
byte-and-mode identity after apply: PASS
git diff --check: PASS
```

## Risks and unverified work

- The shell lacks `dateparser`, `ijson`, `aiosqlite`, `mcp`, `hypothesis`, and `sqlite_vec`. The complete locked test suite, real FastMCP process, production demo archive, and SQLite-vector-backed archive path were not executed.
- The shell lacks `nix`, `nix-instantiate`, and `home-manager`. The Nix module is source-reviewed and statically checked but not parsed, evaluated, activated, or built.
- No live Claude Code, Codex, Gemini CLI, or Hermes process consumed the generated configuration. Provider-native acceptance, hook invocation, and MCP handshake remain local follow-up.
- No live daemon, private archive, browser, deployment, cloud service, secrets, or operator worktree was accessed.
- Sinnix is not in the supplied snapshot, so downstream consumption/parity cannot be certified here.
- No blind-agent trial or ablation experiment was run. The implementation creates the stable authority and instrumentation inputs for those studies but does not claim behavioral proof.
- A true upgrade matrix across two released package versions was unavailable. The tests cover semantic reconciliation of role/archive/content/client-set changes, not packaging behavior across historical releases.
- Home Manager disabling the module entirely cannot invoke an uninstall activation from the disabled module. Operators should first reconcile to an empty or changed client set while enabled, or run `polylogue agent uninstall`, before removing the module declaration.
- Native client file formats and precedence rules are external contracts and can evolve. The executable verifier catches project-side drift, but provider-contract changes still require periodic review.
- The installer uses POSIX `fcntl` locking and was validated in this Linux environment; Windows-native operation is not claimed.
- `ruff` and `mypy` were unavailable, so no final lint/type-check claim is made. Syntax compilation, focused tests, verifier checks, distribution builds, and patch checks all passed.

## Value of another iteration

A small repair pass has bounded value: install the repository’s locked Python dependency closure and Nix tooling, run the full relevant MCP/CLI suites plus `nix flake check`, and repair any environment-realized issue. The implementation already has zero known self-contained failures, so this pass is primarily certification.

A substantial second pass has meaningful product value: run live provider handshakes for all four clients; exercise demo seed/verify and representative queries against a real disposable archive; test no-daemon, wrong-archive, stale-index, incomplete-source, and response-budget recovery; add two released-version fixtures; verify Sinnix consumes only upstream artifacts; and conduct blind-agent plus section-ablation trials. Those activities would convert the remaining behavioral and ecosystem criteria from unverified to empirical evidence rather than materially changing the current architecture.
