# Test design and execution record

## Testing principle

The test strategy exercises production routes rather than a parallel documentation framework. Asset tests load package resources through `importlib.resources`; CLI tests invoke the actual Click command group; installer tests mutate real JSON, TOML, YAML, Markdown, and skill files in isolated homes; MCP tests inspect the production FastMCP registration; query checks call production parsers; command checks resolve through production Click registration; packaging checks open actual wheel/sdist members; and the patch proof applies the unified diff to the exact Git commit.

Each contract below names a representative implementation mutation that should make it fail. These mutation descriptions are anti-vacuity criteria, not claims that an automated mutation-testing tool was run.

## Focused executable tests

Command:

```bash
PYTHONPATH=. /opt/pyvenv/bin/python -m pytest \
  -o addopts='' \
  --confcutdir=tests/unit/agent_integration \
  -q tests/unit/agent_integration
```

Final implementation result:

```text
22 passed, 2 warnings in 0.48s
```

Fresh-clone, post-`git apply` result:

```text
22 passed, 2 warnings in 0.60s
```

The two warnings are `PytestConfigWarning` entries for the repository’s `timeout` and `timeout_method` settings because `pytest-timeout` is not installed in the execution shell. They are not test failures.

### Asset and CLI tests

| Test | Production dependency exercised | Representative mutation/removal that should fail it |
| --- | --- | --- |
| `test_all_packaged_assets_are_nonempty_and_measured` | `polylogue.agent_integration.assets.agent_asset_metadata`, package resources, typed family/recipe declarations | Remove a resource, return a fixed byte count, truncate either document below 20,000 bytes, reduce family/recipe declarations without regenerating metadata, or stop computing a 64-character digest. |
| `test_json_assets_match_typed_authority` | Generated `recipes.json` and `integration-spec.json` versus `spec.py` declarations | Rename/reorder a recipe or capability family in only one authority, or hand-edit generated JSON without updating the typed declarations. |
| `test_manual_contains_cold_start_authority_and_negative_names` | Packaged standing manual | Remove the cold-start section, archive identity, demo route, read-role default, or stale-name warning for `get_session`. |
| `test_claude_session_start_returns_complete_manual` | `claude_session_start_payload` and packaged manual | Replace the full manual with a pointer, use the wrong hook event, truncate context, or load a different asset. |
| `test_manual_and_hidden_session_start_commands` | Real `polylogue agent manual` and hidden `session-start` Click commands | Rename an option, emit invalid JSON, add an extra newline to the plain manual, or stop routing the hook payload through production code. |
| `test_generated_document_mirrors_match_packaged_assets` | Checked-in documentation mirrors versus package resources | Edit only `docs/agent-manual.md`, only the packaged asset, or stop running the renderer. |

### Installer lifecycle tests

| Test | Production dependency exercised | Representative mutation/removal that should fail it |
| --- | --- | --- |
| `test_all_clients_install_native_mcp_and_guidance` | Four native adapters, JSON/TOML/YAML/Markdown/skill output, status and doctor | Use a generic non-native path, edit Hermes `SOUL.md`, omit archive/role identity, omit Claude hook, or fail to install one client’s reference. |
| `test_install_is_idempotent_without_rewriting_files` | Desired-state comparison and atomic writer | Always rewrite files on install, update timestamps despite equal content, or generate unstable ordering. |
| `test_role_and_archive_upgrade_reconciles_owned_values` | Ownership state, previous desired value, MCP entry reconciliation | Refuse all upgrades, silently retain the old role/archive, or overwrite a value after operator drift. |
| `test_operator_additions_survive_uninstall_and_clean_owned_dirs` | Exact operation removal and directory cleanup | Replace entire native files, delete unrelated keys/text, or recursively remove a profile directory. |
| `test_clean_home_uninstall_removes_only_installer_created_tree` | Shared parent tracking and retrying empty-directory removal after all clients | Remove parent directories too early and leave them behind, or remove a non-empty operator directory. |
| `test_preexisting_equal_native_value_is_not_claimed_or_removed` | Satisfied-but-unowned semantics | Treat equality as ownership, upgrade the operator’s coincidentally equal value, or delete it on uninstall. |
| `test_operator_conflict_fails_closed_and_rolls_back_other_clients` | Multi-client transaction snapshots and rollback | Continue after a conflict, leave earlier client writes behind, or overwrite the conflicting operator value. |
| `test_drifted_owned_content_is_retained_on_uninstall` | Drift detection from recorded desired content | Delete drifted content, restore old content silently, or report uninstall success without retained drift. |
| `test_corrupt_state_blocks_status_doctor_and_uninstall` | State schema and self-digest verification | Ignore a modified state digest, infer ownership from current files, or allow uninstall from untrusted state. |
| `test_symlinked_native_config_is_refused` | `_refuse_symlink` at read/write boundaries | Follow a symlink and mutate its target, or replace the symlink with a regular file. |
| `test_opt_down_removes_exact_guidance_but_keeps_mcp` | Desired-operation set reconciliation | Leave owned guidance after `mcp-only`, remove MCP accidentally, or remove operator text around a marked block. |
| `test_codex_guidance_relocates_when_override_becomes_active` | Codex global instruction precedence and transactional marked-block relocation | Keep guidance shadowed in `AGENTS.md`, duplicate it into both files, or remove operator text during relocation. |
| `test_provider_specific_profile_roots_are_respected` | `CLAUDE_CONFIG_DIR`, `CODEX_HOME`, `GEMINI_CLI_HOME`, `HERMES_HOME` | Ignore an override, treat `GEMINI_CLI_HOME` as `.gemini` itself instead of its containing profile, or mix files across homes. |
| `test_replace_clients_removes_unselected_exact_operations` | Selected-client reconciliation and exact uninstall | Leave a removed owned client configured, remove a client not owned by state, or delete shared operator additions. |
| `test_state_records_current_asset_digest` | `agent_asset_digest`, signed ownership state | Store a stale/fixed digest, omit content version, or write unsigned state. |
| `test_malformed_claude_session_start_shape_is_not_overwritten` | Native schema validation for `hooks.SessionStart` | Coerce a mapping/string to a list, clobber malformed operator state, or install without reporting a conflict. |

## Executable integration verifier

Native environment command:

```bash
PYTHONPATH=. /opt/pyvenv/bin/python devtools/verify_agent_integration.py --json
```

Result:

```json
{
  "complete": false,
  "counts": {"fail": 0, "pass": 3, "unverified": 3},
  "ok": true
}
```

The passing lanes were generated assets, native installer round trip, and packaging/Home Manager static contracts. The dependency-gated lanes were query catalog (`dateparser` absent), six lazy CLI/daemon routes (`ijson`, `dateparser`, or `aiosqlite` absent), and live MCP registration (`dateparser` encountered before registration could complete).

Registration/import compatibility harness command:

```bash
PYTHONPATH=/mnt/data/beads06-work/runtime-stubs:. \
  /opt/pyvenv/bin/python devtools/verify_agent_integration.py --json
```

Result:

```json
{
  "complete": true,
  "counts": {"fail": 0, "pass": 6, "unverified": 0},
  "ok": true
}
```

The harness supplies minimal import-compatible stand-ins only for dependency-gated registration and command construction. Its `aiosqlite` and `ijson` implementations throw on database/stream use, and its FastMCP implementation cannot serve traffic. Therefore, it proves declaration compatibility, parser/command construction, and registration shape, not live database or protocol behavior.

### Generated-assets lane

Production dependencies:

- `importlib.resources` reads from `polylogue.agent_integration.data`.
- `agent_asset_metadata()` measures content and computes the aggregate digest.
- Typed `CAPABILITY_FAMILIES` and `RECIPES` declarations are compared to generated JSON.
- Checked-in document mirrors are compared byte-for-byte.

Observed result:

```text
5 assets
standing manual: 24,232 bytes
deep reference: 24,272 bytes
15 capability families
13 recipes
asset digest: 8a9beca255dfcaa6e34bc97f88cbab7f0a5220cc785513cc3aedbf68d6d7148a
```

Anti-vacuity mutations: remove an asset; alter a mirror only; return a hard-coded digest; remove a family from JSON but not Python; change content without advancing the digest/cache key.

### Query-catalog lane

Production dependencies:

- Current session-expression parser.
- Current terminal-unit expression parser.
- Recipe declarations and both Markdown documents.
- Explicit classification between session and terminal query surfaces.

Observed result under the harness:

```text
20 unique executable expressions
10 session expressions
9 terminal-unit expressions
2 deliberate negative boundaries
9 notation-only forms tracked, not executed
```

The negative boundaries ensure invalid sessions-only/terminal-only routing remains rejected rather than “validated” by a permissive parser.

Anti-vacuity mutations: send all expressions through one parser; stop extracting the documents; accept an invalid cross-surface expression; replace a checked query with inert text; remove a recipe’s minimum role or command.

### CLI-command-contract lane

Production dependencies:

- Root lazy Click registration.
- Daemon Click group.
- Actual option/argument parsers for each extracted route.
- Both standing and deeper-reference documents.

Observed result under the harness:

```text
30/30 inline routes parsed
```

Representative routes include `polylogue paths`, `status`, `find ... then continue`, terminal-unit expressions, assertion/file/action queries, agent status/manual/manifest/install/doctor/uninstall, `init --dry-run`, `polylogued run`, and demo seed/verify.

Anti-vacuity mutations: change `polylogue agent manual --kind reference` back to the nonexistent `--kind recipes`; remove lazy registration; rename an option; accept only command names but not options; stop scanning the deeper reference; mutate `find ... then continue` to the stale continuation syntax.

### MCP-role-surface lane

Production dependencies:

- `polylogue.mcp.server.build_server` for all four roles.
- Production tool/resource/prompt decorators.
- Existing expected MCP test declarations.
- Documented identifier extraction and recipe minimum roles.

Observed result under the registration harness:

```text
read:   66 tools, 7 resources, 7 templates, 12 prompts
write:  95 tools, 7 resources, 7 templates, 12 prompts
review: 97 tools, 7 resources, 7 templates, 12 prompts
admin: 104 tools, 7 resources, 7 templates, 12 prompts
61 documented real MCP identifiers resolved
2 deliberate stale-name negatives rejected
13 recipes checked at declared minimum roles
```

Anti-vacuity mutations: remove one decorator; rename a tool without updating the manual; expose `add_tag` in read; reduce role checks to static text; stop checking prompts/resources; treat `get_session` as valid; return a static manifest unrelated to actual managers.

A direct resource-route smoke also called the registered manual/reference/manifest functions. It exposed and then verified the fix for a real defect: passing the manifest dictionary to a Pydantic-only serializer raised `AttributeError`. The final resource returns deterministic JSON directly.

### Native-installer-roundtrip lane

Production dependencies:

- Real `AgentIntegrationManager` install/status/doctor/uninstall methods.
- Native profile path resolution.
- Atomic JSON/TOML/YAML/Markdown writes.
- State signing, drift checks, exact removal, and created-directory cleanup.

Observed result:

```text
all four clients installed
status/doctor successful
second install made no rewrites
role/archive upgrade successful
opt-down cleanup successful
exact clean uninstall successful
```

Anti-vacuity mutations overlap the focused lifecycle tests: claim all current values, ignore state digest, overwrite operator blocks, remove drift, omit rollback, or leave created directories behind.

### Packaging-and-Home-Manager lane

Production dependencies:

- Five package-resource names.
- `nix/agent-integration-module.nix` option and production-CLI invocation contract.
- `flake.nix` export.
- Absence of daemon lifecycle commands from the agent integration module.

Anti-vacuity mutations: omit package assets; remove a required option; call `polylogued` from the module; hard-code a second config merger in Nix; remove the flake export.

## MCP full-suite test changes

The patch updates existing MCP expectations and adds production-route tests in `tests/unit/mcp/test_server_surfaces.py` for:

- Byte-exact manual and reference resources.
- Live read-role manifest content and counts.

It also updates `tests/infra/mcp.py` so the expected resource URI and template sets include the three new agent surfaces.

These tests were not executed in the full repository suite because the shell lacks `hypothesis` and multiple locked runtime dependencies imported by the repository-level test configuration. The direct registration-harness resource smoke exercised the same production functions and caught the serializer defect, but it is not presented as a substitute for the full locked-environment suite.

Representative anti-vacuity mutations: remove the resource decorators; return docs from the filesystem rather than package resources; hard-code a manifest with the wrong tool count; include `add_tag` under read; route the dictionary through `_json_payload` again.

## Generated-document and topology checks

Commands:

```bash
/opt/pyvenv/bin/python devtools/render_agent_manual.py --check
/opt/pyvenv/bin/python -m devtools.render_topology_status --check
/opt/pyvenv/bin/python -m devtools.verify_topology --json
git diff --check
/opt/pyvenv/bin/python -m compileall -q \
  polylogue/agent_integration \
  polylogue/cli/commands/agent.py \
  devtools/render_agent_manual.py \
  devtools/verify_agent_integration.py \
  tests/unit/agent_integration
```

Results:

```text
manual render drift check: PASS
topology status drift check: PASS
topology blocking: false
orphans: 0
missing: 0
conflicts: 0
kernel-rule findings: 0
git diff --check: PASS
compileall: PASS
```

The topology output contains nine advisory `tbd` paths in existing storage areas. No new module is orphaned or missing from topology.

Anti-vacuity mutations: edit only a document mirror; add a `polylogue/` module without rerendering topology; remove the generated-surface registration; introduce trailing whitespace or an unresolved conflict marker; make one new Python module syntactically invalid.

## Distribution checks

Offline build command used cached Hatchling build dependencies and built from the implementation source:

```bash
python -m hatchling build
```

Results:

```text
polylogue-0.2.0-py3-none-any.whl
4,096,599 bytes
ff878bc16ea7c521867092267f7838e031fb045a990758b0d34f8d0f08e21773

polylogue-0.2.0.tar.gz
32,936,161 bytes
21234e105c752b61d4265c7e7a0191254f658eddb8534892d3742b350192c4bf
```

Both archives were opened directly. Each contains:

- `standing-manual.md`
- `deep-reference.md`
- `recipes.json`
- `integration-spec.json`
- `integration-manifest.json`

Per-asset SHA-256 values matched between wheel and sdist. A wheel rebuilt from the extracted sdist had the same wheel SHA-256 as the source-built wheel. Importing `agent_asset_metadata()` directly from that derived wheel reproduced the expected content version, byte counts, cache key, and aggregate digest.

Anti-vacuity mutations: exclude package data in the build configuration; include only Markdown but omit JSON; generate resources after the sdist is made; depend on source-tree relative paths; make the sdist-built wheel differ from the direct wheel.

## Patch application proof

Commands:

```bash
git clone --no-hardlinks . ../beads06-apply-check
cd ../beads06-apply-check
git checkout f654480cadb7cc4c194704e24dfd483199547b35
git apply --check PATCH.diff
git apply PATCH.diff
git diff --check
```

Results:

```text
git apply --check: PASS
git apply: PASS
base HEAD unchanged at named commit: PASS
29 changed paths compared against implementation tree: PASS
byte identity: PASS
mode identity: PASS
focused tests in fresh-applied tree: 22 passed
manual/topology drift checks in fresh-applied tree: PASS
native verifier in fresh-applied tree: zero failures
registration harness in fresh-applied tree: six lanes passed
```

`PATCH.diff` metadata:

```text
260,939 bytes
549a1e3e41d901de08066ce22fda69020cbd06c0f8a76055b2ab38edbff7ff88
```

Anti-vacuity mutations: generate the diff against the incomplete working-tree export instead of the exact Git commit; omit new files; fail to include binary topology deltas; depend on untracked shell files; change file modes after generating the patch.

## Tests not run and why

- Full repository test suite: locked dependencies, including `hypothesis`, are absent.
- Full MCP SDK server tests: `mcp` and other transitive runtime dependencies are absent.
- Production demo archive seed/verify/query: `aiosqlite` and `sqlite_vec` are absent.
- Nix/Home Manager parsing/evaluation/activation: `nix`, `nix-instantiate`, and `home-manager` are absent.
- Ruff and mypy: executables/modules are absent.
- Live provider acceptance: Claude Code, Codex, Gemini CLI, and Hermes processes were not available or invoked.
- Browser, cloud, deployment, private archive, daemon, and secrets: deliberately not accessed.
- Blind-agent trials and manual-section ablations: require controlled live-agent experiments and are not claimed.
