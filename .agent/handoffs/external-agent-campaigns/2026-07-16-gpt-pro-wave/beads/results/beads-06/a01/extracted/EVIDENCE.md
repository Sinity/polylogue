# Source, Bead, history, and contract evidence

## Authority model

The attached snapshot is authoritative for Polylogue source. The mission prompt defines the requested outcome and delivery contract. Repository instructions and complete relevant Beads records constrain implementation. Current source takes precedence when an older plan or note names a path or API that no longer exists.

No operator worktree, private archive, live daemon, browser, secrets, deployment, or cloud environment was used. External provider documentation was consulted only to establish current native client configuration and standing-guidance mechanisms.

## Snapshot identity evidence

The snapshot manifest at `polylogue-manifest.json` states:

```json
{
  "generated_at": "2026-07-17T043202Z",
  "git": {
    "branch": "master",
    "commit": "f654480cadb7cc4c194704e24dfd483199547b35",
    "dirty": true
  },
  "project": "polylogue",
  "source": "/realm/project/polylogue"
}
```

The exact commit exists in the bundled all-refs Git object store. Git reports:

```text
commit=f654480cadb7cc4c194704e24dfd483199547b35
parent=0976039d2ec5ab54e45ef528316ca41cd3b4ffbf
author=Sinity <ezo.dev@gmail.com>
author_date=2026-07-17T03:45:52+02:00
subject=chore(beads): file archive-insight benchmark findings wofr/vhjs/r7p6/1wtm
branch containing commit=master
```

The same snapshot manifest records all three current-branch delta artifacts as empty:

```text
polylogue-branch-delta-files.txt: 0 bytes
polylogue-branch-delta-log.txt:   0 bytes
polylogue-branch-delta.patch:     0 bytes
SHA-256 for each empty artifact: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

Conclusion: the dirty flag does not correspond to a tracked branch delta that can be carried into this handoff. The named commit is the only reproducible patch base.

The working-tree tar export lacked some files that are present in the exact Git commit, including tracked repository-local instruction/state and lock surfaces. This appears to be an export-packaging decision, not an intended source deletion. The implementation was therefore overlaid on a clean checkout from the bundled Git objects, and no omission-only deletion appears in `PATCH.diff`.

## Repository instruction evidence

Relevant `CLAUDE.md` constraints found and followed:

- The architecture has four rings; semantic meaning belongs in substrate/product layers, while CLI/MCP are leaf adapters.
- New semantics must not be invented independently in a surface.
- The CLI uses lazy registration.
- MCP is a large, explicitly tested agent-facing surface.
- Generated references and surfaces are repository-owned and must be checked for drift.
- Adding a module or file under `polylogue/` requires regenerating `docs/plans/topology-target.yaml` and `docs/topology-status.md`.
- New MCP surfaces require updating expected names in `tests/infra/mcp.py`.

The implementation does not create new archive semantics. It packages and validates current contracts, delegates live manifest generation to actual MCP registration, routes native lifecycle through one production installer, registers CLI lazily, updates expected MCP resources, and regenerates topology.

## Beads evidence

### `polylogue-3gd.2` — Inject a comprehensive executable Polylogue manual into agent context

Status in snapshot: open, priority 1.

Core problem:

- Consumer-owned skill/SessionStart text is stale and incomplete.
- It teaches invalid queries and nonexistent tools.
- Agents need the capability map, invocation triggers, semantics, evidence limits, and recovery before they decide whether Polylogue applies.

Current design authority:

- Create canonical `AgentIntegrationSpec` and recipe declarations in Polylogue.
- Generate a comprehensive standing manual.
- Cover archive identity, coverage/freshness/uncertainty, query/read/get/explain algebra, result/ref/continuation semantics, capability families, lineage, orchestration, tasks/attempts/effects, readiness/degraded states, mutations/roles/authority, recovery, and live catalog inspection.
- Deliver the full manual through `SessionStart` where supported and the closest persistent mechanism elsewhere.
- Keep deeper reference/catalog supplemental rather than making ordinary success depend on voluntary lookup.
- Compile checked-in fixtures until broader generated query metadata is available.

Later operator note, dated 2026-07-15, explicitly supersedes same-day pointer-only/tiny-bootstrap formulations: `SessionStart` must deliver the comprehensive project-owned standing manual, and prompt caching plus avoided failure turns are first-class economics.

Implementation mapping:

- Typed `CAPABILITY_FAMILIES`, `RECIPES`, roles, clients, and content version in `spec.py`.
- Full standing manual and deep reference as package resources.
- Complete Claude hook payload; persistent Codex/Gemini/Hermes guidance.
- Live MCP manual/reference/manifest resources.
- Query, command, MCP-name, role, recipe, generated-asset, and installer verification.
- Size/digest/cache reporting without arbitrary cap.

Remaining Bead evidence not closed here:

- Blind unhinted live-agent trials.
- Section ablation experiments.
- Demo-backed execution of every field/value/result claim.

### `polylogue-3gd.3` — Install the agent integration kit through packages, clients, and Nix

Status in snapshot: open, priority 1.

Core problem:

- Polylogue releases CLI/daemon/MCP/Nix surfaces but not the comprehensive client integration needed for routine correct use.
- Sinnix separately carried MCP profiles, a forked skill, and stale SessionStart guidance.
- Running the MCP server alone does not make agents aware of the right capabilities or recovery paths.

Current design authority:

- Package the 3gd.2 spec, comprehensive manual, recipes, deep reference, manifest, and client adapters.
- Provide non-destructive `polylogue agent install/status/doctor/uninstall` for Claude Code, Codex, Gemini, and Hermes.
- Discover native locations and use the full standing manual or closest persistent mechanism.
- Install a role-scoped MCP entry and verify advertised routes.
- Record ownership so upgrades/uninstall touch only managed entries and never overwrite unrelated configuration or operator-authored instruction files.
- Export an upstream per-user Home Manager module separate from daemon lifecycle.
- Offer typed package/client/role/guidance/reference/archive/config options.
- Default to correct routine use while allowing explicit opt-down with disclosed impairment.
- Keep secrets out of generated world-readable files.

Later operator note, dated 2026-07-15, supersedes small discovery-hook and downstream hard-coded tool-list formulations. Packages and the upstream Home Manager module should install the comprehensive project-owned standing manual by default.

Implementation mapping:

- Four tested native adapters.
- Signed ownership state and exact operation records.
- Atomic transactions, symlink refusal, conflict rollback, no-rewrite idempotence, upgrade reconciliation, drift retention, exact uninstall, clean directory removal, and profile-root overrides.
- Typed Home Manager module exported by the flake and delegating to the production CLI.
- Wheel/sdist resource verification and measured asset identity.

Remaining Bead evidence not closed here:

- Nix parse/evaluation/activation/build.
- Sinnix downstream consumption/parity.
- Live native client handshakes.
- Two actual released-package upgrade fixtures.
- Blind agent behavior on a real demo archive and degraded states.

## Source evidence followed through production routes

### Packaging

Inspected:

- `pyproject.toml`
- Hatchling package discovery and source layout
- `flake.nix`
- existing `nix/` package/module files
- distribution build commands

Finding: package resources under a Python package are the correct wheel/sdist authority. The final build confirms all five resources ship. No separate copied snapshot is needed in the handoff.

### CLI registration and command parsing

Inspected:

- `polylogue/cli/click_command_registration.py`
- root command construction and lazy loaders
- `polylogue/cli/commands/*`
- daemon command group
- `devtools/command_catalog.py`
- generated surface registration

Finding: the agent group must be registered lazily rather than imported eagerly into root CLI startup. The verifier resolves complete command lines, including options and arguments, rather than only checking that command-name strings exist.

### Query contracts

Inspected:

- session query parser/spec declarations
- terminal-unit expression parser and command path
- current `find ... then ...` routing
- current result/ref/continuation semantics used by CLI and MCP
- query-related docs and tests

Finding: session expressions and terminal-unit expressions are distinct surfaces. Earlier draft recipes accidentally routed some terminal-unit expressions through session parsing and contained stale continuation syntax. The final recipe/manual corpus classifies them separately and validates deliberate negative boundaries.

### MCP

Inspected:

- `polylogue/mcp/server.py`
- `polylogue/mcp/server_resources.py`
- tool/prompt/resource registration modules
- role gating in server construction/support
- response serializer and response-budget behavior
- `tests/infra/mcp.py`
- `tests/unit/mcp/test_server_surfaces.py`

Finding: the live role manifest should enumerate actual managers produced by `build_server(role=...)`; copying a tool-name list into another file would create exactly the drift the Beads reject.

A final direct registered-resource invocation found a real defect after the broader verifier passed registration: `build_live_manifest()` returns a plain dictionary, while `hooks.json_payload()` requires a Pydantic `BaseModel` and calls `model_dump_json`. The resource raised `AttributeError`. The final patch serializes this plain dictionary directly using deterministic JSON. This finding demonstrates why route invocation was retained in addition to static name enumeration.

### Daemon, paths, readiness, demo, and import

Inspected:

- config and path identity contracts
- daemon entry point and status/readiness commands
- public demo seed/verify commands
- import command/help surface
- archive-root/config environment handling

Finding: client installation must make archive/config identity explicit but must not start the daemon, initialize an archive, ingest data, or grant write authority. The standing manual teaches the production routes and recovery model, while the installer remains a configuration operation.

The demo and live archive routes could not be executed because `aiosqlite` and `sqlite_vec` are absent. Their complete CLI forms were still parsed through the production command tree under the registration/import harness.

### Nix and Home Manager

Inspected:

- flake outputs
- existing NixOS and Home Manager daemon modules
- package executable names
- repository’s daemon ownership boundaries

Finding: agent integration belongs in an independent Home Manager module. It should install/reconcile per-user native config using the production CLI and should not become another daemon owner. The module follows this design. Nix evaluation remains unverified because Nix tooling is absent.

## Relevant source history

History was inspected to avoid reverting recent contract changes or adding a parallel surface.

`polylogue/mcp/server.py` recent history:

```text
113d1af97 feat(mcp): bound agent response payloads (#2790)
eff7c2abe fix(mcp): make call telemetry delivery durable (#2760)
54af5477c feat(mcp): persist durable MCP call telemetry (#2750) (#2750)
ac9cfeb0b refactor: current split-file archive as the sole storage architecture (#1787)
7167e305f fix(suppressions): tighten native enforcement and discipline ledger (#1062) (#1093)
```

`polylogue/mcp/server_resources.py` recent history:

```text
593ef3c62 feat(storage): conserve raw authority replay plans (#2961)
113d1af97 feat(mcp): bound agent response payloads (#2790)
ce2f62ac6 feat(read): add projection render specs (#2503)
f666000f1 fix(mcp): expose shared error envelope core (#1825) (#2044)
fe67e9c66 refactor(mcp): converge MCP error payload to canonical {status,message,code} (#1818) (#1898)
```

`polylogue/cli/click_command_registration.py` recent history:

```text
c2fd1e902 fix(security): close excision bypass and scanner coverage gaps (#2875)
5aa34e6c5 feat(assertions): add reviewed candidate judgment flow (#2791)
009658fe5 feat(cli): capture terminal notes as candidates (#2801)
f4504cb4d feat(annotations): import provenance-stamped JSONL batches (#2767) (#2767)
7e6123cac feat: install and monitor harness hooks (#2644)
```

`flake.nix` recent history:

```text
feb666d3c fix(protocol): split mutable summary head from immutable transcript segments (#2838)
355aa309e fix(nix): embed full git revision in packaged build identity (#2743)
1248887ef docs: position Polylogue as a local evidence system (#2698)
7c3f9144f docs: flight-recorder public surface, evidence-first tour, claims ledger (#2655)
c448f76e5 docs: standalone CLAUDE.md + architecture/internals accuracy (#2546)
```

Implications:

- Preserve MCP response-budget and error-envelope conventions.
- Add resources through the existing registration module.
- Keep CLI loading lazy.
- Export a module without disturbing package revision identity or daemon modules.

## External native-client contract evidence

These sources were current when checked on 2026-07-17. They establish only client integration shape; Polylogue source remains authoritative for Polylogue commands and semantics.

### Claude Code

Official references:

- https://docs.anthropic.com/en/docs/claude-code/hooks
- https://docs.anthropic.com/en/docs/claude-code/hooks-guide

Relevant findings:

- Hooks are configured in settings JSON.
- `SessionStart` is a supported lifecycle event.
- Context injection uses `hookSpecificOutput.additionalContext` with the matching `hookEventName`.
- Hook configurations use event arrays containing matcher/hook entries.

Design consequence: install one managed command hook into the existing `hooks.SessionStart` list and emit the full manual as structured `SessionStart` output. Preserve all unrelated hooks. Do not add a second opaque wrapper framework.

### Codex

Official references:

- https://developers.openai.com/codex/agent-configuration/agents-md
- https://developers.openai.com/codex/mcp
- https://developers.openai.com/codex/config-basic
- https://developers.openai.com/codex/config-reference

Relevant findings:

- User-level configuration is `~/.codex/config.toml` by default.
- MCP servers are configured in `config.toml`.
- Persistent global instructions live in `~/.codex/AGENTS.md`.
- `AGENTS.override.md` takes precedence at a directory level and is the documented temporary global override mechanism.
- `CODEX_HOME` is the profile root used for user-level Codex files.

Design consequence: maintain one marked MCP table and one marked block in the effective global instruction file. Detect when a newly non-empty override shadows an existing owned block, report it in doctor, and relocate transactionally on reinstall.

### Gemini CLI

Official references:

- https://geminicli.com/docs/reference/configuration/
- https://geminicli.com/docs/tools/mcp-server/
- https://geminicli.com/docs/cli/settings/
- https://geminicli.com/docs/cli/enterprise/

Relevant findings:

- User settings are in `~/.gemini/settings.json`.
- MCP servers are declared under the top-level `mcpServers` object.
- `settings.json` supports command, args, and environment for stdio servers.
- The enterprise isolation pattern treats `GEMINI_CLI_HOME` as a containing profile home; Gemini state remains under `.gemini` within it.

Design consequence: merge only `mcpServers.polylogue` into `$GEMINI_CLI_HOME/.gemini/settings.json` (or `~/.gemini/settings.json`) and put standing instructions in the client’s persistent `GEMINI.md` mechanism. Do not guess a non-existent second MCP schema.

### Hermes

Official references:

- https://hermes-agent.nousresearch.com/docs/user-guide/features/skills
- https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp
- https://hermes-agent.nousresearch.com/docs/guides/use-mcp-with-hermes
- https://hermes-agent.nousresearch.com/docs/user-guide/features/plugins

Relevant findings:

- Skills are on-demand knowledge documents following progressive disclosure.
- The primary skill directory is `~/.hermes/skills/` and is recursively scanned.
- MCP servers are config-driven under `mcp_servers.polylogue` in `~/.hermes/config.yaml`.
- A skill’s `SKILL.md` is the procedural instruction content the agent loads when relevant.

Design consequence: install Polylogue as `skills/productivity/polylogue/SKILL.md`, keep its deeper reference below the skill, and leave `SOUL.md` untouched. The earlier draft idea of injecting procedural guidance into identity/personality state was rejected.

## Contradictions and repairs made during implementation

### Snapshot dirty flag versus empty branch delta

Contradiction: metadata says dirty, but the tracked branch patch/list/log are empty.

Resolution: patch only the exact named commit. Record the contradiction instead of inventing missing changes.

### Working-tree export omissions versus exact Git source

Contradiction: the tar export omits tracked repository-local and lock files that exist in the Git object store.

Resolution: use the exact bundled commit as the clean base. Do not encode omissions as deletions.

### Comprehensive manual versus pointer-only bootstrap

Contradiction: older notes/plans favored a small discovery pointer; later Beads operator notes require full standing instruction.

Resolution: deliver the full manual by default, with deep reference/live catalog supplemental. Report actual size and cache key rather than optimizing to an arbitrary ceiling.

### Hermes identity file versus skill mechanism

Contradiction: an earlier implementation draft targeted `SOUL.md`; current Hermes documentation identifies skills as the procedural knowledge mechanism.

Resolution: use `~/.hermes/skills/productivity/polylogue/SKILL.md`; never edit `SOUL.md`.

### Codex base guidance versus active override

Contradiction: writing only `AGENTS.md` can leave guidance completely shadowed by a non-empty `AGENTS.override.md`.

Resolution: select the effective global file, detect later shadowing, report it, and relocate the exact owned block during reconciliation.

### Equal value versus ownership

Contradiction: an earlier state description implied a value equal to desired Polylogue config could be claimed and later upgraded/removed, violating the non-destructive acceptance criterion.

Resolution: coincidentally equal operator values are satisfied but unowned. State records this and uninstall leaves them untouched.

### Session and terminal query surfaces

Contradiction: some draft recipes used valid-looking expressions with the wrong production parser.

Resolution: classify each expression, compile it through the correct parser, and retain deliberate negative boundaries to prevent a permissive regression.

### Stale manual command

Contradiction: a draft documented `polylogue agent manual --kind recipes`, but the CLI only supports `standing` and `reference`.

Resolution: corrected to `--kind reference`; the verifier now parses full option/argument vectors rather than only command names.

### Live manifest serialization

Contradiction: static/live registration enumeration passed, but invoking the manifest resource failed because a dictionary was sent to a Pydantic-only serializer.

Resolution: invoke registered resource functions in a smoke; serialize the dictionary directly with deterministic JSON; retain the full MCP unit test for the locked environment.

### Client-root semantics

Contradiction: treating every environment override as the exact configuration directory is wrong for Gemini’s containing-home isolation contract and for Claude’s split default locations.

Resolution: client-specific path resolution with dedicated tests for all four profile-root variables.

### Shared directory cleanup

Contradiction: one-pass per-client directory cleanup can leave installer-created shared parents because other selected clients still occupy them at the moment of removal.

Resolution: retry all recorded created-directory removals after every selected client has been processed.

## Evidence that would falsify the design

The following observations would require a repair rather than reinterpretation:

- A current provider release no longer reads the documented native config/guidance location or changes its precedence model.
- A real FastMCP SDK exposes different manager contracts or rejects the new instructions/resources despite the current production tests.
- Full locked query parsing rejects any of the 20 expressions that the bounded compatibility harness accepts.
- Home Manager rejects the module syntax/options or its activation command under a real evaluation.
- A live Claude `SessionStart` ignores the documented structured output.
- A client installer modifies unrelated content, claims coincidentally equal values, deletes drifted content, follows a symlink, or fails rollback under a real filesystem edge case.
- A built Nix package omits the package resources despite wheel/sdist inclusion.
- Blind trials show the standing manual reduces correct invocation, causes material overuse, or fails recovery; that would trigger content revision and ablation, not concealment.

## Environment limitations observed

Unavailable executable/tooling:

```text
nix
nix-instantiate
home-manager
ruff
mypy
```

Unavailable Python modules in the execution environment:

```text
dateparser
ijson
aiosqlite
mcp
hypothesis
sqlite_vec
```

The temporary compatibility modules used for the bounded registration harness are outside the repository, outside `PATCH.diff`, and outside the handoff ZIP. They are not a product dependency or proposed implementation.
