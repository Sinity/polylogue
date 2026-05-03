# Coding-agent prompt: Polylogue MK2 surfaces, web reader, and CLI polish

You are implementing the Polylogue MK2 surface direction. Do not code before inspecting the repository.

## First, inspect current state

Run or inspect equivalents:

```bash
git status --short
rg -n "QueryFirstGroupBase|RootModeRequest|query_first|query verbs|messages|raw|browser-capture|polylogue watch|polylogue mcp|dashboard|products|insights|derived" polylogue devtools tests docs pyproject.toml
polylogue --help || true
polylogue completions --help || true
polylogue dashboard --help || true
polylogue doctor --help || true
polylogue messages --help || true
polylogue raw --help || true
```

Then write a short implementation plan that identifies the exact current files. Do not assume old path names from a stale design artifact.

## Non-negotiables

- Preserve query-first CLI grammar: `polylogue [query] [filters] [verb]`.
- Do not replace it with a generic task-first tree.
- Do not add `polylogue ui`.
- Do not add deprecated aliases, compatibility windows, import shims, or old command wrappers.
- Do not use `--json`; machine output is `--format json` only.
- `polylogued` owns long-running local service work.
- `polylogue-mcp` is a separate MCP stdio executable.
- Web UI, TUI, CLI, MCP, and static/export views must consume shared query/read/status/derived contracts.
- Do not implement new frontend-only query/filter semantics.
- Do not copy MK2 demo data into production behavior.

## Implement the first vertical slice

Target:

```text
polylogued run
/api/health
/api/status
/api/conversations
/api/conversations/{id}
/api/conversations/{id}/messages
/api/conversations/{id}/raw
/api/facets
/c/{id}
polylogue --latest open
polylogue doctor daemon
polylogue select
```

### Step 1: add surface contracts

Create typed models for:

```text
DaemonStatus
ComponentHealth
ConversationListRequest
ConversationListResponse
ConversationHeaderResponse
MessagePageRequest
MessagePageResponse
RawArtifactPageResponse
FacetSet
OpenTarget
```

Place them in the existing surface/contracts package if present. If not, create a small ownership module that fits current topology, e.g. `polylogue/surfaces/local_app.py` or equivalent.

These models should wrap existing query/read payloads rather than inventing new shapes.

### Step 2: implement `polylogued`

Add a console script:

```text
polylogued = polylogue.daemon.serve:main
```

Daemon implementation target:

```text
polylogue/daemon/__init__.py
polylogue/daemon/app.py
polylogue/daemon/serve.py
polylogue/daemon/status.py
polylogue/daemon/web/...
```

Use a small local web app shape. If the repo already has ASGI dependencies or browser receiver code that justifies ASGI, use that. Otherwise use a thin local HTTP adapter only if it stays small. Do not grow another hand-rolled protocol framework.

`polylogued run` should be foreground. No service-manager code in this slice.

### Step 3: API uses existing query/read kernel

API list endpoint must call the same query spec/plan path as CLI list. It must not parse filters independently except through descriptor-backed mapping.

API messages endpoint must call the same message-page path as CLI/MCP.

API raw endpoint must call the same raw artifact/read surface as CLI/MCP. Help text should be truthful: raw archive artifact payloads, not provider-event pagination unless that actually exists.

### Step 4: web reader v1

Build a real local reader with:

- top status strip;
- left facet/search rail;
- result list;
- main conversation reader;
- right inspector tabs: outline, raw, provenance, derived, notes placeholder;
- bottom keyboard hint strip.

Use real API data. Hard-coded demo data is allowed only in storybook/prototype/test fixtures, not runtime.

Production bundle must be vendored/built into the Python package. Do not depend on CDN React/Babel at runtime.

### Step 5: `polylogue open`

When daemon is running, `polylogue --latest open` should open the local reader URL for the selected conversation. If daemon is down, fallback to the existing behavior and print a clear warning.

### Step 6: `doctor daemon`

Add daemon health to `polylogue doctor`. Use existing doctor target style if one exists; otherwise use subcommands. Pick one spelling. Do not expose two equivalent forms.

### Step 7: `polylogue select`

Implement descriptor/query-backed fuzzy conversation selection.

Minimum:

```bash
polylogue "daemon" --repo polylogue select
polylogue select conversation --recent
polylogue select repo --print value
```

Use fzf if configured/available, with pure-Python fallback. Preview should reuse existing `show`/`messages` renderers.

### Step 8: verification

Add tests:

```text
tests/daemon/test_smoke.py
tests/daemon/test_api_query_parity.py
tests/cli/test_open_daemon.py
tests/cli/test_select.py
tests/cli/test_completion_descriptor_contract.py
tests/visual/test_local_reader_smoke.py or closest existing visual lane
```

Required checks:

```bash
pytest -q tests/daemon tests/cli/test_open_daemon.py tests/cli/test_select.py
polylogued run --help
polylogue --help
polylogue --latest open --help || true
devtools verify --quick
```

If visual tooling exists, generate one screenshot of the local reader and one terminal/table capture. If not, add a pending scenario/evidence manifest rather than fake evidence.

## Explicit deferrals

Do not implement in slice 1 unless trivial after core work:

- annotations/stars/pins/saved views;
- recall pack builder;
- browser receiver migration into daemon;
- removing `polylogue watch`;
- removing `polylogue browser-capture serve`;
- splitting `polylogue-mcp`;
- derived-data public command cleanup.

But design the contracts so these can plug in without rewriting the app.

## Failure conditions

Stop and report if you find:

- query-first CLI has already changed substantially;
- current `doctor` target grammar conflicts with this plan;
- `products` has already been renamed and the derived-data question is partly closed;
- the repo has no viable local web serving dependency and adding one would be a packaging decision;
- frontend implementation would require a new Node toolchain in the user install path;
- visual verification infrastructure is absent.

In that case, propose the smallest adjusted slice; do not silently invent a parallel system.
