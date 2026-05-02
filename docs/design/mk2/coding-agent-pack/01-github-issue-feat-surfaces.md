# feat(surfaces): implement `polylogued` local reader and query-native CLI UX

## Problem

Polylogue now has a strong archive/query/read foundation, live watcher machinery, browser capture, MCP surfaces, derived archive read models, and serious verification infrastructure. The missing product layer is a coherent local interaction model.

The risk is interface sprawl:

- `polylogue` becomes polluted with long-running service/protocol commands.
- A web UI, TUI, CLI, and MCP surface each invent query/filter/read/status semantics independently.
- The CLI remains technically powerful but too hard to operate without serious dynamic completion and fuzzy selection.
- Static generated pages remain a half-measure for real browsing, annotation, pinning, and saved views.
- Derived read models stay detached as a `products`/`insights` island instead of appearing where the user reads chatlogs.

The MK2 design artifact has the correct spine: four surfaces over one shared kernel.

```text
polylogue       query-first archive CLI
polylogued      long-running local daemon
polylogue-mcp   MCP stdio adapter launched by MCP clients
devtools        source-checkout repo control plane
```

The shared kernel is:

```text
query     ConversationQuerySpec, RootModeRequest, query descriptors, strict validators, completion descriptors
read      conversation header, message page, raw artifact page, session tree
status    archive doctor, live ingestion, browser capture, derived readiness, daemon status
derived   session profile, phases, threads, costs, debt, recall facts
```

## Hard requirements

### Preserve the current CLI grammar

Do not replace the query-first CLI with a generic task tree.

Canonical shape remains:

```text
polylogue [query terms] [root filters] [verb]
```

Examples:

```text
polylogue
polylogue "fts trigger" --repo polylogue --since "last week" list
polylogue --has thinking --sort tokens list --limit 3 --format json
polylogue --latest show
polylogue messages <conversation-id> --message-role user --limit 50
polylogue raw <conversation-id> --format json
polylogue --latest open
polylogue "daemon" --repo polylogue select
```

`messages` is a paginated read surface. `raw` is a raw archive artifact surface. Do not collapse these away just because the web reader can display them together.

### Remove service/protocol modes from the interactive CLI

The following must not remain as public interactive-client commands once replacements land:

```text
polylogue watch
polylogue browser-capture serve
polylogue mcp
polylogue products
```

Replacement target:

```text
polylogued run              # live ingestion + browser receiver + local API + web reader
polylogue-mcp --role read   # MCP adapter
polylogue doctor daemon     # daemon health
polylogue doctor live       # live ingestion health
polylogue doctor capture    # browser capture health
```

If current `doctor` uses `--target`, keep one spelling consistently, but do not expose both `doctor capture` and `doctor --target capture` without a deliberate CLI decision.

### No aliases, no compatibility windows

Use clean breaks. This project is under development; do not add deprecated aliases, compatibility wrappers, import shims, or old command names unless a specific issue explicitly overrides this.

Machine output is `--format json` only. Remove new or existing `--json` mentions in this surface work.

### Do not create a detached `insights` island

Derived archive data should appear where the user is already looking:

- `polylogue show <id>` includes relevant derived/session facts.
- Web reader has a `derived` / `session facts` inspector tab.
- Dashboard shows stale/ready derived facts.
- Aggregate views may use concrete nouns such as `sessions`, `threads`, `costs`, `debt`.
- Keep a compact `derived status` / `derived export` admin group only if necessary.

Do not implement `polylogue insights` as a prettier `polylogue products` dump.

## Scope

### 1. `polylogued` daemon

Add a daemon executable that runs foreground by default:

```text
polylogued run [--host 127.0.0.1] [--port PORT] [--archive PATH] [--open]
```

Daemon responsibilities:

- local loopback API;
- local web reader static bundle;
- live ingestion supervision/status;
- browser-capture receiver/status, once migrated;
- bounded maintenance/status checks;
- event/status stream if useful.

Security/privacy:

- bind to loopback by default;
- require token auth for non-loopback binding;
- do not expose absolute local paths to browser clients except in authenticated/debug developer mode;
- never return raw Python/Pydantic/filesystem exception details to browser clients;
- use sanitized status summaries;
- browser-capture artifacts remain private-permission where applicable.

No `POLYLOGUE_DAEMON=1` feature flag. Ship the daemon or do not.

### 2. Local API contracts

API endpoints must be adapters over existing query/read/status contracts. They must not duplicate CLI filter semantics in frontend/backend web code.

Initial endpoints:

```text
GET /api/health
GET /api/status
GET /api/conversations
GET /api/conversations/{id}
GET /api/conversations/{id}/messages
GET /api/conversations/{id}/raw
GET /api/facets
GET /api/doctor
GET /api/live
GET /api/capture
```

Later endpoints:

```text
GET/POST/DELETE /api/marks
GET/POST/PATCH/DELETE /api/annotations
GET/POST/DELETE /api/saved-views
POST /api/recall-packs
```

### 3. Web reader

Build a real localhost reader, not a static export substitute.

Primary screens:

```text
/search                  faceted search and browse
/c/{id}                  conversation reader
/live                    live ingestion status
/capture                 browser-capture status/debug
/doctor                  archive/daemon health
```

First version layout:

```text
Top status strip: daemon, archive, live, capture, FTS/cache/schema
Left rail: search, saved views, facets
Result column: matching conversations/sessions
Main pane: conversation header + messages
Right inspector: outline, raw, provenance, notes, derived facts
Bottom hint strip: keyboard shortcuts and current query summary
```

Keyboard:

```text
/        focus search
Ctrl-K   command palette
g s      search
g l      live
g c      capture
g d      doctor
g r      recent/session list
[ ]      previous/next result
p        pin selected
s        star selected
a        annotate selected
y        copy id/link
o        open selected
Esc      close modal/palette
?        shortcut help
```

The right panel should not have a `messages` tab if the main pane is already the message reader. Use tabs such as:

```text
outline
raw
provenance
notes
derived
```

### 4. CLI enhancements

Upgrade the CLI where it is already strongest: query-first shell-native usage.

Add or improve:

- archive-backed dynamic completions;
- descriptor-backed completion metadata;
- `polylogue select` fuzzy selection;
- Rich/cold-cockpit human tables;
- consistent semantic colors;
- fast, bounded completions that do not run expensive FTS/readiness checks.

`polylogue select` target shape:

```text
polylogue [query] [filters] select
polylogue select conversation --recent
polylogue select repo
polylogue select cwd
polylogue select provider
polylogue select tag
polylogue select tool
polylogue select saved-view
```

Composition examples:

```text
polylogue messages "$(polylogue "fts trigger" select --print id)" --limit 50
polylogue --repo "$(polylogue select repo --print value)" list
polylogue "daemon" --repo polylogue select --open web
```

Use `fzf` if configured/available, with a pure-Python fallback. Do not make an external binary a hard dependency.

### 5. TUI/dashboard

Do not create a second reader. Keep or strengthen the existing terminal dashboard as an operator cockpit:

```text
polylogue dashboard
```

It answers:

- is the daemon alive?
- is live ingestion fresh?
- is browser capture connected?
- is FTS/cache/schema healthy?
- which recent sessions arrived?
- what needs attention?

It should consume the same status/query/read contracts as CLI and web.

### 6. `polylogue-mcp`

Move MCP out of the interactive client:

```text
polylogue-mcp --role read|write|admin
```

MCP should talk to `polylogued` when available and fall back to direct archive access when not. It must not require the daemon to run unless a future issue explicitly chooses daemon-only MCP.

### 7. Visual verification

The web UI and CLI visuals must become verifiable artifacts, not one-off screenshots.

Add scenario/exercise outputs for:

- CLI query/table screenshot;
- CLI fuzzy selector screenshot or terminal cast;
- local web reader screenshot;
- daemon/live/capture status screenshot;
- dashboard screenshot;
- README curated hero/media assets.

Visual checks should cover:

- generated artifact exists;
- expected dimensions;
- not blank;
- no clipping of key controls;
- no private absolute paths or private sample content;
- expected status text appears;
- generated from synthetic/demo fixture;
- current-generation evidence envelope exists;
- coding-agent/VLLM aesthetic review cell is fresh for README hero artifacts.

## First vertical slice

Implement the smallest visible slice that proves the architecture:

1. Add `polylogued` console script.
2. `polylogued run` serves loopback API and static web shell.
3. API exposes health, status, conversation list, conversation header, messages page, raw artifact page, and facets.
4. Web reader has search/facets/result list/conversation reader/provenance/raw/status strip using real API data.
5. `polylogue --latest open` opens daemon URL when daemon is running and falls back to existing behavior otherwise.
6. `polylogue doctor daemon` reports daemon status.
7. `polylogue select` exists for conversations, backed by existing query/read renderers.
8. Tests prove API/CLI parity for list/messages/raw.
9. Tests prove `--format json` remains canonical and no new `--json` path is added.
10. Screenshot/visual smoke artifact is generated for the web reader and CLI table.

Defer until later slices:

- annotations/stars/pins/saved views;
- recall-pack builder;
- moving browser-capture receiver into daemon;
- removing `polylogue watch` after `polylogued` actually owns live ingestion;
- splitting `polylogue-mcp` after daemon/web slice if needed;
- derived data public command cleanup.

## Acceptance criteria

- `polylogued --help` and `polylogued run --help` work from an installed wheel.
- `polylogued run` binds loopback and serves `/api/health` plus the local reader.
- API conversation list for a filter set returns the same IDs as `polylogue ... list --format json`.
- API message pagination returns the same IDs/content windows as `polylogue messages ... --format json`.
- API raw page matches `polylogue raw ... --format json` semantics.
- Web reader uses API data, not hard-coded MK2 demo data.
- Web frontend never implements query validation/filter semantics by itself.
- `polylogue select` uses query descriptors and existing query/read renderers.
- Shell completion includes providers, repos, cwd prefixes, tags, tools, message types, conversation IDs, doctor scopes, and saved views where data exists.
- CLI human output uses semantic colors and readable tables; non-TTY/plain mode remains clean.
- `polylogue mcp` is absent once `polylogue-mcp` lands.
- `polylogue watch` is removed only after `polylogued` owns equivalent live ingestion behavior.
- `polylogue browser-capture serve` is removed only after receiver endpoints move into `polylogued`.
- `polylogue products` is removed only after derived read models have inline/concrete replacement surfaces.
- No compatibility aliases are added.
- No new `--json` option is added, and old mentions are removed in touched help/docs/tests.
- Visual smoke artifacts exist for CLI table and web reader.
- `devtools verify --quick` passes.

## Related issues

- #620 live cursor/source freshness: daemon live status must expose the same content-aware cursor/freshness model.
- #621 query descriptors/read filters/raw semantics: web/API/completion/select must consume this contract.
- #622 browser capture: receiver/status must become one model for daemon/web/TUI/extension/doctor.
- #624 product/derived registry coverage: derived read models need consumer coverage without public `products` vocabulary.
- #625 devtools/control-plane: filter picker may become the seed for `polylogue select`; otherwise delete it.
- #593 distribution: installed scripts should include `polylogue`, `polylogued`, and `polylogue-mcp`; devtools should be source-checkout-only unless intentionally shipped.
- #635 naming: surface vocabulary must match this decision; no aliases.
- #618 storage: live daemon is not trustworthy until FTS/cache/write side effects are correct.
- #590/#594 visual and surface verification should make this non-vacuous.

Ref #578
