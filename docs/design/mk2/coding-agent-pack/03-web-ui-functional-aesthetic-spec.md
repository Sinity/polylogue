# Polylogue local web reader: functional and aesthetic spec

## Product role

The web reader is the sane way to browse, read, annotate, pin, star, save views, inspect provenance, and understand live/capture status. It is not a static export. It is not a generic SaaS dashboard. It is a localhost archive reader served by `polylogued`.

## Visual language

Direction: cybernetic calm / cold monochrome cockpit.

Avoid:

- warm cream editorial Claude default;
- purple-blue gradient SaaS;
- giant rounded cards with fake metrics;
- fluffy AI chat sidebar;
- graph-first interfaces;
- large empty dashboard tiles.

Use:

- graphite/near-black surfaces;
- slate borders;
- thin gridlines;
- muted cyan/green/amber/red status accents;
- compact readable typography;
- monospace for IDs, commands, paths, provider tokens;
- strong keyboard affordances;
- visible privacy/local-only state.

## Layout

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ status strip: archive ok · live 0.4s · capture connected · fts ok · local │
├──────────────┬─────────────────────┬──────────────────────┬──────────────┤
│ search/facet │ result list          │ reader               │ inspector    │
│ rail         │                     │ header + messages    │ outline/raw/ │
│              │                     │                      │ provenance/  │
│              │                     │                      │ notes/derived│
├──────────────┴─────────────────────┴──────────────────────┴──────────────┤
│ hints: / search · Ctrl-K palette · p pin · s star · a annotate · y copy    │
└────────────────────────────────────────────────────────────────────────────┘
```

### Top status strip

Show compact status chips:

- daemon state and uptime;
- archive path redacted display form;
- live ingestion freshness / stale / paused;
- browser capture receiver and extension state;
- FTS/cache/schema health;
- current query result count;
- local-only/privacy indicator.

Each chip opens the relevant status route or panel.

### Left rail

Search and filters:

- query box;
- saved views;
- provider facets;
- repo facets;
- cwd prefix facets;
- tags;
- message type;
- has flags: thinking, tool use, paste, action events;
- time range;
- typed-only / authored filters if present.

Facet counts must come from `/api/facets`, not frontend scans over a partial result set.

### Result list

Each item:

- short ID;
- provider badge;
- repo/cwd;
- title or first useful turn;
- started/updated time;
- message count;
- token/cost if available;
- flags: tools, thinking, paste, raw, capture, live;
- star/pin marks.

Use high-density rows, not big cards.

### Reader pane

Reader header:

- provider, model, repo, cwd;
- title/context;
- time span;
- counts;
- derived/session fact chips;
- actions: copy ID, open raw, export, build recall pack later.

Message blocks:

- user / assistant / tool / system / summary / thinking roles clearly distinguished;
- prose is readable;
- code blocks are compact but not cramped;
- tool calls are collapsible with command/path/status;
- pasted content can be collapsed with explicit `paste` marker;
- message anchors visible on hover/focus;
- keyboard navigation between messages.

### Right inspector

Tabs:

- `outline`: message jump list and semantic sections.
- `raw`: raw archive artifacts, not fake provider-record pagination.
- `provenance`: source path display token, source provider, cursor/parser/materialization facts, hash/dedupe facts.
- `notes`: annotations, stars, pins; placeholder in slice 1 if not implemented.
- `derived`: session profile, phase, thread, cost/debt/readiness facts.

Do not put a `messages` tab here if messages are already the reader's main pane.

## Keyboard model

```text
/        focus query
Ctrl-K   command palette
?        help overlay
g s      search
g l      live
g c      capture
g d      doctor
g r      recent sessions
[ / ]    previous/next result
j / k    next/previous result or message depending focus
Enter    open selected result
o        open selected in reader/new tab based on context
p        pin selected target
s        star selected target
a        annotate selected target
y        copy stable link/id
r        refresh current route
Esc      close overlay / clear focus
```

## Command palette

Commands should map to backend operations:

- Search in current repo.
- Open conversation by ID.
- Show live sources.
- Show capture status.
- Run doctor check.
- Create saved view.
- Pin current message.
- Star current conversation.
- Copy MCP resource URI.
- Build recall pack from selected pins.

## States

### Empty archive

Message: “No conversations archived yet.”

Actions:

- show `polylogue run --input PATH` if still current;
- show daemon/live setup if configured;
- show browser capture setup if extension installed.

### No results

Show active filters and one-click removals. Offer fuzzy selector or broaden query. Do not show fake recommendations.

### Daemon degraded

Show exact degraded component:

- live stale;
- capture receiver disconnected;
- FTS stale;
- schema read-only;
- archive locked;
- derived rollup stale.

### Privacy/capture warning

If capture is auto by default, show that clearly:

```text
capture: auto · connected · supported page · writes local archive
```

Include disable/pause action if implemented.

### Loading

Use skeleton rows with stable layout, not spinners everywhere.

### Failure

Show sanitized error summaries and next diagnostic command, e.g. `polylogue doctor capture`.

## Accessibility

- All keyboard operations visible in help overlay.
- Focus rings have high contrast.
- Status colors are accompanied by text/icons.
- Tables/lists are navigable without mouse.
- Contrast target: readable on GitHub/browser screenshots.

## Implementation notes

- Runtime web bundle must be local/vendored; no CDN React/Babel.
- Frontend should be small. React/Preact/Solid/vanilla is less important than respecting backend contracts.
- The first route can be `/search` with embedded reader; `/c/{id}` should deep-link cleanly.
- Use hash-free stable URLs where possible.
- Static export/site can reuse visual tokens later, but local reader is primary.
