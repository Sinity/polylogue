# Polylogue WebUI v2 design system

This directory is the first-party Preact + TypeScript design-system substrate for
WebUI v2. It gives the independently developed `webui-02` through `webui-06`
verticals one generated vocabulary, one semantic component kit, and one
deterministic verification route. It does not interpret archive payloads.
Server-side owners decide whether evidence is exact, qualified, stale, unknown,
or degraded and pass that state into the kit.

The executable reference is the sanitized SSR fixture in `src/fixture/`. It
contains no real archive rows, credentials, daemon dependency, remote font, CDN,
or runtime network request.

## Install and verify

From `webui/`:

```console
npm ci
npm run check
npm run install:e2e-browser
npm run test:e2e:design-system
```

`npm run check` verifies generated files, design-system lint rules, strict
TypeScript, Vitest component tests, and both Vite builds. The focused Playwright
command builds and starts `scripts/fixture-server.mjs` on an ephemeral
`127.0.0.1` port. The complete existing browser lane remains `npm run test:e2e`.

To use a preinstalled Chromium explicitly during local verification:

```console
PLAYWRIGHT_CHROMIUM_EXECUTABLE=/usr/bin/chromium npm run test:e2e:design-system
```

Regenerate public contracts after changing `polylogue/core/enums.py`,
`polylogue/ui/theme.py`, or the evidence vocabulary:

```console
npm run generate
npm run generate:check
```

The repository-level equivalent is `devtools render webui-design-system`.
Never edit `src/generated/contracts.ts`, `src/generated/tokens.css`, or
`src/generated/contrast-report.json` by hand.

## Imports

```tsx
import {
  DataTable,
  EvidenceStateBadge,
  OriginBadge,
  VerticalFrame,
  type DataColumn,
} from './design-system';
import './design-system/design-system.css';
```

The package exports the TypeScript entry at `.` and the CSS surfaces at
`./tokens.css` and `./design-system.css`. Production bundling may resolve those
exports through the repository's final WebUI build once that mount is added.

## Component inventory

### Layout and controls

| Export | Props and contract |
| --- | --- |
| `SkipLink` | `target?: string`, default `main-content`. The destination must be focusable. |
| `Stack` | `children`, `className?`, `space?: 1..8`; vertical layout. |
| `Cluster` | `children`, `className?`, `space?: 1..8`; wrapping inline layout. |
| `Grid` | `children`, `className?`, `min?: string`, `space?: 1..8`; responsive auto-fit grid. |
| `Surface` | `children`, `className?`, `as?: 'div' | 'section' | 'article'`. Choose the semantic element for the content. |
| `Button` | Native button attributes plus `variant?: 'primary' | 'secondary' | 'quiet'`; defaults to `type="button"`. |
| `PageHeader` | `title`, `eyebrow?`, `description?`, `actions?`; generic page header without a vertical contract. |
| `VerticalFrame` | `VerticalFrameProps`: `id`, `state`, `title`, `description?`, `actions?`, `children`; emits the exact harness slot described below. |
| `SearchField` | `label`, `name?`, `defaultValue?`, `action?`, `placeholder?`; native GET search form, default action `/search`. |
| `VisuallyHidden` | `children`; accessible text with no visual box. |
| `ExternalLink` | Native anchor props; adds `rel="noreferrer"`. Callers still supply an accessible name and `href`. |

### Public state and provenance

| Export | Props and contract |
| --- | --- |
| `OriginBadge` | `origin: OriginToken`; accepts only the ten generated public Origin tokens. |
| `UnknownOriginBadge` | No props; renders explicit unknown provenance. `unknown-export` is not promoted into the public badge union. |
| `EvidenceStateBadge` | `state: EvidenceState`, `qualifiedBy?: string`; always exposes text and a non-color symbol. |
| `Skeleton` | `lines?: number`, `label?: string`; visual placeholders are `aria-hidden` behind a live status label. |
| `HonestState` | `kind: loading | empty | unknown | degraded | error`, `title?`, `description`, `action?`. |
| `LoadingState` | `description?`; combines an honest loading state with a skeleton. |
| `EmptyState` | `description`; means the operation completed and found no data. |
| `UnknownState` | `description`; means absence could not be established. |
| `DegradedState` | `description`; readable partial evidence. |
| `RetryState` | `description`, `onRetry`; error state with a first-party button. |

`empty` and `unknown` are not interchangeable. A completed query returning zero
rows is `empty`. A missing source, unconsulted dependency, incomplete frame, or
unavailable record is `unknown` or `degraded`, according to the server-owned
contract.

### Dense data and continuation

`DataTable<Row>` takes:

| Prop | Type / behavior |
| --- | --- |
| `caption` | Required accessible table name. |
| `rows` | `ReadonlyArray<Row>`. |
| `columns` | `ReadonlyArray<DataColumn<Row>>`; each column supplies `id`, `header`, `cell`, optional alignment, and optional responsive priority. |
| `rowKey` | Stable `(row) => string`. |
| `density` | `comfortable | compact`, default `comfortable`. |
| `onRowActivate` | Optional keyboard/double-click activation callback. Native links inside cells remain the no-JS path. |
| `continuation` | Optional `DataTableContinuation`: `hasMore`, `loading`, `error?`, `onLoadMore`, `label?`. |
| `absence` | `empty | unknown`, default `empty`. |
| `absenceDescription` | Required truthful explanation when no rows are rendered. |

Rows use roving keyboard focus. Up/Down, Home/End, and Enter are supported.
The scroll region itself is focusable so keyboard users can discover and scroll
wide evidence tables.

`useContinuationPaging<Item, Cursor>` takes `initialItems`, `initialCursor`, and
an abort-aware `loadPage(cursor, signal)`. It returns `items`, `cursor`,
`hasMore`, `loading`, `error`, `loadMore`, and `reset`. `reset` aborts the active
request and advances a generation counter; an old response cannot append into a
new query or facet state.

### Facets

`FacetChipGroup<Value>` takes a group `label`, `FacetOption<Value>[]`, a
`ReadonlySet<Value>` named `selected`, and `onChange`. Chips use native buttons,
`aria-pressed`, one maintained tab stop, wrap-around arrow navigation, and
Home/End.

### Evidence content

| Export | Props and contract |
| --- | --- |
| `CodeBlock` | `code`, `language?`, `caption?`; Preact text escaping only, no HTML injection. The preformatted region is keyboard-scrollable. |
| `DiffBlock` | `diff`, `caption?`; added and removed lines carry hidden `Added:` / `Removed:` labels so color is redundant. |
| `TranscriptBlock` | `messages: TranscriptMessage[]`, `label?`; semantic ordered list with role, author, time, body, and optional evidence badge. |
| `Disclosure` | `summary`, `children`, `open?`; native `details`/`summary`, readable and operable without enhancement. |

`TranscriptMessage.role` is `user | assistant | system | tool | unknown`. This is
presentation input, not a parser or archive role classifier.

### Timeline and trend

`Timeline` takes `TimelineItem[]` and `label?`, and renders an ordered list with
native `time` elements and optional evidence badges. `Sparkline` takes numeric
`values`, a required accessible `label`, and optional `width`/`height`. It uses
one inline SVG polyline, title, and description; there is no chart dependency.

### Theme

`ThemeToggle` cycles `system -> light -> dark -> system`. `system` removes the
explicit root attribute so `prefers-color-scheme` remains authoritative.
`readThemePreference` and `applyThemePreference` are exported for a future
server/bootstrap integration. The explicit preference is stored under
`polylogue.webui.theme`.

## Generated contracts

`src/generated/contracts.ts` contains:

- `DESIGN_SYSTEM_CONTRACT_VERSION = 1`;
- `PUBLIC_ORIGINS` and `OriginToken`, generated from every `Origin` except the
  `unknown-export` fallback;
- `ORIGIN_LABELS` for those ten tokens;
- `EVIDENCE_STATES = exact | qualified | stale | unknown | degraded` and labels.

The CSS badge selectors are generated from the same Python values. Adding or
removing an Origin without regeneration makes `npm run generate:check` fail;
the Python renderer test also requires exactly ten public tokens and rejects
`unknown-export`.

## Theme token table

All authored component CSS consumes custom properties. Raw hexadecimal colors
outside `src/generated/` fail `npm run lint`.

### Shared scale

| Token family | Tokens | Values |
| --- | --- | --- |
| Sans font | `--pl-font-sans` | System UI stack; no runtime font fetch. |
| Mono font | `--pl-font-mono` | System monospace stack; no runtime font fetch. |
| Type sizes | `--pl-font-size-xs/sm/md/lg/xl` | `0.75rem`, `0.8125rem`, `0.9375rem`, `1.125rem`, `1.5rem`. |
| Line height | `--pl-line-height-tight/body` | `1.25`, `1.55`. |
| Space | `--pl-space-0..8` | `0`, `0.25`, `0.5`, `0.75`, `1`, `1.5`, `2`, `3`, `4rem`. |
| Radius | `--pl-radius-sm/md/lg` | `0.25`, `0.5`, `0.75rem`. |
| Borders/focus | `--pl-border-width`, `--pl-focus-width` | `1px`, `3px`. |
| Dense rows | `--pl-density-row-comfortable/compact` | `2.75rem`, `2.125rem`. |
| Measures | `--pl-content-measure`, `--pl-transcript-measure` | `76rem`, `72ch`. |
| Motion | `--pl-motion-fast/normal` | `120ms`, `180ms`; both become `0ms` under reduced motion. |

### Semantic palette

| Token | Light | Dark |
| --- | --- | --- |
| `--pl-color-bg` | `#f7f9fb` | `#0b1015` |
| `--pl-color-surface` | `#ffffff` | `#111820` |
| `--pl-color-surface-raised` | `#eef2f6` | `#18222d` |
| `--pl-color-surface-inset` | `#f3f6f8` | `#0d141b` |
| `--pl-color-text` | `#18212b` | `#eef4f8` |
| `--pl-color-text-muted` | `#465564` | `#b3c0cb` |
| `--pl-color-text-subtle` | `#607080` | `#8998a6` |
| `--pl-color-border` | `#b8c3ce` | `#344453` |
| `--pl-color-border-strong` | `#7b8a99` | `#526578` |
| `--pl-color-accent` | `#075985` | `#7dd3fc` |
| `--pl-color-accent-strong` | `#0369a1` | `#38bdf8` |
| `--pl-color-focus` | `#92400e` | `#fbbf24` |
| `--pl-color-selection` | `#d9edf7` | `#14354a` |
| `--pl-color-code-bg` | `#f1f5f9` | `#090e13` |
| `--pl-color-shadow` | `rgba(15, 23, 42, 0.16)` | `rgba(0, 0, 0, 0.42)` |

Each evidence state and public Origin also has generated `-fg`, `-bg`, and
`-border` variables. The complete machine-readable evidence is
`src/generated/contrast-report.json`.

The renderer checks 42 opaque foreground/background pairs with WCAG 2.x
relative luminance. The current minimum normal-text ratio is **4.82:1** and the
minimum focus-indicator ratio against a surface is **7.09:1**. Normal text and
badge pairs must remain at least 4.5:1; focus pairs must remain at least 3:1.

## Exact vertical slot contract

Every `webui-02` through `webui-06` route must render exactly one
`VerticalFrame` and therefore exactly one root matching:

```css
main#main-content[data-webui-contract="1"][data-webui-vertical][data-state]
```

For vertical `webui-NN`, the required DOM is:

```html
<main
  id="main-content"
  tabindex="-1"
  data-webui-contract="1"
  data-webui-vertical="webui-NN"
  data-state="ready|loading|empty|unknown|degraded|error"
  aria-labelledby="webui-NN-heading"
>
  <header>
    <h1 id="webui-NN-heading">Vertical title</h1>
  </header>
  <section aria-label="Vertical content">Vertical content</section>
</main>
```

The TypeScript interface is `VerticalFrameProps` in
`src/design-system/vertical-contract.ts`. IDs and states are exported as
`WEBUI_VERTICAL_IDS`, `WEBUI_VERTICAL_STATES`, and
`WEBUI_VERTICAL_CONTRACT`. Playwright's
`tests/support/vertical-contract.ts::expectVerticalContract` is the executable
consumer. Do not fork or reproduce this markup in a vertical.

The deterministic fixture maps the journeys as follows:

| Vertical | Fixture route | Required native path |
| --- | --- | --- |
| `webui-02` list | `/` | Table cell links open `/sessions/:id`; GET search form opens `/search?q=term`. |
| `webui-03` reader | `/sessions/sanitized-alpha` | Transcript and native disclosure are in SSR; GET search and back link work without JS. |
| `webui-04` search | `/search?q=evidence` | Result table and session links are in SSR. |
| `webui-05` evidence states | `/evidence` | All public origin/evidence badges and honest states are present. |
| `webui-06` timeline | `/timeline` | Ordered timeline and labelled SVG are present. |

Real production routes can differ, but each vertical must expose the same root
contract and native list -> reader -> search capabilities to the production
Playwright adapter. The fixture is a contract harness, not a parallel archive
API.

## Per-vertical retrofit

### `webui-02`: inventory/list

Replace page wrappers with `VerticalFrame`, raw table markup with `DataTable`,
filter buttons with `FacetChipGroup`, and provider strings with
`OriginBadge`/`UnknownOriginBadge`. Feed continuation cursors through
`useContinuationPaging`; abort/reset it whenever server-owned filters or query
identity change. Keep an ordinary session anchor in the first cell even when
row activation is also enabled.

### `webui-03`: reader

Use `VerticalFrame`, `TranscriptBlock`, `Disclosure`, `CodeBlock`, and
`DiffBlock`. Preserve native links, headings, time elements, and expanded
content in SSR. The server still chooses role/evidence/provenance values; the
component kit only renders them.

### `webui-04`: search

Use `SearchField` for the GET contract and `DataTable` for results. Render a
specific `EmptyState` after a completed zero-result query. Use `UnknownState`
or `DegradedState` instead when the search backend did not establish a complete
result frame.

### `webui-05`: evidence/analysis

Use generated `EvidenceStateBadge`, `OriginBadge`, and honest-state components.
Never map an unrecognized origin to a nearby public provider. Never encode
exactness, age, failure, or unknown status through color alone.

### `webui-06`: timeline/trends

Use `Timeline` for event order and `Sparkline` only for bounded scalar series.
Supply a complete accessible label; keep the numeric values visible elsewhere
when they are decision-relevant. Do not add a chart runtime for these
primitives.

## Accessibility acceptance checklist

- A skip link targets the focusable main region.
- Every route has one `h1`; component sections use ordered headings.
- Tables have captions, scoped row/column headers, a discoverable scroll
  region, row keyboard navigation, and native cell links.
- Facets use buttons, `aria-pressed`, one tab stop, arrow keys, and Home/End.
- Disclosure uses native `details` and `summary`; no focus trap is introduced.
- Loading, errors, and continuation changes expose text through status/alert
  regions.
- Empty and unknown states have different machine-readable attributes and
  prose.
- Badge meaning is carried by text and symbol as well as color.
- Code, diffs, and transcripts are rendered as text; `dangerouslySetInnerHTML`
  is forbidden.
- Focus outlines use a generated 3px token and pass the recorded 3:1 threshold.
- The generated palette passes recorded normal-text contrast thresholds in
  light and dark modes.
- Motion tokens collapse to zero under `prefers-reduced-motion: reduce`;
  skeleton animation is disabled there.
- SSR links, GET forms, tables, transcript content, and native disclosures are
  readable and navigable with JavaScript disabled.
- Axe runs on list, reader, evidence, and timeline routes; keyboard and no-JS
  journeys are independent Playwright tests.

## Mechanical adoption gates

`npm run lint` parses source with the TypeScript compiler. It rejects:

- `dangerouslySetInnerHTML` anywhere under `src/`;
- raw `button`, `code`, `details`, `pre`, `summary`, `svg`, and table-family
  elements in `src/fixture/` or future `src/verticals/` modules;
- raw hexadecimal color literals in non-generated CSS.

The vertical restriction intentionally leaves semantic structural HTML such as
headings, paragraphs, lists, links, forms, `dl`, and `time` available. When a
new governed interactive primitive is added, add it to the kit and then extend
the restricted tag set.

`tsconfig.json` enables strict mode, unchecked-index checks, unknown catch
variables, isolated modules, and bundler resolution. Public Origin and evidence
states are generated literal unions; invalid tokens fail at type-check time.

## Harness design and anti-vacuity

Vitest exercises the production component modules directly. The tests fail if:

- `unknown-export` enters the public Origin list;
- state text/symbols, escaped code, diff labels, native disclosure, timeline
  labels, or SVG descriptions are removed;
- facet/tabular keyboard handlers or maintained tab stops are removed;
- continuation abort/generation checks are removed and a stale page appends;
- theme persistence or explicit/system switching is removed;
- SSR omits semantic tables, links, forms, vertical attributes, or honest
  empty/unknown distinctions.

Playwright starts the built SSR fixture and fails if:

- list -> load continuation -> reader -> search stops composing;
- any request leaves the local fixture origin;
- any of the five vertical roots deviates from contract version 1;
- keyboard-only focus order/facet/table activation breaks;
- the same list -> reader -> search path stops working with JavaScript off;
- axe reports a violation on the representative routes;
- light or dark evidence snapshots change without an explicit rebaseline.

Rebaseline only after reviewing both images:

```console
npm run test:e2e:update
npm run test:e2e:design-system
```

The snapshots are committed under `tests/snapshots/design-system.spec.ts/`.
