Title: "WebUI v2 design system: components, theming, accessibility, and the Vitest+Playwright verification harness"

Result ZIP: `webui-07-design-system-a11y-r01.zip`

## Mission

Give WebUI v2 one coherent visual/interaction system so the verticals
(webui-02…06, built by parallel jobs) compose into one product instead of
five pages. You are defining the system and harness they retrofit onto; make
adoption mechanically cheap.

Deliver:

1. **Component kit** (Preact + TypeScript, zero external UI framework):
   layout primitives, data table with continuation paging hooks, badge set
   (origin badges for the 10 public Origin tokens — read `core/enums.py`;
   evidence-state badges: exact/qualified/stale/unknown/degraded), code/
   diff/transcript blocks, facet chips, timeline/sparkline primitives (CSS/
   SVG, no chart library unless bundled and justified), skeleton/loading and
   empty-vs-unknown states (these are DIFFERENT states — an honest-absence
   pattern is part of the system).
2. **Theming**: light/dark via CSS custom properties, `prefers-color-scheme`
   default + explicit toggle; typography/spacing scale; density comfortable
   for dense evidence tables. No webfonts fetched at runtime — system stack
   or bundled.
3. **Accessibility**: keyboard navigation through lists/tables/facets,
   focus management for expand/disclosure patterns, ARIA on interactive
   islands, contrast-checked palette (state the checked ratios), reduced-
   motion respect. SSR pages must be readable and navigable without JS.
4. **Harness**: Vitest component tests for the kit; Playwright journeys
   (against a fixture-serving daemon or static SSR snapshots — document the
   route) covering: list→read→search journey, keyboard-only pass, a11y scan
   (axe or equivalent, bundled), and visual snapshots for light+dark.
   Deterministic: no network, no real archive.
5. **Adoption guide**: per-vertical retrofit notes (what webui-02…06 replace
   with kit components), lint/type rules that keep raw HTML patterns from
   creeping back.

## Constraints

- Zero CDN/network at runtime and in tests; everything vendored/bundled.
- The kit must not encode archive semantics (it renders states it is given;
  semantics stay server-side) — one exception: the Origin badge map and
  evidence-state vocabulary, which are PUBLIC contracts worth centralizing
  client-side; source their token lists from generated output if the
  snapshot has one (check `docs/cli-reference.md` generation machinery),
  else hardcode with a regeneration note.
- Sanitized fixture content only.

## Deliverable emphasis

HANDOFF.md: component inventory with props, theming tokens table, a11y
checklist results, harness invocation commands, adoption guide, and the
exact interface the other verticals must expose to slot into the Playwright
journeys.
