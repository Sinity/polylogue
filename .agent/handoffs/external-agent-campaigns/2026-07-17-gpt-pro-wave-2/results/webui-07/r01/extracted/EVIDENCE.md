# Source, Beads, and history evidence

## Snapshot reconstruction

The supplied project-state archive contained an all-ref Git bundle and rendered repository evidence rather than a directly usable worktree. The bundle was cloned to reconstruct the tracked source.

| Evidence | Finding |
| --- | --- |
| `polylogue-overview.json` | Branch `master`, commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d`, `dirty=true`, generated `2026-07-17T180950Z`. |
| `polylogue-manifest.json` | Project `polylogue`, source `/realm/project/polylogue`, same generation time, valid XML export. |
| `polylogue-branch-delta.md` | Base `origin/master`; merge base is the same commit; no diff stat and no branch-only commits. |
| `polylogue-branch-delta.patch` | Zero bytes. |
| `polylogue-branch-delta-files.txt` | Zero bytes. |
| `polylogue-branch-delta-log.txt` | Zero bytes. |
| Reconstructed Git worktree | Tracked tree clean before implementation; branch `master`; HEAD exactly matches the overview. |

The representable authority is therefore the commit. The snapshot's dirty flag cannot be reconciled to tracked bytes because every supplied dirty-delta artifact is empty. This package does not claim to reconstruct ignored files, unstaged local state, or operator-only secrets.

## Repository-level constraints

### `AGENTS.md`

The repository guidance establishes that substrate/domain modules own semantics and surfaces remain leaf adapters. Generated surfaces are refreshed through devtools, typed interfaces are preserved, and tests should target meaningful production behavior. This ruled out:

- provider-specific conditional behavior inside generic UI components;
- a separate client evidence model;
- hand-maintained browser token lists when Python authority exists;
- replacing current daemon APIs with a fixture-only schema.

### `CONTRIBUTING.md`

The contributing guidance identifies generated-surface checks, strict Python style/type expectations, focused test execution, and repository-standard environments. It supports registering the new renderer in the command catalog and generated-surface registry rather than invoking an untracked script directly.

## Current production route trace

### `polylogue/daemon/http.py`

The inspected route dispatch shows that the current browser paths `/`, `/s/:id`, and `/w/:mode` reach `_serve_web_shell`. `_serve_web_shell` imports and serves `WEB_SHELL_HTML`.

### `polylogue/daemon/web_shell.py`

The current shell is a large Python-composed HTML/CSS/JavaScript document. It remains the production route after this patch. That fact is decisive: mounting a new client without current auth, API, SSR, and asset ownership analysis would create an unsafe parallel surface.

### Consequence

This revision defines and proves the component/harness contract without changing those daemon routes. The fixture is explicitly a deterministic contract server, not a replacement production API. A later production integration must either use the ratified strangler route or an updated current-source route decision and must keep native SSR semantics.

## Public Origin authority

### `polylogue/core/enums.py`

`Origin` contains eleven enum values. Ten identify public providers/source families. `unknown-export` is a fallback for unrecognized export provenance.

The mission asks for ten public badges. The correct interpretation is:

- generate the public badge union from all enum values except `Origin.UNKNOWN_EXPORT`;
- render unknown provenance through a separate unknown badge/state;
- fail generation tests if the public count or exclusion changes unexpectedly.

This preserves the fallback's meaning rather than presenting it as an eleventh provider.

## Theme authority

### `polylogue/ui/theme.py`

The file already owns Python-side palette and rendering values. The ratified WebUI notes explicitly require browser tokens to be generated from this authority. The patch extends that file with browser-semantic values and keeps generation leaf-safe.

The new renderer reads the theme module through `runpy.run_path` instead of importing the entire `polylogue.ui` package. This avoids dragging unrelated runtime/storage/provider dependencies into a generation command while still executing the authoritative Python declarations.

## Existing WebUI baseline

Before the patch, `webui/` contained the package lock/manifest, Playwright configuration, and the first-party daemon credential journey. It did not contain the ratified Preact/Vite component scaffold, generated tokens, SSR fixture, Vitest lane, axe journey, or visual baselines.

The existing credential journey was preserved. Only a strict-index assertion was added so the new `noUncheckedIndexedAccess` configuration does not hide an empty fixture behind unsafe indexing.

## Beads findings

### `polylogue-bby.11` — WebUI architecture v2

The original design chooses TypeScript + Preact + Vite, committed/bundled offline assets, generated typed API bindings, shared components, keyboard-first behavior, and a strangler migration. Later notes refine and supersede the initial SPA framing:

- daemon routes serve semantic HTML plus typed JSON;
- Preact hydrates progressive-enhancement islands;
- SSR-first behavior must work for phone and non-JavaScript readers;
- tokens are generated from `theme.py` through `polylogue-9xuk`;
- `polylogue-1ilk` is binding for component, browser, accessibility, and visual tests;
- the recovered static cockpit is design reference only and must not become a parallel contract.

Decision applied: build an SSR-first Preact package and fixture; do not import the recovered prototype or mount a client-only SPA.

### `polylogue-1ilk` — Web test plan

The acceptance criteria require a documented test stack, a per-change component lane, an end-to-end journey, visual evidence, a seeded regression demonstration, and rebaseline procedure. Its intent also covers keyboard/focus behavior and accessibility.

Decision applied: Vitest for production components, Playwright for list -> reader -> search, stable slot assertions, keyboard-only, JavaScript-disabled, axe, and two committed visual baselines. `DESIGN_SYSTEM.md` and `TESTS.md` identify representative mutations that make each layer fail.

### `polylogue-bkzv` and `polylogue-9xuk` — visual semantics

These records establish one typed evidence-honest vocabulary across surfaces. Semantic/accessibility text comes first; color and glyphs are generated presentation roles. Unknown, unavailable, stale, degraded, and zero values must not collapse.

Decision applied:

- literal generated unions for origins/evidence states;
- visible state labels and redundant symbols;
- separate empty, unknown, degraded, loading, and error components;
- generated color pairs and measured contrast;
- no component-level inference of evidence semantics.

### `polylogue-lu1` — ambient theming

The record calls for CSS custom properties, `prefers-color-scheme`, curated themes, and deterministic captures. It rejects runtime webfont/CDN dependence.

Decision applied: generated light/dark semantic tokens, system default, explicit persisted toggle, system font stacks, reduced-motion handling, and light/dark snapshots.

### `polylogue-bby.10` — timeline/firehose

The detailed future view favors virtualized SVG and server-owned time buckets. The mission in this package asks only for reusable timeline/sparkline primitives and forbids archive semantics in the kit.

Decision applied: ordered semantic `Timeline` plus labelled SVG `Sparkline`; no chart dependency, no raw archive scan, no invented time-bucket endpoint, and no attempt to implement the full future firehose.

## Handoff and review findings

### `.agent/handoffs/polylogue-gpt-pro-2026-07-06/B-bby11-cockpit.md`

This handoff reinforces the Preact/Vite choice, generated API/token direction, and need to avoid deepening JavaScript embedded in Python strings. It also distinguishes foundation work from later cockpit features.

### `.agent/handoffs/polylogue-legibility-kit-v2-2026-07-10/web-cockpit-v2-delivery-summaries.md`

This record documents visual vocabulary and delivery expectations across the web cockpit work. Useful vocabulary was treated as design evidence, not as an independent client schema.

### `.agent/scratch/archive/scratch-2026-07-16-loose-files/2026-07-10-webui-rewrite-and-proof-audit.md`

The audit's load-bearing findings are:

- the current shell is Python raw strings plus injected JavaScript/CSS;
- the ratified direction is TypeScript + Preact + Vite with SSR semantic HTML and progressive enhancement;
- test evidence should include axe, keyboard/focus, reduced motion, and visual proof;
- the strongest first proof is an SSR list/reader slice with generated tokens and Playwright;
- the old shell cannot be retired before parity.

Decision applied: the deterministic fixture composes list, reader, search, evidence, and timeline through one contract, while production retirement is explicitly deferred.

## Relevant Git history inspected

| Commit | Subject | Relevance |
| --- | --- | --- |
| `9163d0134` | `feat(query): bound agent-facing archive reads (#3018)` | Reinforces bounded read semantics and avoiding client-invented archive behavior. |
| `1d3145afa` | `feat(archive): admit Test Diet survivor laws (#3044)` | Confirms current test/evidence discipline. |
| `efadb404e` | `feat(evidence): add provenance value canaries (#3033)` | Supports explicit provenance and value-state treatment. |
| `fc770dbd9` | `feat(reader): render ordered semantic transcripts (#3016)` | Current transcript semantics to preserve. |
| `876358610` | `fix(query): bound multi-field aggregate execution (#3012)` | Supports bounded, server-owned aggregation. |
| `fd7b35492` | `feat(query): interruptible, admission-controlled archive read execution (#2964)` | Supports abort-aware continuation and no client raw scans. |
| `ad3eb5e77` | `fix: make silent exception swallows loud across evidence surfaces (#2963)` | Supports degraded/error honesty. |
| `c1f7704fa` | `feat(web): add bounded cockpit aggregate routes (#2793)` | Existing web route/read model context. |
| `0c251b600` | `feat(rendering): wire semantic transcript cards into the web reader (#2736)` | Existing reader render semantics. |
| `5479beabc` | `fix(web): render truthful loading/stale/error states, not false emptiness (#2673)` | Direct authority for empty-versus-unknown/degraded behavior. |
| `79fb50d94` | `feat(web): evidence-cockpit IA — four-verb navigation, landing view, session evidence strip (#2675)` | Existing cockpit information architecture context. |
| `0e0cddaee` | `fix(web): bootstrap first-party daemon credentials (#2715)` | Reason to preserve and continue running the existing auth journey. |

No history was found showing a completed production Preact mount that current source had superseded. Current source therefore wins over plans that describe an already-existing scaffold.

## Generated-surface machinery

The implementation inspected and follows:

- `devtools/command_catalog.py` for public command registration;
- `devtools/generated_surfaces.py` for renderer ownership and cache inputs;
- `devtools/render_support.py` behavior through existing renderer tests;
- `docs/devtools.md` generated command reference;
- `tests/unit/devtools/test_command_catalog.py`;
- `tests/unit/devtools/test_generated_surfaces.py`;
- `tests/unit/devtools/test_render_devtools_reference.py`.

The renderer is not a one-off npm script. It participates in repository generation drift checks and exposes the same public command style as other generated surfaces.

## Contradictions and resolution

| Apparent contradiction | Resolution |
| --- | --- |
| Snapshot says dirty, supplied branch delta is empty. | Use the named commit as tracked authority and disclose that ignored/local state cannot be reconstructed. |
| Early `bby.11` wording describes SPA/list-reader parity; later ratified notes require SSR semantic HTML and progressive enhancement. | Later notes supersede early framing. Build SSR-first and keep native paths. |
| A recovered static cockpit offers vocabulary/schema ideas. | The later no-import ruling makes it non-authoritative. Use only compatible vocabulary; do not import its envelope or client architecture. |
| `Origin` has eleven values; mission asks for ten badges. | `unknown-export` is a fallback, not a public origin. Generate ten and render unknown separately. |
| Ratified architecture wants committed production assets; no current production Preact mount exists. | Commit generated source contracts and visual evidence, build deterministically, but do not invent a production static target. Record committed dist/mount as future integration. |
| Mission asks for a real end-to-end behavior; production verticals are parallel work. | Use one built SSR fixture with real component modules and native routes, not disconnected stories or mock component screenshots. Keep production daemon integration explicitly incomplete. |
| Full future timeline includes server bucket APIs and virtualization. | Deliver generic ordered/SVG primitives only; server/archive semantics remain outside the kit. |
| Zero runtime network conflicts with npm/browser installation in CI. | Runtime and tests make no application network requests; package/browser installation is a build-time dependency acquisition step. The lock is complete and runtime assets are bundled. |

## Evidence-to-implementation mapping

| Evidence | Implementation consequence |
| --- | --- |
| Substrate owns semantics | Components accept already-decided origin/state/role values. |
| Generated tokens from `theme.py` | Python renderer emits CSS, TypeScript, and contrast JSON. |
| Public Origin contract | Exact ten-token generated union; fallback separate. |
| Truthful loading/stale/error history | Honest state components and no empty/unknown collapse. |
| SSR-first ratification | URL-driven SSR fixture, native links/forms/disclosure, no-JS journey. |
| `1ilk` test binding | Vitest, Playwright, axe, keyboard, visual snapshots, rebaseline docs. |
| Offline/no CDN | System fonts, local assets, no UI/chart framework, external-request assertion. |
| Agent-buildable scaffold | Typed exports, complete prop inventory, stable vertical interface, lint gates. |
| Old shell current authority | No unsafe production route replacement in this patch. |

## Source and test coverage inspected beyond obvious files

The implementation followed dependencies into:

- daemon route dispatch and shell serving;
- enum and role authorities;
- Python theme ownership;
- generated command catalog and docs;
- existing WebUI auth fixture and Playwright config;
- package/CI workflow behavior;
- Beads dependency and later-note supersession;
- relevant transcript, provenance, truth-state, bounded-query, and web-route history.

No operator live daemon, browser profile, archive database, provider export, credential, cloud source, NixOS deployment, or active worktree outside the supplied snapshot was treated as evidence.

## Remaining evidence needed for production integration

A production-mount revision should inspect and prove, against then-current source:

- exact daemon SSR/JSON route ownership and authentication middleware;
- generated OpenAPI client output and route payloads consumed by each vertical;
- static asset packaging and cache/CSP headers;
- route coexistence/strangler behavior with the old shell;
- list and reader parity on a sanitized production fixture;
- current outputs of `webui-02` through `webui-06` parallel jobs;
- repository-native Ruff, mypy, full Python generated-surface tests, and full Playwright auth suite;
- wheel/Nix inclusion of built assets.

Those are integration facts that could not safely be inferred from plans alone.
