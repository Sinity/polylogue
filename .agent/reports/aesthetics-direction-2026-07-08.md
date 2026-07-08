# Polylogue Aesthetics Direction

Status: awaiting operator sign-off (bead polylogue-tjx1)
Date: 2026-07-08
Scope: web shell, canonical HTML renderer, pages theme, TUI, CLI output vocabulary.

## Thesis

Polylogue is a **forensic instrument** — the flight recorder for AI work. The
feeling every surface must induce is *"this tool is not lying to me."*
Aesthetics here is not decoration; it is the visual arm of the honesty
doctrine (evidence provenance, degraded-modes-are-loud, number-over-empty).
Instruments inspire confidence through precision typography, restrained color
that always means something, visible provenance, and high signal density held
together by calm hierarchy — not through gloss.

## Anti-goals

- **SaaS landing-page feel.** Gradient text, hero typography, marketing
  spacing. (The current `rendering/templates/session.html` title gradient is
  the one surface that reads off-brand.)
- **Dashboard soup.** Panels for panels' sake; charts that decorate rather
  than answer a question.
- **Chat-app cosplay.** Bubble UIs. This is an archive reading instrument,
  not a messenger.
- **Decoration color.** Any hue that does not encode role, origin, status, or
  provenance.

## Comparison points

Terminal-native tooling density (lazygit, k9s), Linear's keyboard-first
density, Obsidian's local-first calm, sourcehut's legibility-over-chrome,
and instrument panels (aviation/medical): status conveyed redundantly through
shape + color + position, never color alone.

## Principles

1. **Color is semantic or absent.** One palette: neutral base + role hues
   (user / assistant / tool / system) + provider brand hues (already in
   `ui/theme.py` PROVIDER_COLORS) + status triad (ok/warn/err) + provenance
   treatments. Verified/reported = solid; derived/estimated = muted;
   unknown/degraded = dashed or hatched. Nothing else gets a color.
2. **Density with rhythm.** Keep the dense default (the web shell's 13px base
   is right). Legibility comes from rhythm, not whitespace: consistent row
   heights, tabular numerals for all metrics, strict grid alignment.
3. **Evidence has a face.** IDs, hashes, counts, timestamps are monospace,
   muted, always present, and resolvable (click/copy). Any number may carry a
   provenance chip; the chip vocabulary is shared across surfaces.
4. **Unknown is a first-class visual state.** Never render blank or zero for
   unknown. Explicit "—" plus an `unreported`/`unknown` chip with the hatched
   treatment. This is the visual arm of the number-over-empty gates
   (polylogue-9e5.29) and text-derived provenance (9e5.30).
5. **The transcript is the hero object.** The reading surface is tuned like a
   book page: ~70–75ch measure for prose, speaker rhythm via a left rail
   (role glyph + hue), tool blocks collapsed to outcome-first headers.
6. **One token vocabulary, enforced.** `polylogue/ui/theme.py` is promoted
   from aspirational single-source-of-truth to the *generator* of every
   surface's tokens. No surface hand-writes hex values.
7. **Dark-first, light-supported.** Dark is the native mode of the audience;
   light mode is maintained through token pairs, never per-surface overrides.

## Current state (evidence, 2026-07-08)

- `ui/theme.py:1-6` claims "Single source of truth … used across CLI, HTML
  rendering, and daemon web surfaces" — but only
  `rendering/renderers/html.py:16` imports it.
- `daemon/web_shell.py:30-46` defines its own token set (`--bg: #070B10`,
  cyan accent family) — visually the *best* surface today: status dots,
  quality chips (`q-canonical`/`q-explicit`), instrument feel. Directionally
  right; promote it to the standard rather than replacing it.
- `devtools/pages_style.py:6-36` defines a third palette (`#0B0E14` family)
  with its own light mode.
- `rendering/templates/session.html:8-63` defines a fourth, with Inter and a
  gradient title.
- TUI uses stock `textual-dark`/`textual-light` (`ui/tui/app.py:55-59`).

Four-plus surfaces, four token vocabularies. The direction is convergence on
the web shell's instrument language, generated from `theme.py`.

## Three before/after decisions (per AC)

1. **Session list row.** Before: title + gray date. After: instrument row —
   `[origin glyph·hue] [title] [sort-key timestamp, provenance-styled]
   [token/cost chips with confidence treatment] [status dot]`. Timeless
   sessions show a hatched time chip, not an epoch date (composes with the
   temporal-correctness lane: cuxz `time_confidence`).
2. **Tool blocks in transcripts.** Before: full payload dump in flow. After:
   collapsed header row — `[tool glyph] [name] [target path, mono]
   [outcome chip: exit code / error] [duration]` — expandable for payload.
   Failures visible without expansion (reads `tool_result_is_error` /
   `exit_code`, the v16 keystone).
3. **Unknown metrics.** Before: `0` or blank. After: `—` + hatched
   `unreported` chip; tooltip names the absent source (maps to
   `FallbackReason` — the honesty machinery already carries the reason).
4. *(Bonus)* **session.html header.** Before: gradient title. After: flat
   accent title + provenance strip (origin, native id mono, short content
   hash, capture time).

## Scope

**In:** daemon web shell, canonical HTML renderer + template, pages theme,
TUI theme adoption, CLI role-glyph/color vocabulary.
**Out:** webui-v2 component library (owned by polylogue-bby.11 — this
document is an *input* to that stack's styling layer), logo/brand identity,
docs prose tone (3tl), marketing surfaces.

## Implementation beads

Filed as children/relations of polylogue-tjx1:

1. Design tokens generated from `theme.py`, consumed by all surfaces
   (keystone).
2. Provenance/honesty visual vocabulary — shared chip + glyph system.
3. Transcript reading-surface pass (measure, rhythm, tool-block collapse).
4. CLI/TUI alignment with the shared vocabulary.
