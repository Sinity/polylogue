# Browser-capture extension redesign (Claude Design output, 2026-07-09)

**Status: resolved.** A follow-up pass grounded in real authenticated
ChatGPT/Claude.ai screenshots resolved the one open question the first pass
left (F2/F3 vs. F4 placement, see below). `mockup.dc.html` is the current
(second-pass) version; `f2-fixed-verification.png` is a rendered check of
the resolved layers against ChatGPT's real layout. The original brief that
produced the first pass is reproduced in full in each bead's design notes
(`polylogue-90y`, `polylogue-3v1`, `polylogue-ptx`).

Source: operator-commissioned redesign via the "Claude Design" tool, seeded
with a comprehensive context brief covering the extension's architecture,
existing internal design work (`polylogue-90y`, `polylogue-3v1`,
`polylogue-ptx`), and the operator's own taste constraints. Full brief that
produced this: `.agent/scratch/` history is not committed, but the brief's
content is reproduced in each bead's design notes below.

- `mockup.dc.html` (+ `support.js`) — the rendered design canvas. Open in a
  browser to view; six labeled frames (F1-F6). Fully synthetic mockup
  content, no real user data.
- Reference screenshots of the real, authenticated ChatGPT/Claude.ai UI
  (dark mode, home + open-conversation views) were captured to ground a
  follow-up pass reconciling the in-page placement strategies against the
  real host layouts. **Not committed here** — they contain real chat
  titles/message content from an authenticated session and this repo is
  public. Delivered directly to the operator instead; kept local only.

## The six frames

- **F1 — Popup mission control**: multi-tab list (per-tab status: safe /
  catching-up / partial-fidelity / not-saved), an active-conversation detail
  panel (cost/tokens/captured-count), a **"What Polylogue did here" event
  timeline** (the core fix for the diagnosed silent-failure bug — every
  decision, including "held auto-capture, retried automatically," becomes a
  visible logged event, not silence), an **Agent control** section for the
  reverse/posting channel (off by default, "OFF — safe" badge, and a new
  safety idea beyond the original brief: every posted command lands as a
  **dry-run draft first, never an auto-send**), and a provider-coverage
  footer (ChatGPT/Claude/Grok checked, Gemini shown as not-yet-supported).
- **F2 — Ambient in-page chip**: bottom-right fixed pill, shadow-DOM,
  zero layout shift, shows state + running cost + the `⌥P` keyboard hint.
- **F3 — Slide-over deep-dive**: triggered from the chip; shows archive
  state, cost/token strip, and **"the archive already knows"** — relevant
  judged assertions with a kind badge (claim/correction), a ref id, and a
  match-confidence percentage; actions to recall more or open the archive.
- **F4 — Native-blended inline**: an alternative in-page placement — a small
  per-message "capture dot" in a gutter beside each assistant message, a
  "Save to Polylogue" action woven into the host's own per-message action
  row (alongside copy/share), and a Polylogue status line docked above the
  composer ("archived through this message... auto-captures on reply").
  **This is presented as an alternative to F2/F3, not reconciled with it —
  see the follow-up brief below.**
- **F5 — Selection → assertion (the killer feature)**: select any passage on
  the host page → a floating "Save to Polylogue" pill appears → an editor
  with a segmented kind picker (Claim/Note/Correction), the selection
  prefilled as the body, and the evidence ref auto-attached and visibly
  confirmed before saving.
- **F6 — States gallery**: four calm, specific state cards (receiver asleep
  = explicitly *not* an error; partial fidelity with a re-capture action;
  failed with the actual reason + Fix/Retry actions; stale/new-messages with
  a capture-latest action) plus a banner making the design's thesis explicit:
  *"Silent failure is designed out... doing nothing is itself a logged,
  visible event."*

Visual language: dark-first, IBM Plex Sans/Mono, violet (`#8b7bf2`) accent —
deliberately distinct from ChatGPT's and Claude.ai's own palettes (confirmed
against the real screenshots, not just asserted).

## Resolved: F2/F3 vs. F4 placement

The first pass presented F2/F3 (shadow-DOM chip + slide-over) and F4
(blended into the host's per-message action row) as parallel alternatives.
A follow-up pass, grounded in real authenticated ChatGPT/Claude.ai
screenshots (kept local, not committed), resolved this: **they aren't
alternatives, they're a two-layer split with one boundary rule** —

> Per-message state blends in. Cross-conversation intelligence floats.

- **Layer 1 — ambient, blended (F4)**: a capture-status dot + "save to
  Polylogue" action woven into the host's *existing* per-message action row
  (copy/feedback/regenerate), matched to its exact icon size (~30px),
  ghost/hover style, and placement — both ChatGPT and Claude.ai already have
  this pattern, this extends it rather than inventing something new. Always
  present, answers "is *this* captured?" without a click.
- **Layer 2 — deep-dive, separate (F2/F3)**: the corner chip (⌥P) + 360px
  slide-over, holding everything with *no host equivalent* — session cost,
  archive recall, relevant judged assertions, the "what Polylogue did"
  timeline. Deliberately its own surface; inventing host-blended UI for
  this would read as foreign.

`f2-fixed-verification.png` shows both layers rendered together against
ChatGPT's real sidebar/composer/message-action-row layout (sanitized
content, not real chat data) — confirming no visual collision with what the
host already docks near the composer.

Recorded verbatim on `polylogue-90y`'s design notes and folded into its
acceptance criteria.

## Beads updated with this design as concrete implementation input

- `polylogue-90y` — F2/F3/F4/F5 (in-page overlay + selection→assertion)
- `polylogue-3v1` — F1/F6 (popup mission control, timeline, states gallery)
- `polylogue-ptx` — F1's dry-run-draft safety refinement for the reverse channel

## Child disposition matrix

The epic is not closeable while a row marked deferred remains open. Satisfied
rows point to the shared production contract and its focused verification;
deferred rows retain the named bead that owns the missing slice.

| Slice | Disposition | Evidence or owner |
| --- | --- | --- |
| `polylogue-r2kb` — operator status vocabulary | Satisfied | `browser-extension/src/operator_status.js`; `browser-extension/tests/operator_status.test.js` |
| `polylogue-l40k` — multi-tab and offline spool | Satisfied | `browser-extension/src/popup.js` and `browser-extension/src/background.js`; popup/background tests |
| `polylogue-4g3n` — conversation timeline | Satisfied | Shared event presentation and durable capture-job event projection; closed as superseded by `polylogue-06zm` for remaining archive work |
| `polylogue-bkff` — popup mission control | Satisfied | Popup multi-tab and active-conversation rendering; closed by the merged `#2780` implementation |
| `polylogue-yyvg.7` — automatic exception-driven UX | Satisfied | Shared popup/ambient/message presentation and diagnostics; closed by merged `#4210` |
| `polylogue-r4no` — automatic capture trigger | Satisfied | Capture-or-held-decision path and timeline event coverage |
| `polylogue-ys30` — Layer 1 | Satisfied | `browser-extension/src/content/message_layer.js`; `browser-extension/tests/content/message_layer.test.js` |
| `polylogue-yyvg.4` — provider identity resolver | Satisfied | Typed ChatGPT/Claude identity observations and receiver-resolved acknowledgements; closed by merged `#4254` |
| `polylogue-yyvg.5` — Sol Pro work packages | Misframed | Generic browser actions belong to `polylogue-ptx`; campaign orchestration belongs to `polylogue-yyvg.6` |
| `polylogue-bj5h` — selection to assertion | Deferred to `polylogue-bj5h` | Requires the shared identity contract and user-tier candidate write route |
| `polylogue-wvji` — Layer 2 corner chip and slide-over | Deferred to `polylogue-wvji` | The ambient shell exists; the remaining deep-dive slice is still open |
| `polylogue-yqof` — reverse-channel popup | Deferred to `polylogue-yqof` | Keep the existing receiver posting gates and dry-run contract as prerequisites |
| `polylogue-yyvg.1` — reverse organization | Deferred to `polylogue-yyvg.1` | Provider mutation organization remains a separate plan/authorize/apply/receipt slice |
| `polylogue-yyvg.2` — provider collections | Deferred to `polylogue-yyvg.2` | Collection identity and membership history are not yet modeled |
| `polylogue-yyvg.3` — archive-context intelligence | Deferred to `polylogue-yyvg.3` | Cross-conversation intelligence remains Layer 2 work and must use the floating surface |
| `polylogue-yyvg.6` — external campaign incorporation | Deferred to `polylogue-yyvg.6` | Campaign ledger and orchestration stay outside extension and receiver transport |

The matrix is also the boundary for the two-layer rule: `ys30` owns blended
per-message state, while `wvji` and `yyvg.3` own floating cross-conversation
intelligence. Capture, timeline, popup, assertion, and reverse-channel work
consume the shared receiver identity and status contracts rather than creating
parallel ledgers.
