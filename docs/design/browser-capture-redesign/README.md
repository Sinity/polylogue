# Browser-capture extension redesign (Claude Design output, 2026-07-09)

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

## Open question this pack doesn't resolve

F2/F3 (fully separate shadow-DOM chip + slide-over) and F4 (blended into the
host's existing per-message action row) are two different placement
strategies for overlapping capability, presented in parallel rather than
reconciled. A follow-up brief requesting a single recommended direction
(or an explicit division of responsibility between the two) has been
prepared, grounded in real host screenshots (kept local, not committed —
see above) — see the bead notes on `polylogue-90y` for the exact follow-up
prompt.

## Beads updated with this design as concrete implementation input

- `polylogue-90y` — F2/F3/F4/F5 (in-page overlay + selection→assertion)
- `polylogue-3v1` — F1/F6 (popup mission control, timeline, states gallery)
- `polylogue-ptx` — F1's dry-run-draft safety refinement for the reverse channel
