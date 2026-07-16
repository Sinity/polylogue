# 171. polylogue-90y — In-page overlay: Polylogue presence on chat sites — archive state, context, assertion capture

Priority/type/status: **P2 / feature / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

The extension currently only EXTRACTS; it could also PRESENT. A tasteful injected surface on chatgpt.com/claude.ai (shadow-DOM isolated, keyboard-summonable, per-site opt-in) turns every chat page into a Polylogue-aware surface: is this chat archived and through when; what has this conversation cost; what does the archive already know that is relevant (judged assertions matching the current topic); and — the operator-flagged killer feature — CREATE and EDIT assertions directly from the page: select any passage -> save as note/claim/correction with an evidence ref pointing at that exact message. Memory capture at the moment of reading, where the thought occurs, instead of a separate tool later. This is also the first assertion-WRITE surface that meets the ambient criterion (jgp): zero invocation distance.

## Existing design note

TASTE CONSTRAINTS FIRST (operator: 'must feel non-crappy'): shadow-DOM component, zero layout shift on the host page (fixed corner chip + slide-over panel, never inline injection into the chat column); respects prefers-color-scheme; one keyboard chord (e.g. Alt+P) summons/dismisses; a per-site toggle and a global kill in the popup; NO badges on messages, NO buttons sprayed into the page — selection-triggered affordance appears only on text-selection (a small floating 'save to Polylogue' pill, the pattern users know from Medium/Hypothesis). (2) READ SURFACE: chip shows capture state (ties the reliability bead's per-tab truth); panel shows: session cost/tokens so far (archive knows), canonical archive link (open in workbench), and top-K relevant judged assertions retrieved via the daemon (semantic recall, mhx.4, when available; FTS fallback) — indices/refs, expandable, never a wall. (3) WRITE SURFACE: selection pill -> minimal editor (kind: note/claim/correction, body prefilled with selection, evidence ref = provider-native message anchor captured from DOM position -> resolved to archive message ref by the receiver); lands as candidate assertion (judgment gate unchanged); edit/withdraw own candidates from the panel list. (4) TRANSPORT: everything through the existing receiver channel to the daemon — the extension gains no new network surface; auth posture unchanged (loopback). (5) STAGING: read-only chip+panel first (ships value, zero write risk), selection-capture second, in-panel editing third. Dep: agent-write role machinery (27p) provides the assertion-write path the receiver calls.

## Acceptance criteria

On chatgpt.com and claude.ai: chip+panel render with zero host-page layout shift in light and dark themes; selection pill appears only on text selection; saving a selection creates a candidate assertion whose evidence ref resolves to the exact archived message; per-site toggle and global kill work; panel shows relevant judged assertions when embeddings are enabled.

## Static mechanism / likely defect

Issue description localizes the mechanism: The extension currently only EXTRACTS; it could also PRESENT. A tasteful injected surface on chatgpt.com/claude.ai (shadow-DOM isolated, keyboard-summonable, per-site opt-in) turns every chat page into a Polylogue-aware surface: is this chat archived and through when; what has this conversation cost; what does the archive already know that is relevant (judged assertions matching the current topic); and — the operator-flagged killer feature — CREATE and EDIT assertions directly from the page: select any passage -> … Design direction: TASTE CONSTRAINTS FIRST (operator: 'must feel non-crappy'): shadow-DOM component, zero layout shift on the host page (fixed corner chip + slide-over panel, never inline injection into the chat column); respects prefers-color-scheme; one keyboard chord (e.g. Alt+P) summons/dismisses; a per-site toggle and a global kill in the popup; NO badges on messages, NO buttons sprayed into the page — selection-triggered afforda…

## Source anchors to inspect first

- No precise source anchor was localized in this static pass. Start from the bead description and repository search.

## Implementation plan

1. TASTE CONSTRAINTS FIRST (operator: 'must feel non-crappy'): shadow-DOM component, zero layout shift on the host page (fixed corner chip + slide-over panel, never inline injection into the chat column)
2. respects prefers-color-scheme
3. one keyboard chord (e.g.
4. Alt+P) summons/dismisses
5. a per-site toggle and a global kill in the popup
6. NO badges on messages, NO buttons sprayed into the page — selection-triggered affordance appears only on text-selection (a small floating 'save to Polylogue' pill, the pattern users know from Medium/Hypothesis).
7. (2) READ SURFACE: chip shows capture state (ties the reliability bead's per-tab truth)

## Tests to add

- Acceptance proof: On chatgpt.com and claude.ai: chip+panel render with zero host-page layout shift in light and dark themes
- Acceptance proof: selection pill appears only on text selection
- Acceptance proof: saving a selection creates a candidate assertion whose evidence ref resolves to the exact archived message
- Acceptance proof: per-site toggle and global kill work
- Acceptance proof: panel shows relevant judged assertions when embeddings are enabled.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
