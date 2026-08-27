Summary

Add the complete 16-slice disposition matrix to the browser-capture redesign
README and state the two-layer ownership boundary.

Problem

The README documented the visual handoff and only three follow-up beads, while
the epic's current child set contains 16 slices. Without a matrix, closure
could not distinguish satisfied work from intentionally deferred or superseded
scope.

Solution

Record every child as satisfied, deferred to its named owner, or misframed with
the production evidence and shared-contract boundary that supports the
disposition. Keep the epic open while seven named child slices remain open.

Verification

- `npm test -- --reporter=dot tests/content/message_layer.test.js tests/content/ambient_surface.test.js tests/operator_status.test.js tests/popup.test.js tests/background.test.js tests/backfill.test.js tests/capture_jobs.test.js` — 7 files, 232 tests passed.
- `npm run lint` — passed.
- `npm run validate` — `manifest ok: .../browser-extension/manifest.json (v0.1.0)`.
- `npm run build` — `polylogue browser-extension build: version 0.3.0 (raw 0.3.0)`.
- `uv run python -m devtools verify doc-commands` — `84 doc files scanned, no stale commands`; `blocking=False`.
- `uv run python -m devtools verify atlas` — blocked by stale anchors inherited from commits after atlas verification commit `bb20b20d4266c47a0cb9cc8d63a39250c61810d6`.
- Full `npm test -- --reporter=dot` — 381 passed, 2 inherited asset-fixture expectation failures in `tests/content/chatgpt.test.js` and `tests/content/chatgpt_bridge.test.js`.

Residual risk

The matrix is documentation evidence, not implementation of the seven open
child slices. Real authenticated provider canaries and browser-backed visual
proof remain outside this lane.
