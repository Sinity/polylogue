# 101. polylogue-da1 — Provider format-drift sentinel: detect upstream export-shape changes from live ingest

Priority/type/status: **P2 / feature / open**. Lane: **11-interoperability-origin**. Release: **K-interop-origin**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Claude Code, Codex, ChatGPT, and Gemini change their export/JSONL shapes without notice. Today drift surfaces as silent parse degradation — dropped fields or nodes discovered manually weeks later (e.g. the ChatGPT asset-only-node and Antigravity non-UTF-8 drops in polylogue-qda). Nothing watches for new-unseen-shape rates in live ingest. A sentinel turns format drift from a forensic discovery into a daemon health signal.

## Existing design note

Reuse the existing schema-inference machinery (schemas/ shape signatures) rather than building a new detector: at ingest, count records whose shape does not match the committed provider schema package, keyed by (origin, element kind, unseen-key signature), into ops.db telemetry (the ops tier explicitly allows additive columns — no schema bump). Daemon health check + 'polylogue ops status' line: 'origin X: N% of records since <date> carry unseen shapes', with bounded example native_ids. The follow-up action stays 'devtools lab schema generate/promote' — the sentinel only detects and points. Pitfalls: (1) never fail or skip ingest on drift — raw payloads are stored, so parsing is always redoable after a parser update; that is the payoff of the fresh-first doctrine. (2) distinguish 'new optional payload field' (benign, common) from 'known field disappeared / type changed' (parser regression risk) — alert thresholds should differ. (3) rate must be windowed since-date, not lifetime, or old archives dilute the signal.

## Acceptance criteria

- At ingest, records whose shape does not match the committed provider schema package are counted, keyed by (origin, element kind, unseen-key signature), into ops.db telemetry via additive columns only (no schema bump — the ops tier explicitly allows additive columns).
- Ingest never fails or skips on drift: a drift-shaped record still ingests and its raw payload is stored (parsing is redoable after a parser update) — verified by a test feeding an unseen-shape record.
- The detector distinguishes benign 'new optional payload field' from risky 'known field disappeared / type changed', with different alert thresholds.
- The rate is windowed since a date (not lifetime) so old archives do not dilute the signal.
- A daemon health check + `polylogue ops status` line reads 'origin X: N% of records since <date> carry unseen shapes' with bounded example native_ids; the follow-up action points at `devtools lab schema generate/promote` (the sentinel only detects).
- Sentinel alerting does not depend on the daemon operating healthily — a last-resort status-line marker is visible on any `polylogue` invocation.
- `devtools test <sentinel tests>` green.

## Static mechanism / likely defect

Issue description localizes the mechanism: Claude Code, Codex, ChatGPT, and Gemini change their export/JSONL shapes without notice. Today drift surfaces as silent parse degradation — dropped fields or nodes discovered manually weeks later (e.g. the ChatGPT asset-only-node and Antigravity non-UTF-8 drops in polylogue-qda). Nothing watches for new-unseen-shape rates in live ingest. A sentinel turns format drift from a forensic discovery into a daemon health signal. Design direction: Reuse the existing schema-inference machinery (schemas/ shape signatures) rather than building a new detector: at ingest, count records whose shape does not match the committed provider schema package, keyed by (origin, element kind, unseen-key signature), into ops.db telemetry (the ops tier explicitly allows additive columns — no schema bump). Daemon health check + 'polylogue ops status' line: 'origin X: N% of reco…

## Source anchors to inspect first

- `polylogue/sources/dispatch.py` — Current origin/source dispatch logic; target for OriginSpec consolidation.
- `polylogue/sources/import_preflight.py` — Preflight/readiness should report origin strictness and ambiguity.
- `polylogue/sources/provider_completeness.py` — Provider completeness is adjacent to OriginSpec readiness.
- `polylogue/sources/parsers/base.py` — Parser base contracts should be folded into OriginSpec.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.

## Implementation plan

1. Reuse the existing schema-inference machinery (schemas/ shape signatures) rather than building a new detector: at ingest, count records whose shape does not match the committed provider schema package, keyed by (origin, element kind, unseen-key signature), into ops.db telemetry (the ops tier explicitly allows additive columns — no schema bump).
2. Daemon health check + 'polylogue ops status' line: 'origin X: N% of records since <date> carry unseen shapes', with bounded example native_ids.
3. The follow-up action stays 'devtools lab schema generate/promote' — the sentinel only detects and points.
4. Pitfalls: (1) never fail or skip ingest on drift — raw payloads are stored, so parsing is always redoable after a parser update
5. that is the payoff of the fresh-first doctrine.
6. (2) distinguish 'new optional payload field' (benign, common) from 'known field disappeared / type changed' (parser regression risk) — alert thresholds should differ.
7. (3) rate must be windowed since-date, not lifetime, or old archives dilute the signal.

## Tests to add

- Acceptance proof: At ingest, records whose shape does not match the committed provider schema package are counted, keyed by (origin, element kind, unseen-key signature), into ops.db telemetry via additive columns only (no schema bump — the ops tier explicitly allows additive columns).
- Acceptance proof: Ingest never fails or skips on drift: a drift-shaped record still ingests and its raw payload is stored (parsing is redoable after a parser update) — verified by a test feeding an unseen-shape record.
- Acceptance proof: The detector distinguishes benign 'new optional payload field' from risky 'known field disappeared / type changed', with different alert thresholds.
- Acceptance proof: The rate is windowed since a date (not lifetime) so old archives do not dilute the signal.
- Acceptance proof: A daemon health check + `polylogue ops status` line reads 'origin X: N% of records since <date> carry unseen shapes' with bounded example native_ids
- Acceptance proof: the follow-up action points at `devtools lab schema generate/promote` (the sentinel only detects).
- Acceptance proof: Sentinel alerting does not depend on the daemon operating healthily — a last-resort status-line marker is visible on any `polylogue` invocation.
- Acceptance proof: `devtools test <sentinel tests>` green.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
