# 131. polylogue-27m — Excision and secret hygiene: the archive can forget on purpose

Priority/type/status: **P2 / task / open**. Lane: **00-trust-floor**. Release: **A-trust-floor**. Readiness: **blocked-hard**.

Hard blockers: polylogue-b5l

## What the bead says

Keep-everything is doctrine; 'cannot remove that API key I pasted in 2025' is a bug in that doctrine's shadow. Two halves: (1) EXCISION — a supported, auditable operation that provably removes one specific piece of content from source rows, index rows, FTS, embeddings, blobs, and derived models, leaving a tombstone recording that something was excised (not what); hard because content-addressing and idempotency assume immutability. (2) SECRET SCANNING at ingest — sessions demonstrably contain pasted credentials; detect (entropy + known patterns, gitleaks-class ruleset), flag as candidates in the judgment queue (never auto-redact — judgment gate), and offer excision on acceptance.

## Existing design note

Excision mechanics: source.db rows get a redacted_at + reason tombstone with the payload replaced by a hash-of-removed marker (content hash boundary: the session's content_hash is recomputed and the old hash recorded on the tombstone so idempotency does not resurrect it); derived tiers rebuild the affected session (blue-green machinery makes this cheap); blobs: reference-counted delete with GC-lease discipline; embeddings: delete rows by message ref. The operation is a single 'polylogue ops excise <ref>' with a mandatory reason, a dry-run diff, and an ops.db audit row. Scanning: an ingest-side detector emitting secret_candidate assertions (kind exists check — registration traps) with the span ref; the judge queue surfaces them grouped; accepted -> excise flow. Retro pass command over the existing corpus, rate-limited. PITFALL: never log the secret itself anywhere in the flow, including the candidate body — store span coordinates, not content.

## Acceptance criteria

Excising a seeded session's message removes it from every tier (verified by grep across source/index/FTS/embeddings/blob refs) and leaves the tombstone; re-ingesting the original source does not resurrect it; the secret scanner flags a seeded fake credential as a candidate without logging its value; retro scan on the live archive produces a bounded candidate list.

## Static mechanism / likely defect

Issue description localizes the mechanism: Keep-everything is doctrine; 'cannot remove that API key I pasted in 2025' is a bug in that doctrine's shadow. Two halves: (1) EXCISION — a supported, auditable operation that provably removes one specific piece of content from source rows, index rows, FTS, embeddings, blobs, and derived models, leaving a tombstone recording that something was excised (not what); hard because content-addressing and idempotency assume immutability. (2) SECRET SCANNING at ingest — sessions demonstrably contain pasted credentials; de… Design direction: Excision mechanics: source.db rows get a redacted_at + reason tombstone with the payload replaced by a hash-of-removed marker (content hash boundary: the session's content_hash is recomputed and the old hash recorded on the tombstone so idempotency does not resurrect it); derived tiers rebuild the affected session (blue-green machinery makes this cheap); blobs: reference-counted delete with GC-lease discipline; embe…

## Source anchors to inspect first

- `polylogue/daemon/http.py:983` — _check_auth_logic uses direct equality and allows all when token is unset.
- `polylogue/daemon/http.py:1037` — _check_auth currently accepts query-string access_token broadly.
- `polylogue/daemon/http.py:1294` — do_GET dispatches without central Host/Origin admission.
- `polylogue/daemon/http.py:1301` — _check_cross_origin applies only to POST and allows absent Origin.
- `polylogue/browser_capture/receiver.py:45` — BrowserCaptureReceiverConfig defaults auth_token to None.
- `polylogue/browser_capture/server.py:54` — _origin_allowed accepts absent Origin.
- `polylogue/browser_capture/server.py:68` — _check_token accepts every request when auth_token is None and uses direct equality otherwise.
- `polylogue/browser_capture/server.py:47` — Only per-request max body exists; add spool file/count/bytes governor.

## Implementation plan

1. Excision mechanics: source.db rows get a redacted_at + reason tombstone with the payload replaced by a hash-of-removed marker (content hash boundary: the session's content_hash is recomputed and the old hash recorded on the tombstone so idempotency does not resurrect it)
2. derived tiers rebuild the affected session (blue-green machinery makes this cheap)
3. blobs: reference-counted delete with GC-lease discipline
4. embeddings: delete rows by message ref.
5. The operation is a single 'polylogue ops excise <ref>' with a mandatory reason, a dry-run diff, and an ops.db audit row.
6. Scanning: an ingest-side detector emitting secret_candidate assertions (kind exists check — registration traps) with the span ref
7. the judge queue surfaces them grouped

## Tests to add

- Acceptance proof: Excising a seeded session's message removes it from every tier (verified by grep across source/index/FTS/embeddings/blob refs) and leaves the tombstone
- Acceptance proof: re-ingesting the original source does not resurrect it
- Acceptance proof: the secret scanner flags a seeded fake credential as a candidate without logging its value
- Acceptance proof: retro scan on the live archive produces a bounded candidate list.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
