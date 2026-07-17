# 030. polylogue-cpf.4 — Enforce degrade-loudly: sweep silent soft-failure paths to carry a signal

Priority/type/status: **P2 / task / open**. Lane: **00-trust-floor**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

The cpf degraded-mode doctrine says "degrade loudly once", but a deep read (2026-07-05) found the codebase systematically degrades SILENTLY on derived/fallback/probe paths — robust (never crashes) but serves incomplete/stale data with no signal, which for a system-of-record is a construct-validity hole. Concrete instances found: convergence freshness probes fail-closed to converged with no log (1xc.11); lineage composition truncates on depth>64 or dangling branch point with no completeness signal (4ts.6); coordination archive-evidence returns empty tuples on a 0.2s SQLite timeout (envelope.py:610/616/639) — indistinguishable from "no evidence"; generic-messages parser drops timestamps silently (tf0e). This bead is the CLASS: audit derived-read, fallback, and freshness-probe paths for silent-vs-signaled degradation, and make each carry a typed degradation signal (reason + provenance/confidence) OR log-loudly-once, per the doctrine. Deliverable: a checklist of soft-fail sites + the signal each now emits; a lint or review-gate so new silent soft-fails are caught (composes with the standing hygiene lint 8jg9.1).

## Acceptance criteria

Each identified silent soft-fail path (probe fail-closed, lineage truncation, timeout-to-empty, fallback data-drop) either emits a typed degradation signal consumers can read, or logs-loudly-once; a reader/agent can distinguish "no data" from "degraded/timed-out/truncated". A review-gate or lint flags new bare soft-fails. Verify: the instance beads (1xc.11, 4ts.6, tf0e) close against this, and a test asserts the timeout/truncation/probe-fail paths surface a reason.

## Static mechanism / likely defect

Design direction: The cpf degraded-mode doctrine says "degrade loudly once", but a deep read (2026-07-05) found the codebase systematically degrades SILENTLY on derived/fallback/probe paths — robust (never crashes) but serves incomplete/stale data with no signal, which for a system-of-record is a construct-validity hole. Concrete instances found: convergence freshness probes fail-closed to converged with no log (1xc.11); lineage comp…

## Source anchors to inspect first

- `polylogue/insights/audit.py:173` — build_insight_rigor_audit_report is the audit entry point.
- `polylogue/insights/audit.py:194` — Current code iterates list_rigor_contracts, not the product registry.
- `polylogue/insights/audit.py:216` — Registry lookup is secondary and skipped for products without contracts.
- `polylogue/insights/rigor.py:85` — _RIGOR_MATRIX declares only a subset of registered products.
- `polylogue/insights/registry.py:294` — INSIGHT_REGISTRY is the universe the audit should iterate.
- `polylogue/archive/actions/followup.py` — Action/followup classification is a real structural analytics input.
- `polylogue/archive/actions/fields.py` — Action fields determine what can be measured without prose heuristics.
- `polylogue/insights/registry.py:294` — Insight registry should become measure/product registry input.
- `scripts/agent_forensics.py` — Existing forensics script is a proof artifact and candidate product surface.
- `polylogue/core/dates.py:10` — parse_date has no injected clock parameter.
- `polylogue/core/dates.py:37` — RELATIVE_BASE uses ambient datetime.now.
- `polylogue/archive/query/expression.py:2440` — Query grammar recognizes relative-date literals.
- `polylogue/archive/query/spec.py:498` — SessionQuerySpec.from_params is the central query-spec constructor.
- `polylogue/insights/temporal_source.py:66` — classify_profile_hwm_source promotes any updated_at to provider_ts.
- `polylogue/insights/temporal_source.py:97` — classify_aggregate_hwm_source currently collapses all non-empty source updates to provider_ts.

## Implementation plan

1. The cpf degraded-mode doctrine says "degrade loudly once", but a deep read (2026-07-05) found the codebase systematically degrades SILENTLY on derived/fallback/probe paths — robust (never crashes) but serves incomplete/stale data with no signal, which for a system-of-record is a construct-validity hole.
2. Concrete instances found: convergence freshness probes fail-closed to converged with no log (1xc.11)
3. lineage composition truncates on depth>64 or dangling branch point with no completeness signal (4ts.6)
4. coordination archive-evidence returns empty tuples on a 0.2s SQLite timeout (envelope.py:610/616/639) — indistinguishable from "no evidence"
5. generic-messages parser drops timestamps silently (tf0e).
6. This bead is the CLASS: audit derived-read, fallback, and freshness-probe paths for silent-vs-signaled degradation, and make each carry a typed degradation signal (reason + provenance/confidence) OR log-loudly-once, per the doctrine.
7. Deliverable: a checklist of soft-fail sites + the signal each now emits

## Tests to add

- Acceptance proof: Each identified silent soft-fail path (probe fail-closed, lineage truncation, timeout-to-empty, fallback data-drop) either emits a typed degradation signal consumers can read, or logs-loudly-once
- Acceptance proof: a reader/agent can distinguish "no data" from "degraded/timed-out/truncated".
- Acceptance proof: A review-gate or lint flags new bare soft-fails.
- Acceptance proof: Verify: the instance beads (1xc.11, 4ts.6, tf0e) close against this, and a test asserts the timeout/truncation/probe-fail paths surface a reason.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
