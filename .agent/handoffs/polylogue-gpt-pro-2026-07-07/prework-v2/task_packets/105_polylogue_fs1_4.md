# 105. polylogue-fs1.4 — Report: polylogue forensics for Hermes sessions

Priority/type/status: **P2 / feature / open**. Lane: **11-interoperability-origin**. Release: **K-interop-origin**. Readiness: **needs-acceptance-criteria**.

## What the bead says

Five-section per-session/per-corpus report, computed from the canonical archive (composition over existing primitives where possible): 1) session topology — parents, resumes, compactions, subagents, branches, long turns; 2) LLM/request economy — token lanes, cost, retry/fallback causes, model/provider shifts, cache-read amplification; 3) tool execution profile — durations, failures, approvals, repeated calls, parallel groups; 4) failure patterns — loops, stalls, empty-response retries, repeated shell failures, truncation, compaction-induced loss, reasoning burn; 5) local causal footprint — git diff/commits, commands, files, build/test runs. The 2-minute demo artifact (sanitized sessions, one command, README section) is the campaign-grade packaging of this report.

## Existing design note

Composition first: sections 1-3 and most of 4 should lower onto existing primitives — get_session_topology/logical session (topology), session_provider_usage_events + cost rollups (economy), actions/tool timing (tool profile), pathology detectors + structural outcomes (failure patterns), session_commits/git correlation (footprint). Only add new detectors where Hermes-specific (loop detection over repeated identical tool calls; stall = long gap between spans; reasoning burn = reasoning-token share per turn). Surface: a named read view/report profile (`polylogue forensics hermes --session <id>` or read --view forensics), rendered markdown + JSON. Demo packaging: sanitized fixture sessions, one command, <2min, README section — that packaging is a legitimate one-off; the five sections' facts must be query-composable (capabilities-not-silos rule).

## Acceptance criteria gap

This active bead lacks acceptance criteria in the export. Add checkable acceptance criteria before coding unless this packet explicitly supplies a temporary gate.

## Static mechanism / likely defect

Issue description localizes the mechanism: Five-section per-session/per-corpus report, computed from the canonical archive (composition over existing primitives where possible): 1) session topology — parents, resumes, compactions, subagents, branches, long turns; 2) LLM/request economy — token lanes, cost, retry/fallback causes, model/provider shifts, cache-read amplification; 3) tool execution profile — durations, failures, approvals, repeated calls, parallel groups; 4) failure patterns — loops, stalls, empty-response retries, repeated shell failures, tru… Design direction: Composition first: sections 1-3 and most of 4 should lower onto existing primitives — get_session_topology/logical session (topology), session_provider_usage_events + cost rollups (economy), actions/tool timing (tool profile), pathology detectors + structural outcomes (failure patterns), session_commits/git correlation (footprint). Only add new detectors where Hermes-specific (loop detection over repeated identical …

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

1. Composition first: sections 1-3 and most of 4 should lower onto existing primitives — get_session_topology/logical session (topology), session_provider_usage_events + cost rollups (economy), actions/tool timing (tool profile), pathology detectors + structural outcomes (failure patterns), session_commits/git correlation (footprint).
2. Only add new detectors where Hermes-specific (loop detection over repeated identical tool calls
3. stall = long gap between spans
4. reasoning burn = reasoning-token share per turn).
5. Surface: a named read view/report profile (`polylogue forensics hermes --session <id>` or read --view forensics), rendered markdown + JSON.
6. Demo packaging: sanitized fixture sessions, one command, <2min, README section — that packaging is a legitimate one-off
7. the five sections' facts must be query-composable (capabilities-not-silos rule).

## Tests to add

- OriginSpec fixture coverage: match, ambiguous, unknown, drift.
- Idempotent import/export or parse re-run test.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
