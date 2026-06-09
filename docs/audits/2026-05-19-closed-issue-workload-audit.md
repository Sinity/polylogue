# Closed-Issue Workload Audit — 2026-05-19

Ref: #1310

## Summary

Audit of ~110 closed issues from 2026-05-01 → 2026-05-18 (the last 17 days of
closures, capturing the full window since the prior governance reset in
#944/#970/#971/#972). Most closures are clean — the assured-close discipline
introduced after #1002/#1004 has clearly raised the floor. Decomposition
umbrellas (#994, #995, #1058, #998) close with explicit per-AC matrices and
verified successors. The 2026-05-01 mass-NOT_PLANNED batch (#626–#633) cleanly
folds into open composite owners (#614, #621, #623, #624). However, four cases
violate the close-discipline contract and one chain-fold leaks scope across
three issues.

### Counts

- Clean closures: ~95
- **PARTIAL** (closed with explicit unfinished AC, no successor filed): **3**
- **ABANDONED** (closed with "incremental / remaining / etc." and no `#NNN`): **2**
- **ORPHANED-FOLD** (scope folded through a chain to a target that doesn't cover it): **1** (one chain, one root)

---

## Findings

### PARTIAL — closed with unfinished acceptance and no successor

| # | Title | What fell through | Recommended remediation |
|---|-------|-------------------|-------------------------|
| **#1012** | Pre-existing test failures discovered during #1007 verification | The 2026-05-15 audit comment explicitly states "the issue is not closable yet" — `tests/unit/cli/test_command_aux_runtime.py::test_tags_command_plain_paths_cover_empty_hint_and_tabular_counts` remains XFAIL for provider-aware empty hint and Rich table renderer; stale xfails on the two XPASS rows were also flagged. Closed the next day by PR #1095 which addressed harness staleness only. The tags XFAIL is not in #1180 or any open issue. | File a successor issue: "fix(cli): provider-aware empty hint + Rich table for tags command (#1012 remainder)" — include the exact test node and the two stale xfails that should be removed. |
| **#1283** | audit(devtools): agent inner-loop tooling and iteration speed | Issue body explicitly enumerates Tier-1 recommendations A, B, C, D (each annotated "≤N LOC"). PR #1285 ships only **Tier-1 C** (`failure-context`). Tier-1 **A** (`use_when` + `examples` in `--help`), **B** (`verification-impact --paths` speculative mode), and **D** (`pipeline-probe` quiet-mode default) have no successor. Only Tier-2 E was carried over (to #1289). | File three small successor issues (or one umbrella) for Tier-1 A, B, D. All are explicitly bounded, named, and scoped. |
| **#802** | feat(acquisition): Claude Code + Codex hook integration for 100% session data coverage | Closing comment: "the paste-detection promise is not wired through to stored message/query semantics; leaving this issue closed and tracking reconciliation in #944." But #944 then closed with "no unique implementation scope remains here" (superseded by docs PR #1042). The hook → `has_paste` wiring is not in any open issue. #1199 mentions paste as an *attachment-surface concern* but does not own the UserPromptSubmit → message.has_paste pipeline AC. | Either reopen #802 or file a focused successor: "fix(pipeline): wire UserPromptSubmit hook events into message.has_paste facts (#802 remainder)". |

### ABANDONED — closed with classic abandonment phrases, no `#NNN`

| # | Title | Closing phrase | Recommended remediation |
|---|-------|----------------|-------------------------|
| **#723** | feat(api): formalize Python library API as the canonical Polylogue surface | "Closing: core implementation landed. **Remaining edges are incremental**." No issue number, no concrete scope handed off. The body AC said *every operation that touches the archive should have an API method* and *CLI and MCP tools call the same API methods rather than constructing queries independently* — both are still partial today (CLI/MCP still have local filter/query construction in several paths). | Either reopen, or file an audit issue: "audit(api): remaining CLI/MCP paths that bypass Polylogue API methods (#723 remainder)" with a concrete file inventory. |
| **#849** | feat(publication): replace run-stage render/site with daemon-owned publication | "Closing as misframed... Existing static-site code can be retained as devtools/manual archive export only if it still earns its keep; otherwise cleanup belongs under #805/#826-style structural maintenance." Cleanup is named but unfiled — #805 and #826 are both *already closed*. No open issue owns the "decide whether site/render are dead; retain or delete" decision. | Confirm whether the dead-or-retain decision is genuinely complete (in which case no action). If `polylogue site` / `render` are still in the tree without active scope, file a successor "chore(maintenance): retain-or-delete polylogue site/render commands". |

### ORPHANED-FOLD — scope leaked through a chain to a non-covering target

| # | Chain | What was lost | Recommended remediation |
|---|-------|---------------|-------------------------|
| **#809** | #809 → folded into #815 → folded into #817 (OPEN/REOPENED) | #809 body: `_SESSION_INSIGHT_REBUILD_PAGE_SIZE = 1` causes ~17,240 SQL queries on rebuild; "increase to 50-200." Also flagged per-call FTS readiness COUNT(*), pre-computing provider metrics from `conversation_stats`, lazy content block lookup. **None of these perf items appear in #817's body or AC.** #817 is now scoped to FTS bloat, stale read paths, and trigger safety. | Add a perf line item to #817 AC, OR file "perf(storage): session insight rebuild page size + per-call FTS readiness (#809 remainder)". |

---

## Notes on what did *not* trigger

- The 2026-05-01 NOT_PLANNED batch (#626–#633): every "Superseded by #N" target (#614/#618/#621/#623/#624) verified — all closed COMPLETED with proper close-out commentary.
- The 2026-05-04 NOT_PLANNED batch (#808/#812/#815/#816): folds verified. #808 → #828 (vector dimension AC present, completed via PR #975). #812 was a legitimate "not needed — single-user, fresh-on-mismatch" decision. #815 → #817 covers the FTS-repair-before-commit scope. **#816** (MCP `_safe_call` overloads, dead code, missing tool-name contracts) was folded into #811 — and #811's close-out comment says "MCP resource handler gaps should be tracked separately if still relevant." Marginal: the 6 items in #816 body are low-severity by author's own classification, so I did not flag this as ABANDONED, but it is the closest borderline case. Recommend a quick triage pass.
- The MK3 decomposition (#956 → #993 → #1199–#1205) and the session/cost decompositions (#994/#995 → #1129–#1140) all close with verified-existing children.
- The verifiability rollout (#1058) closes with an exhaustive child-PR matrix and names the remaining out-of-scope work with live issue numbers (#1019, #1022, #997, #845, #957, #999).

## Method

- `gh issue list --state closed --limit 200` (Sinity/polylogue), filtered to 2026-05-01+.
- Per-issue `gh issue view --json body,comments,closedByPullRequestsReferences`.
- Cross-referenced every "superseded by", "folded into", "tracking in", "subsumed by" target for state and scope coverage.
- Spot-grepped closing PRs for AC satisfaction on the highest-risk closures.

## Recommendation summary

| Action | Targets |
|--------|---------|
| File successor issue | #1012, #1283 (×3 or ×1 umbrella), #802, #723, #849 (conditional), #809 |
| Quick triage pass | #816 (borderline) |
| Accept close as-is | ~95 clean closures |

This audit does not itself reopen or file successor issues; remediation is left
to repo owners.
