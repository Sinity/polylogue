# Migration decisions, parity-golden design, and normalization rules

## D1 — Current source is a 104-tool baseline

Freeze `536a53efac0cbe4a2473ad379e4db49ef3fce74d` as the old-surface anchor for this revision. Capture all 104 names even though the original mission said 103. Store the anchor SHA, generated equivalence artifact SHA-256, Python version, dependency lock hash, archive fixture manifest hash, and normalization profile version in every golden run manifest.

## D2 — Six default reads; graph is a `get` view

The target read discovery set is exactly `query/read/get/explain/context/status`. Recursive lineage uses:

```json
{"ref":"session:<id>","view":"graph.topology|graph.tree|graph.logical","direction":"ancestors|descendants|both","page":{"limit":N,"cursor":"..."}}
```

The response reports nodes, edges, current frontier, `next_cursor`, cycle detection, and whether the logical traversal is complete. This is not a semantic cap: physical pages are bounded; logical enumeration continues.

## D3 — Privileged role algebra and authority

`write` owns declaration-backed state operations, including read/list operations whose current authority is write-scoped (`get_metadata`, annotations, marks, saved views, recall packs, tags, workspaces, corrections). `judge` owns candidate decisions. `run` executes saved query/recipe refs and gains no generic instruction authority. `maintenance` owns preview/authorize/execute/status/reconcile.

Role inheritance is exact: read 6, write 8, review 9, admin 10. Prompts/resources cannot bypass those gates. `unacknowledged_failures` is raised to write role unless its acknowledgment check is redesigned as an explicitly read-authorized projection.

## D4 — Disposition policy

This revision recommends **58 parity-golden** and **46 mapping-only** rows; no row is retired without a successor.

- Parity-golden: observed in committed usage evidence; absent from that older census; or continuity/incident/destructive/authority critical.
- Mapping-only: zero in the committed census and ordinary, subject to mandatory live 90-day recheck.
- `archive_list_sessions` and `archive_search_sessions`: mapping-only regardless of historical table because `.2.1` already records zero use and exhaustive filter parity.
- A live call promotes a row. A zero call never demotes review/admin or explicitly critical rows.

Parity-golden rows:

```text
action_affordances, add_tag, agent_coordination, archive_coverage, archive_debt, blackboard_list,
blackboard_post, capture_assertion_candidate, clear_corrections, compile_context, compose_context_preamble,
cost_outlook, cost_rollups, delete_session, explain_query_expression, facets, find_abandoned_sessions,
find_resume_candidates, find_stuck_sessions, get_context_delivery, get_messages, get_metadata,
get_pathologies, get_resume_brief, get_session_summary, get_session_topology, get_session_tree, get_stats_by,
import_annotation_batch, join_typed_annotations, judge_assertion_candidate, judge_assertion_candidates,
list_assertion_candidate_reviews, list_assertion_candidates, list_sessions, list_tags, maintenance_execute,
maintenance_list, maintenance_preview, maintenance_status, named_source_freshness, neighbor_candidates,
query_units, readiness_check, rebuild_index, rebuild_session_insights, record_correction, resolve_ref, search,
session_costs, session_latency_profile, session_profiles, session_tag_rollups, session_work_events, stats,
tool_usage, update_index, workflow_shape_distribution
```

Mapping-only rows:

```text
add_mark, aggregate_sessions, archive_get_session, archive_list_sessions, archive_search_sessions,
build_context_image, bulk_tag_sessions, compare_sessions, correlate_session, correlate_sessions,
delete_annotation, delete_metadata, delete_recall_pack, delete_saved_view, delete_workspace,
embedding_preflight, embedding_status, explain_import, find_similar_sessions, get_logical_session,
get_postmortem_bundle, insight_rigor_audit, list_annotations, list_assertion_claims, list_corrections,
list_marks, list_read_view_profiles, list_recall_packs, list_saved_views, list_workspaces, provider_usage,
query_completions, raw_artifacts, remove_mark, remove_tag, save_annotation, save_recall_pack, save_saved_view,
save_workspace, session_phases, session_profile, session_tool_timing, set_metadata, threads,
tool_call_latency_distribution, usage_timeline
```

## D5 — Canonical response contract

```json
{
  "ok": true,
  "data": {},
  "items": [],
  "page": {
    "limit": 100,
    "cursor": null,
    "next_cursor": null,
    "total": null,
    "exact": null,
    "semantics": "single_object|exhaustive_page|top_k|sample|aggregate|bounded_context|recursive_graph"
  },
  "result_ref": "result-set:<stable-id>",
  "meta": {
    "schema_version": 1,
    "archive_epoch": "...",
    "coverage": {},
    "degraded": []
  },
  "receipt": null
}
```

Omit inapplicable slots or keep them null consistently; do not create per-verb envelope dialects. Top-k `total` is null or a truthful candidate count with `exact=false`; it is never presented as exhaustive membership. Errors use a typed envelope with stable code, retryability, field/path, authority requirement, and recovery hint; prose is secondary.

## D6 — Fixture archive requirements

The parity fixture is source-controlled as deterministic source records plus a builder, not as an opaque copied live archive. It must contain:

- multiple providers and at least two repositories;
- exact and ambiguous session prefixes;
- >2 physical pages for sessions/messages/actions/files/annotations and at least one ranked tie;
- parent/child/sibling lineage, a deep chain, and malformed/cycle evidence handled without infinite traversal;
- text, fenced code, tool actions/results, failures, paths, tags, marks, metadata, annotations, saved queries, recall packs, workspaces, corrections, assertion candidates/reviews/claims;
- accepted/rejected/deferred/superseded judgment states and partial batch success;
- cost/usage and timing evidence with known assumptions;
- embedding ready/unavailable/degraded cases;
- raw import revisions, authority refs, exact-source freshness stages, and a missing source;
- maintenance preview/apply/status/retry/reconcile fixtures;
- Unicode, empty/null fields, timezone boundaries, deterministic duplicate rows, and permission-denied cases;
- a frozen clock, deterministic IDs, stable ordering/tie-breakers, and explicit archive epoch.

Recommended layout:

```text
tests/fixtures/mcp_parity/v1/
  manifest.json                 # fixture version, builder inputs, clock, IDs, hashes
  source/                       # deterministic parsed/raw/user/ops seed material
  cases.jsonl                   # old call, target call, role, semantics, normalization profile
  expected-state/               # public before/after state for governed operations
  README.md

tests/golden/mcp_parity/v1/
  OLD_SURFACE_SHA
  capture-manifest.json
  normalization.json
  old/<tool>/<case>.json
  comparisons/<tool>/<case>.json
```

Do not commit production data, a copied live archive, or SQLite files containing nondeterministic page layout. The builder creates tier DBs in a temporary archive and records public content hashes.

## D7 — Required harness and exact capture invocation

Add a repository command `devtools mcp-parity` implemented in `devtools/mcp_parity.py` and registered in `devtools/command_catalog.py`. It is not present in the snapshot; this is the required interface. Subcommands share one library used by tests:

```text
devtools mcp-parity build-fixture
devtools mcp-parity census
devtools mcp-parity capture-old
devtools mcp-parity compare
devtools mcp-parity shadow-report
```

The harness is added on the transition checkout while all legacy handlers still exist. Do **not** execute the command from the detached old worktree: commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d` predates `devtools mcp-parity`. Run the orchestrator from the transition checkout and make it launch an isolated worker against the old worktree:

```bash
set -euo pipefail
ROOT_DIR="$PWD"
OLD_SHA="536a53efac0cbe4a2473ad379e4db49ef3fce74d"
OLD_TREE="$ROOT_DIR/.cache/mcp-parity/old"
ARCHIVE_ROOT="$ROOT_DIR/.cache/mcp-parity/archive-v1"

mkdir -p tests/golden/mcp_parity/v1
test ! -e "$OLD_TREE"
git worktree add --detach "$OLD_TREE" "$OLD_SHA"
test "$(git -C "$OLD_TREE" rev-parse HEAD)" = "$OLD_SHA"
test -z "$(git -C "$OLD_TREE" status --porcelain)"
printf '%s\n' "$OLD_SHA" > tests/golden/mcp_parity/v1/OLD_SURFACE_SHA

rm -rf "$ARCHIVE_ROOT"
devtools mcp-parity build-fixture \
  --server-worktree "$OLD_TREE" \
  --server-sha "$OLD_SHA" \
  --manifest tests/fixtures/mcp_parity/v1/manifest.json \
  --archive-root "$ARCHIVE_ROOT" \
  --verify-hash

POLYLOGUE_MCP_FROZEN_NOW="2026-07-01T12:00:00Z" \
devtools mcp-parity capture-old \
  --server-worktree "$OLD_TREE" \
  --server-sha "$OLD_SHA" \
  --archive-root "$ARCHIVE_ROOT" \
  --role admin \
  --cases tests/fixtures/mcp_parity/v1/cases.jsonl \
  --equivalence docs/generated/mcp-equivalence.json \
  --normalization tests/golden/mcp_parity/v1/normalization.json \
  --output tests/golden/mcp_parity/v1/old

git diff --exit-code -- tests/fixtures/mcp_parity/v1
test -z "$(find tests/golden/mcp_parity/v1/old -type f -name '*.json' -size 0 -print -quit)"
```

`build-fixture` and `capture-old` must start fresh child interpreters whose working directory and first import root are `--server-worktree`, verify that tree's Git SHA and dependency-lock hash, and invoke the old FastMCP handlers through `tests.infra.mcp.invoke_surface`. The transition checkout owns orchestration, case selection, and normalization only. Old and new Polylogue module graphs must never coexist in one interpreter. A separate protocol smoke test exercises stdio discovery/framing.

## D8 — Normalization rules

Normalization is path-declared and versioned. It must never make a real semantic mismatch disappear.

1. Parse JSON and sort object keys recursively. Preserve array order unless a declaration marks that exact JSON path as a set; then sort by canonical ref/tie-breaker.
2. Freeze time in fixtures. For unavoidable timestamps, replace only declared volatile paths with stable tokens while separately asserting parseability, timezone, monotonicity, and expected relative ordering.
3. Alias generated UUID/call/result/operation IDs by first occurrence (`<uuid-1>`, `<result-ref-1>`) and preserve cross-field referential equality. Stable fixture refs/content hashes are not masked.
4. Replace only the exact temporary archive/repository roots with `<ARCHIVE_ROOT>`/`<REPO_ROOT>`. Do not basename arbitrary paths.
5. Remove wall-clock durations from equality only at declared paths; assert type, non-negativity, and bounded class separately. Timing-derived domain buckets remain semantic.
6. Compare typed errors by code, retryable flag, field/path, required role/capability, and recovery route. Ignore prose wording only.
7. For exhaustive pages, follow every cursor; compare each page's order/progression and the full logical ref sequence. Reject repeated/non-progressing cursors, missing/duplicate members, false totals, and full-result buffering.
8. For top-k, compare requested k, rank order, score/tie policy, evidence refs, frontier/result ref, candidate-count truthfulness, and model/version. Never normalize scores away.
9. For aggregates, compare dimensions, measures, scope, exactness, coverage, and null/empty behavior; formatting and map key order are nonsemantic.
10. For graphs, canonicalize nodes by ref and edges by `(kind,from,to,ordinal)` only after asserting traversal order/frontier behavior; compare cycle/completeness flags.
11. For bounded context, compare selected refs/order, omission reasons, policy, recipient, budget, and receipt. Token count may use a declared tolerance only when tokenizer/version is pinned; selected evidence may not differ.
12. For mutations/maintenance, clone the fixture per case. Compare public before/after state, receipt, authorization denial, idempotent repeat, partial failure, and recovery. Never dual-apply a destructive operation.
13. Text, query plans, authority roles, candidate/judged state, totals/exactness, refs, and content are never volatile.

## D9 — Shadow-call comparison shape

Read transition adapters lower old input and new input to one canonical plan, then execute both projections against the same immutable archive epoch. They emit only normalized fingerprints/diff classifications to transition telemetry; raw user payloads are not persisted. Required mismatch classes: missing/extra refs, order/rank, field/value, totals/exactness, continuation, error/authority, coverage/degradation, and resource bound/lifecycle.

Writes do not dual-write. The old handler is the sole authority during shadowing; the new route runs in `dry_run`/preview against a cloned fixture or transaction and its intended receipt/state delta is compared. After authority flips, reverse this arrangement for a bounded validation window if the old implementation remains in branch-only test code. Production never registers both public names.

Comparison gate:

```bash
POLYLOGUE_ARCHIVE_ROOT="$PWD/.cache/mcp-parity/archive-v1" \
devtools mcp-parity compare \
  --cases tests/fixtures/mcp_parity/v1/cases.jsonl \
  --old tests/golden/mcp_parity/v1/old \
  --normalization tests/golden/mcp_parity/v1/normalization.json \
  --require-zero-unexplained \
  --report .cache/mcp-parity/compare.json
```

Every intentional delta must be declared by row/path/reason in `cases.jsonl`; blanket ignores are forbidden. The branch cannot delete old handlers while any parity-golden case is missing, unexplained, or marked “not run.”

## D10 — Mapping-only evidence

A mapping-only row still requires: old declaration/required args/output semantics; exact target expression; role gate; field/continuation delta; capability test that the target plan validates; live census row proving zero use; and a reviewer-owned justification. It does not require an old output fixture. This keeps the long tail cheap without allowing silent capability deletion.

## D11 — Resources, prompts, and saved recipes

Stable objects/receipts are resources; parameterized workflows are prompts; execution remains in transactions. Add context and maintenance receipt resources missing from the landed target declaration. Prompts carry zero mutation authority, are exact-role scoped, and must pass a dependency check showing every referenced transaction/resource is visible at that role.

Do not retire the five existing server-side data prompts in this lane. Prompt/resource telemetry is missing, and their current implementations can remain useful atop the six-tool surface.

## D12 — Analytics continuity without runtime aliases

Extend transition call telemetry with canonical dimensions such as `transaction_name`, `operation`, `projection`, `semantics`, and `legacy_tool_name` (nullable during shadow). Keep old qualified-name parsing in analytics. Dashboard grouping maps old names to new families; server discovery exposes only new names.
