# HANDOFF — polylogue-z9gh.7 terminal continuity gate

## Operator summary

This package is an implementation draft for the deterministic and authorization/redaction portions of `polylogue-z9gh.7`. It composes the current continuity catalog, the shared MCP query transaction, Claude orchestration artifact admission, provider-neutral work evidence, repository-effect reconciliation, graph persistence, and the selective action-query planner into one terminal receipt.

The deterministic sanitized lane is executable and passes through the real MCP stdio client/server route. It starts the incident from only repository, approximate date, and parallel-agent wording; discovers the run and coordinator through public query evidence; binds all later queries from those observations; pages the 91-member population; tears down and reopens the MCP route after page one; resumes from the retained continuation; cites public evidence; composes the admitted native Claude artifact family; reconciles separate git/PR/Beads effects with explicit uncertainty; persists and traverses the claim/effect graph; and runs a 512-session selective-SQL canary.

This package does **not** claim full mandate closure. No external cold model, operator live archive, private 4.85-million-block corpus, live daemon, browser, secrets, GitHub check API, or current Beads database was available. The receipt therefore says the external cold-model proof is unavailable, the PR check evidence is unavailable, and Beads closure evidence is unavailable. The authorized live-lane mechanism was exercised only over the sanitized fixture to prove fail-closed authorization and deterministic redaction; that is not a live-scale execution claim.

## Snapshot identity and patch base

The attached project-state manifest identifies the authoritative snapshot as:

- generated: `2026-07-18T013442Z`
- branch: `master`
- commit: `bf8191b3f56aa40da8f271df7f3385c712825497`
- dirty working tree: `true`

The supplied dirty state changed `polylogue/daemon/http.py`, `polylogue/hooks/__init__.py`, `polylogue/archive/query/unit_results.py`, and added an ignored `browser-extension/package-lock.json`. I preserved that exact tree as local baseline commit `3a23389823b9a78fe03f497ee719ac9af670d815` and generated `PATCH.diff` against it. Those supplied dirty files are not included in this result patch.

The current t8t implementation found in the all-refs authority was source commit `1963ef875a20b960509460e250c0e594f8384ae2` (`feat: replay all continuity scenarios over real MCP stdio transport`). It was imported into the implementation tree as `fb8339a56d99563cfa653e5fbf9a3c362b2b38b0`, then extended by the terminal composition in this package.

## Evidence inspected

Repository instructions and architecture:

- `AGENTS.md` / `CLAUDE.md`, especially substrate-first ownership, current-source precedence, focused verification, generated-surface checks, acceptance-criteria honesty, and the prohibition on reviving parallel harnesses.
- Query and MCP production routes in `polylogue/archive/query`, `polylogue/storage/sqlite/archive_tiers/archive.py`, `polylogue/mcp`, and `devtools/continuity_replay.py`.
- Claude artifact admission/parsing in `polylogue/sources/origin_specs.py` and `polylogue/sources/parsers/claude/`.
- Work evidence and effect reconciliation in `polylogue/insights/claude_workflow_evidence.py`, `polylogue/insights/work_reconciliation.py`, and `polylogue/storage/repository`.
- Existing scenario/test infrastructure under `polylogue/scenarios`, `tests/infra/archive_scenarios.py`, and relevant query, parser, graph, and persistence tests.

Beads records from the attached archive:

- `polylogue-z9gh.7`: sole terminal gate, sparse incident replay, 4/50/91/65/49/1/38 census, effect uncertainty, paging/cancellation/SLOs, cold-model proof, mutation curriculum, and per-bead disposition.
- `polylogue-t8t`: eight executable declarations, independent known-answer oracles, original-attempt grading, and continuity mutations.
- `polylogue-z9gh.9.1`: canonical query transaction, opaque continuation state, lossless paging, query/result refs, cancellation, selective plans, and noted residual HTTP/API continuation parity.
- `polylogue-2qx.2`: complete Claude artifact family, exact incident census, provider provenance, transcript/meta pairing, exclusion of 38 unrelated children, and degraded facts instead of invented links.
- `polylogue-1vpm.6.2`: claims, observed effects, and evaluated judgments as distinct facts with repository/corpus snapshots and uncertainty.

Relevant history inspected:

- `4053787ab547299a4402e33e05f73d04840c74c3` — MCP continuity transaction replay.
- `9163d0134f3d334960e4c249c96c5671919a9a06` — bounded agent-facing archive reads / shared query transaction.
- `ed44be18f448c31f9fa5b9289c75da7eee99b131` — MCP tool algebra declarations.
- `fd7b3549292927fbd69e0cb07dff9a1205d8e6c8` — interruptible, admission-controlled reads.
- `bce7336d3bb2e493080b37fd0bc76b429b0c1cbd` — Claude/effect continuity precursor work.
- `1963ef875a20b960509460e250c0e594f8384ae2` — current t8t stdio replay implementation.

## Gate topology

### 1. Canonical declarations and independent oracle

`polylogue/product/continuity_scenarios.py` remains the declaration registry. `tests/data/continuity/catalog.json` plants synthetic corpus values and expected facts before any production route executes. The runner never calls the route under test to construct an expected answer.

The incident declaration now authors only this initial query:

```text
messages where text:"repo:polylogue" AND text:"date:2026-01-15" AND text:"parallel-agent"
```

The returned public message contains the sanitized run and coordinator identities. `ContinuityBindingProjection` extracts them with `regex_single`; every later membership, topology, invocation, final-result, and curriculum expression is materialized from those runtime bindings. Fixture IDs remain only in the independent oracle and corpus compiler.

### 2. Real MCP route and capability discovery

`devtools/continuity_replay.py` uses the official MCP SDK over stdio by default. It initializes the installed read server, follows paged `list_tools`, records the protocol/server/tool inventory, validates each scenario's required tool schema fields, and invokes the public tools with JSON-RPC framing. The registered FastMCP route remains only as a fast mutation seam over the same production handlers.

The standalone smoke discovered protocol `2025-11-25`, server `polylogue` version `1.28.1`, and 64 tools. No additional terminal transport or scenario harness was introduced.

### 3. Lossless page / stop / restart / resume

The incident member step performs an independent count probe, requests pages of 17, verifies each page envelope, query/result identity, offsets, totals, item identities, continuation progress, and population exactness, then merges exactly 91 unique messages.

After page one, the route is closed and reopened. The same opaque continuation is retained outside the route process and submitted to the new MCP server process. Discovery is fingerprinted before and after restart. The successful receipt records six pages (`17,17,17,17,17,6`), one restart, stable discovery, and exact enumeration.

This is a route/process stop-and-resume proof. Existing z9gh.1 execution-control tests remain the production proof for interrupting an in-flight SQLite statement; this patch does not invent a second cancellation mechanism.

### 4. Advisory SLO receipts

Call count, page bytes, total bytes, elapsed time, cancel/restart grace, and process peak RSS are measured and reported. Thresholds provide paging/resume guidance and never reject otherwise valid complete work solely because an SLO is exceeded.

In the standalone stdio run, the incident used 27 calls, 281,027 response bytes, a 23,958-byte largest page, 5,056.612 ms scenario time, and 372,809,728 bytes process peak RSS. Those were within their declared guidance. Route restart/resume took 4,514.813 ms against a 1,000 ms grace target, so the receipt correctly contains `max_cancel_grace_ms_guidance_exceeded` while the exact result remains valid.

### 5. Claude artifact admission, provenance, and work graph

The fixture compiler writes a deterministic native Claude corpus under `continuity-evidence/`:

- one coordinator JSONL stream with one direct human prompt and four Workflow tool invocations;
- one run-state snapshot;
- one journal with 50 call records, 91 attempt records, and 65 result records;
- 91 agent transcripts and 91 metadata sidecars;
- one adopt manifest linking the resumed run;
- independent git, GitHub PR, and Beads receipts.

The terminal verifier inventories the paths through `artifact_rule_for_path` and `inventory_claude_orchestration_artifacts`, parses fact artifacts through the production Claude orchestration parser, parses transcript authoredness through the production Claude Code parser, and projects the generic work graph through `project_claude_workflow_evidence`.

The graph must contain four invocations, 50 calls, 91 attempts, 65 structured results, three claims, one resumed edge, 182 `represented_by` edges (transcript plus metadata for every attempt), one unresolved call, and no node derived from the 38 unrelated child sessions. All 91 worker prompts must be `generated_context_pack`; the one coordinator operator prompt must remain `human_authored`.

### 6. Claim/effect/judgment separation

The final structured result emits direct effect refs for a git commit, a GitHub PR, and a Beads artifact. Those remain claims until independent receipts are loaded. The terminal verifier creates three separate `ObservedRepositoryEffect` nodes and three separate `ReconciliationJudgment` rows:

- git commit: `supported` because the sanitized independent receipt says the commit is present and the tree matches;
- GitHub PR: `partial` because the PR/head relation is observed but checks are unavailable;
- Beads: `unresolved` because the issue is still open and closure evidence is unavailable.

`reconcile_work_effects` attaches the effects without collapsing claim identity. The reconciled graph is persisted through `SessionRepository.replace_work_evidence_graph`, and each claim-to-effect edge is read back with `traverse_work_evidence`.

### 7. Selective SQL canary

`verify_selective_action_sql` seeds 512 sessions and submits an exact-session, multi-dimensional action aggregate through the shared production query route. Before execution it calls the production `_action_relation_for_query` planner and requires:

- relation `bounded_actions`;
- a bounded-actions CTE;
- exactly three target-session bind parameters.

It then verifies the semantic result, selected-row census, emitted page, cleanup receipt, and VM-work advisory. This direct production relation contract replaces two stale tests whose old `>= 50,000 VM steps` mutant threshold was defeated by the current SQLite optimizer's predicate pushdown.

### 8. CI and authorized redacted lanes

The CI lane rejects any catalog that is not explicitly `classification=sanitized` and `contains_live_identifiers=false`.

The live lane requires both:

1. `--authorize-live-redacted`; and
2. `POLYLOGUE_LIVE_CONTINUITY_AUTHORIZATION=I_AUTHORIZE_POLYLOGUE_REDACTED_LIVE_SCALE`.

It also requires a redaction salt of at least 32 bytes. Before a receipt leaves the process, all non-whitelisted strings are replaced by deterministic HMAC-SHA256 values; route argument objects are replaced wholesale by a canonical-JSON HMAC. A sanitized-fixture smoke confirmed that raw archive paths, fixture IDs, run IDs, coordinator IDs, query text, continuation tokens, query/result refs, and evidence refs were absent.

### 9. Cold-model oracle boundary

An optional external receipt may be supplied with `--cold-model-receipt`. It must bind to the SHA-256 of the sparse question, declare only allowed input classes (`mcp_tool_schema`, `mcp_error`, `continuity_catalog`), use a declared equivalent plan signature, and provide evidence refs and facts.

When no receipt is supplied, the terminal result explicitly reports `status=unavailable`, `deterministic_sparse_route_verified=true`, and `model_competence_claimed=false`. No model-success claim is inferred from deterministic code execution.

## Changed files

| File | Change |
|---|---|
| `devtools/continuity_replay.py` | Real MCP stdio/registered routes, schema discovery, runtime bindings, exact paging/count receipts, route restart/resume, advisory budgets, CI/live authorization, HMAC redaction, terminal composition, and cold-receipt CLI input. |
| `devtools/continuity_terminal.py` | New composition layer over existing admission, parser, graph, reconciliation, persistence, and query seams; no second transport/harness. |
| `polylogue/product/continuity_scenarios.py` | Eight declarations, runtime binding projections, sparse incident route, restart point, budgets, facts/evidence/curriculum/mutations. |
| `polylogue/insights/claude_workflow_evidence.py` | Admitted artifact nodes, attempt-to-transcript/meta representation edges, structured-result claim nodes, and retained claim text. |
| `polylogue/mcp/server_prompts.py` | Correct query examples to use public session fields and separate `since` argument. |
| `tests/data/continuity/catalog.json` | Sanitized corpus, independent oracles, incident curriculum, privacy declaration. |
| `tests/infra/archive_scenarios.py` | Preserve structured tool-result IDs/error/exit-code facts in scenario records. |
| `tests/infra/continuity.py` | Deterministic archive and native Claude/effect corpus compiler plus direct SQLite census. |
| `tests/infra/continuity_mutations.py` | Named t8t discovery/formulation/execution/pagination mutations. |
| `tests/integration/test_continuity_replay.py` | Official stdio all-scenario walk, exact incident/terminal assertions, t8t mutation curriculum. |
| `tests/integration/test_terminal_continuity_gate.py` | Source coverage, orchestration link, provenance, effect mapping, identity-collapse, and selective-SQL mutations. |
| `tests/unit/devtools/test_continuity_replay_lane.py` | CI privacy, two-party live authorization, deterministic redaction, and cold-receipt scope tests. |
| `tests/unit/product/test_continuity_scenarios.py` | Declaration/oracle independence, sparse authored route, runtime binding, and schema validation. |
| `tests/unit/insights/test_claude_workflow_evidence.py` | Admitted representation links and claim preservation. |
| `tests/unit/mcp/test_prompt_query_parity.py` | MCP prompt/query grammar parity. |
| `tests/unit/archive/query/test_execution_control.py` | Stable direct relation-selection mutation check while retaining execution receipts. |
| `tests/unit/storage/test_archive_tiers_archive.py` | Stable direct bounded-action planner canary while retaining provider-pipeline semantics. |

No existing test or helper was deleted. No dominated deletion is proposed in this package.

## z9gh.7 acceptance matrix

| AC | Result | Evidence / limitation |
|---|---|---|
| 1. Seven t8t flows plus incident pass as real MCP walks | **Satisfied for sanitized CI** | Official MCP stdio standalone and integration run: 8 passed, 0 failed. |
| 2. Sparse incident discovers coordinator/run and proves 4/50/91/65/49/1/38/final | **Satisfied for sanitized CI** | Runtime bindings from public discovery; exact page census; terminal provider graph. Synthetic IDs stand in for the private IDs named by the Bead. |
| 3. Distinguish model/material/call/attempt/effect scopes and cite git/PR/Beads with uncertainty | **Satisfied for sanitized CI** | Positive material-origin checks, provider-owned membership, separate claim/effect/judgment facts, supported/partial/unresolved evaluations. |
| 4. Lossless paging, cancellation, measured latency/memory within SLOs | **Partial** | Lossless page/restart/resume and metrics are verified. Calls/page/bytes/time/RSS were within guidance; restart grace was 4.515 s against 1 s and is reported as advisory. In-flight SQLite cancellation remains consumed from existing z9gh.1 tests rather than duplicated here. |
| 5. Cold model succeeds from schemas/errors/catalog alone | **Still blocking / unverified** | Receipt validator is implemented; no external cold-model execution was available. Gate reports unavailable and makes no competence claim. |
| 6. Mutations fail for continuation, selective SQL, links, coverage, provenance | **Satisfied for sanitized CI** | t8t route curriculum plus terminal source/provenance/effect/SQL tests fail at named boundaries and classes. |
| 7. Record each mandate Bead as satisfied/deferred/blocking | **Satisfied by this handoff matrix** | Tracker state was not mutated; current archived Beads remain open/in-progress. |

## Dependency/Bead disposition

| Bead | Mechanism result in this package | Tracker truth / remaining scope |
|---|---|---|
| `polylogue-t8t` | **Satisfied for gate consumption**: current catalog imported, sparse binding repaired, all eight stdio scenarios and named mutation curriculum pass. | Archived record is `in_progress`; no Beads mutation was made. |
| `polylogue-z9gh.9.1` | **Consumed for MCP terminal route**: canonical query/result refs, opaque continuations, exact totals, restart resume, bounded action relation, cancellation guidance. | Archived record is `in_progress`; its own notes still name HTTP/API continuation parity residue outside this gate. |
| `polylogue-2qx.2` | **Substantial deterministic subset satisfied**: complete sanitized artifact family, 91 transcript/meta pairs, 4/50/91/65/49/1 census, generated-vs-human provenance, explicit link gaps. | Still blocking full Bead closure: no live-row semantic reparse census and no real `wf_54d4fb2e-841` artifact intake were executed. Archived record is `open`. |
| `polylogue-1vpm.6.2` | **Substantial deterministic subset satisfied**: claims/effects/judgments remain distinct; three authorities, uncertainty, persistence, and traversal are exercised. | Still blocking full Bead closure: no real git/GitHub/complete-Beads-history reconciliation, 25-open-P1 baseline proof, branch/squash/correction matrix, or live seeded production query. Archived record is `open`. |
| `polylogue-z9gh.7` | **Deterministic CI and authorization/redaction implementation complete**. | Still blocking full closure on external cold-model success and explicitly authorized live-scale execution/receipt. Archived record is `open`. |

## Mutation matrix

| Mutation | Production dependency exercised | Required diagnosis |
|---|---|---|
| Lost continuation request state | MCP `query_units` continuation transaction | execution failure / non-resumable route |
| Capped pseudo-total | query page totals and independent count probe | execution / population mismatch |
| Identical-call topology replay | advancing continuation and unique identities | execution / duplicate or non-progressing continuation |
| Hidden fact/grammar discovery | MCP tool schema and declaration requirements | discovery failure |
| Missing public source coverage | public incident partition/evidence | source coverage |
| Unreasonable query classification | t8t attempt grader | reasoning classification |
| Global-first action relation | `_action_relation_for_query` | `selective_sql_plan_amplification` / `plan` |
| Remove one agent metadata sidecar | OriginSpec inventory and pairing | `terminal_source_coverage_mismatch` / `source_coverage` |
| Remove journal `metaPath` | graph representation links | `terminal_orchestration_representation_mismatch` / `source_coverage` |
| Reclassify generated agent prompt as human | Claude Code material-origin parser | `material_origin_collapse` / `provenance` |
| Change direct effect ref | final result vs independent effect receipt | `terminal_effect_ref_mismatch` / `effect_mapping` |
| Collapse PR effect onto commit identity | reconciliation graph identity | `claim_effect_collapse` / `effect_mapping` |

## Apply order

1. Start from the exact attached working-tree snapshot: `master@bf8191b3f56aa40da8f271df7f3385c712825497` plus its supplied dirty delta. The locally reconstructed equivalent tree is `3a23389823b9a78fe03f497ee719ac9af670d815`.
2. From the repository root, run:

```bash
git apply --check PATCH.diff
git apply PATCH.diff
git diff --check
```

3. Run the focused static and test commands from `TESTS.md`.
4. For deterministic CI, seed only the repository fixture and run the default `ci` lane.
5. For an operator live archive, require explicit command-line authorization, the exact environment acknowledgement, and a fresh 32-byte-or-longer HMAC salt. Store only the redacted receipt.
6. Supply an independently produced cold-model receipt with `--cold-model-receipt` before claiming z9gh.7 AC5.

The patch was reapplied successfully in a detached worktree at `3a23389823b9a78fe03f497ee719ac9af670d815`; all 17 patched files matched the implementation tree byte-for-byte after application.

## Verification performed

Successful checks:

- Ruff format check over all 16 changed Python files.
- Ruff lint over all 16 changed Python files.
- strict Mypy over all 16 changed Python files.
- `compileall` over all 16 changed Python files.
- `git diff --check` against the exact dirty-snapshot baseline.
- `python -m devtools render all --check` with every generated surface in sync.
- 45 focused dependency/unit tests passed.
- 6 terminal integration/mutation tests passed.
- 10 official continuity replay integration tests passed.
- Standalone official MCP stdio script: 8 scenarios passed, 0 failed; terminal gate passed.
- Authorized redaction smoke over the sanitized fixture: status passed and all scanned raw secrets absent.
- Fresh-worktree `git apply --check`, apply, diff check, byte-for-byte 17-file comparison, and changed-file static checks passed.

Execution constraints and non-passing orchestration attempts are recorded in `TESTS.md`; none is represented as a product pass.

## Risks and remaining verification

The largest remaining risks are evidentiary, not scaffolding:

- The private incident IDs and live artifact volumes were not exercised. The synthetic corpus proves composition and mutations, not real-source completeness at operator scale.
- No external cold model was run. The receipt contract prevents hidden fixture inputs but cannot substitute for the run.
- The redaction lane was exercised over sanitized data only. A real live archive may expose unanticipated dynamic mapping keys or output shapes; the first authorized operator run should scan the entire serialized receipt for raw source tokens before retention.
- `process_peak_rss` measures the runner process and reaped child behavior available to this environment; it is not a full per-process memory profile of a live daemon deployment.
- The 1-second restart-grace guidance was exceeded locally. It correctly remained advisory, but live operators should decide whether to raise the guidance or optimize MCP startup after observing their deployment.
- GitHub checks and Beads closure remained unavailable. The effect judgments are intentionally `partial` and `unresolved` rather than upgraded by self-report.
- Full repository Mypy did not complete within the local 10-minute command cap; focused strict Mypy passed. The managed test wrapper refused this container's 64 MiB `/dev/shm`; focused raw pytest was used with deterministic plugin exclusions.

## Value of another iteration

A code-only follow-up is likely a **small repair pass**: review the 17-file patch, tighten receipt shape or diagnostics, and address any apply/environment defect found by the operator.

A **substantial second pass** becomes valuable only with new authority: explicit access to the live archive/daemon, an operator authorization and redaction key, an external cold-model runner constrained to MCP schemas/errors/catalog evidence, real git/GitHub/complete-Beads receipts, and permission to update tracker state. That pass could close the currently unverified ACs; repeating the deterministic fixture work alone would add little value.
