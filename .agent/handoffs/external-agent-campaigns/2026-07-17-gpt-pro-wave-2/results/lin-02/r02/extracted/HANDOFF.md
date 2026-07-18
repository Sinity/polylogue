# HANDOFF — continuity replay scenarios and independent known-answer oracles

Revision: `lin-02-continuity-scenarios-r02`

## Mission and second-pass outcome

This revision is a complete replacement package, not a supplemental patch. It preserves the valid declaration, fixture, and known-answer work from `r01`, then closes the highest-value gaps left by that iteration:

1. The default replay path now launches the production read-role MCP server as a child process and communicates through the official MCP SDK over stdio JSON-RPC. It records initialized protocol/server metadata and discovered input-schema hashes before executing any scenario.
2. Paginated routes no longer trust a page-local or wrapper-rewritten total. Every aggregate-capable terminal unit receives a separate production `| count` probe; the runner then verifies stable query/result refs, advancing offsets, exact identity uniqueness, page-local totals, and equality between independently selected count and enumerated rows.
3. The parallel incident now includes a six-case sanitized curriculum whose raw inputs are planted separately from its expected grades. The replay classifies those inputs and compares its grades with the independent oracle.
4. All six mutation families named by `polylogue-t8t` are executable and must fail in the intended failure layer.
5. Scenario declarations now expose structured discovery requirements, coverage inventories, result semantics, accepted plan signatures, exact item-identity paths, stop conditions, and call/page/aggregate/elapsed/cancellation-grace bounds.
6. Two shipped MCP prompt recipes that advertised parser-invalid query expressions are corrected, and a parser/schema parity test now checks every shipped `query_units` recipe.

The resulting synthetic replay covers the seven operator jobs plus the parallel-Claude incident variant. The incident fixture mirrors the corrected Beads census: 129 coordinator children partitioned into 91 target-run attempt transcripts and 38 other children, with 50 call keys, 65 result records, 49 completed keys, one unresolved key, four Workflow invocations, one final structured result, and six original-call curriculum cases.

This package remains the reusable `polylogue-t8t` catalog and harness. It does not claim to complete the authorized live-archive, cold-model, effect-evidence, SLO, or terminal-gate work owned by `polylogue-z9gh.7`.

## Snapshot identity and patch target

The supplied authority identifies the snapshot as:

- Branch: `master`
- Commit: `536a53efac0cbe4a2473ad379e4db49ef3fce74d`
- Commit date: `2026-07-17 18:55:47 +0200`
- Subject: `fix(repair): harden raw authority convergence (#3046)`
- Snapshot metadata: `dirty=true`

Identity was confirmed from `polylogue-overview.json`, the bundled Git graph, and `git show` on the reconstructed commit. The supplied branch-delta report names the same commit as the `origin/master` merge base and has an empty diff, empty commit list, empty file list, and zero-byte patch. Comparing the captured tracked working tree with the commit likewise found no recoverable tracked source delta. Therefore `PATCH.diff` targets the exact named commit. No unknown dirty state was fabricated.

Patch identity:

- Paths changed: 10
- Insertions: 4,006
- Deletions: 305
- Bytes: 183,943
- SHA-256: `4ad396b5c1563d45a7b5859431f48aed45e8bbff108ae71cb92d06e0a55ce9d2`

## Authority and repository evidence inspected

Process and repository contracts:

- `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`, and `TESTING.md`
- `.beads/issues.jsonl`, especially the complete current records for `polylogue-t8t` and `polylogue-z9gh.7`
- snapshot overview, manifest, branch-delta artifacts, working-tree archive, repository tree, all-refs bundle, and Git path history
- prior continuity implementation in commit `9163d0134 feat(query): bound agent-facing archive reads (#3018)`

Existing scenario and fixture seams:

- `polylogue/scenarios/specs.py`, `metadata.py`, `sources.py`, `projections.py`, and package exports
- `polylogue/product/workflows.py`
- `tests/infra/archive_scenarios.py`, `tests/infra/storage_records.py`, and `tests/infra/mcp.py`

Production dependencies followed beyond the declaration files:

- `polylogue/mcp/server.py`, `server_tools.py`, `server_support.py`, `server_prompts.py`, declarations, and role registration
- `polylogue/mcp/cli.py` and the child-process entry point
- archive query expression parsing, terminal-unit selection, aggregation, lowering, transaction, and execution control
- `polylogue/api/archive.py` and runtime service construction
- SQLite archive tiers, assertions, provider-usage storage, run lineage, message blocks, tool-use/result pairing, and read-view profiles
- existing MCP contract, prompt, query-unit, fixture, and execution-control tests

## Seam decision

The canonical declaration type is `ContinuityScenarioSpec(NamedScenarioSource)`. `NamedScenarioSource` already participates in the repository's `ScenarioSpec`, `ScenarioMetadata`, and projection-source contracts. The continuity declarations therefore remain authored validation-lane scenarios instead of becoming a competing continuity-only registry.

The fixture compiler uses `ArchiveScenario`, `ScenarioMessage`, `ScenarioContentBlock`, and `SessionBuilder`. The only extension to that established seam is optional tool-result linkage/error/exit metadata required for production `actions` rows. Assertions and usage records use their existing storage primitives because they are not transcript-only facts.

`tests/infra/continuity.py` is a compiler from an independent manifest into those existing builders. `devtools/continuity_replay.py` is a black-box executor over the actual MCP server. It does not implement a fake parser, fake query engine, alternate archive, or alternate scenario framework.

## Scenario inventory

| Job | Sparse operator question | Independently planted answer | Public route and completeness proof |
|---|---|---|---|
| Resume | “What was I doing in this repo?” | One matching message, exact session ID, exact continuation text, exact message ref | `query_units(messages …)`, continuation exhausted; separate exact count; unique `message_id` enumeration |
| Forensic debug | “Where did this failure happen?” | One failed action, `src/continuity_target.py`, exit 17, one file row, exact file path, message/tool/file refs | `query_units(actions … is_error:true)` plus `query_units(files …)`; each count-probed and identity-checked |
| Prior art | “Have we solved this before?” | One prior-art hit, exact session ID, exact prior solution text, exact message ref | `query_units(messages …)`, count-probed and identity-checked |
| Decision | “What did we decide about this?” | One active decision assertion, exact assertion ID/body/status, assertion and source refs | `query_units(assertions where kind:decision …)`, count-probed and identity-checked |
| Postmortem | “Why did the last run fail?” | One failed Bash action, exit 2, exact output, message/tool refs | `query_units(actions … is_error:true)`, count-probed and identity-checked |
| Cost audit | “How much did this work cost?” | 1,200 input, 300 output, 400 cached-input, 50 cache-write, 1,950 total tokens; physical-session pricing grain | `provider_usage(origin=…, detail=headline)` through MCP; exact aggregate counters retained in bounded response metadata |
| Self-inspection | “What can you answer about agent work?” | Selected terminal unit `assertion`, zero unsupported nodes, eleven exact read-view IDs | `explain_query_expression` plus `list_read_view_profiles`; accepted in either order |
| Parallel-Claude incident | “Which agents handled the concerns and what changed?” | 91 attempts, 50 call keys, 65 results, 49 completed, one unresolved, 38 non-run children, 129 total children, four invocations, one final result, six curriculum cases | Six `query_units` steps; message populations count-probed; 91 target members enumerated once across six pages; run rows use lineage identity plus independent 129/91/38 oracle because `runs | count` is not currently lowered |

Every scenario additionally declares:

- allowed public tools and machine-readable discovery requirements;
- canonical and accepted equivalent plan signatures;
- fact and evidence projections with failure classes;
- source/coverage inventory and result exactness semantics;
- item-identity paths for paginated rows;
- stop conditions;
- named mutations;
- maximum calls, page bytes, total bytes, elapsed time, and cancellation grace.

## Independent corpus and oracle design

`tests/data/continuity/catalog.json` is schema version 2 and has deliberately separate `corpus` and `oracles` sections.

The corpus section contains only source facts used to construct a synthetic archive. The oracle section contains expected answer facts, expected evidence refs, source refs, and expected incident grades. `seed_continuity_archive()` reads the corpus and builds fresh archive tiers before the replay starts. `validate_continuity_population()` then performs a direct SQLite census without importing or invoking `query_units`, `provider_usage`, MCP handlers, or the replay projector.

For the incident curriculum, the archive contains raw features such as query shape, physical size, corpus match, structural-discovery availability, shipped-instruction exposure, and execution outcome. It does not contain the expected grade. The independent oracle maps each case ID to its expected grade. The runner derives grades from the raw features, then compares them with the oracle.

Two explicit independence axes are tested:

- Production-route mutation: removing the target Workflow filter broadens the observed population from 91 to 129 and produces a source-coverage fact mismatch.
- Oracle-only mutation: changing the planted expected value from 91 to 92 leaves production output at 91 and reports expected 92 versus observed 91.

## Public-route execution

The default `StdioMCPContinuityRoute` launches:

```text
<active-python> -c "from polylogue.mcp.cli import main; main()" --role read
```

It isolates archive/config state through environment variables, initializes an MCP client session using the official SDK, receives the production tool catalog, and calls tools over stdio JSON-RPC. The executed path is:

```text
MCP SDK client
  -> stdio JSON-RPC framing
  -> polylogue.mcp.cli
  -> build_server(role="read")
  -> registered FastMCP handler
  -> RuntimeServices / Polylogue archive facade
  -> production parser, lowering, query transaction, or diagnostic API
  -> SQLite archive tiers
  -> production MCP response
```

Discovery receipts retain transport, MCP protocol version, server name/version, tool count, selected input-schema hashes, description hashes, required arguments, and visible arguments. A scenario fails before execution when a required tool or argument is absent.

`MCPContinuityRoute`, the registered-handler adapter retained from `r01`, is now used only for controlled fault injection in fast mutation tests. The primary all-scenario acceptance path is stdio JSON-RPC.

## Pagination, exact-count probes, and receipts

For each paginated step the runner:

1. executes a separate `expression | count` call where the terminal unit supports aggregate lowering;
2. invokes the original expression with the declared page limit;
3. follows the production continuation token until exhaustion;
4. checks that offsets advance exactly and no continuation repeats;
5. checks query and result refs remain stable across the sequence;
6. checks every page reports a page-local total equal to the items returned;
7. hashes each page response, arguments, and identity set;
8. rejects duplicate item identities across pages;
9. verifies the enumerated row count equals the exact count probe;
10. enforces per-page, total-byte, call-count, and elapsed budgets.

The runner does not rewrite production totals. Receipts distinguish page totals, enumerated totals, unique identities, count-probe selection, and terminal continuation state.

The incident target population is intentionally paged at 17 rows. The fresh replay enumerated 91 unique members as `[17, 17, 17, 17, 17, 6]`, and the separate aggregate route selected exactly 91.

The `runs` unit currently has no aggregate lowerer, so its route has no `| count` probe. Completeness there is supported by exact run-row identities, exhausted continuation, the independently planted 129-child census, and the separately count-probed 91/38 message partition. This is a stated boundary, not an implied aggregate guarantee.

## Incident curriculum and mutation coverage

The six independently graded sanitized cases are:

| Case | Expected grade | Basis |
|---|---|---|
| Candidate list | `reasonable_oversized` | Reasonable corpus/shape, but physically oversized request |
| Exact operator phrase | `wrong_corpus_assumption` | Bounded query against the wrong corpus assumption |
| Sonnet lexical proxy | `weak_lexical_proxy` | Lexical fallback induced by absent model/material structure discovery |
| Sessions-only query | `product_induced_hidden_grammar` | Reasonable under shipped instructions that advertised a parser-invalid nonterminal shape |
| Correct topology call | `execution_failure` | Correct formulation with transport/executor failure |
| Correct delegation call | `execution_failure` | Correct formulation with timeout |

The six executable mutation families are:

| Mutation | Required diagnosis |
|---|---|
| Lost request-state continuation | `pagination_offset_mismatch` / execution |
| Capped pseudo-total | `pagination_count_mismatch` / execution |
| Identical-call topology replay | `duplicate_pagination_identity` / execution |
| Hidden fact/grammar discovery | `missing_discovered_arguments` / discovery |
| Missing source coverage | `non_single_projection` / source coverage |
| Unreasonable-query classification | `attempt_grade_mismatch` / reasoning |

## Shipped prompt parity repair

The second pass found two real product/curriculum defects in `polylogue/mcp/server_prompts.py`:

- `unacknowledged_failures` embedded `since` inside an action predicate even though `since` is a top-level `query_units` argument;
- `sessions_touching_file` used `repo` as a file predicate even though repository scope is expressed through the session field/top-level scope accepted by the parser.

The recipes are corrected to executable forms. `tests/unit/mcp/test_prompt_query_parity.py` renders every shipped prompt containing a `query_units` recipe, parses each advertised expression with the production parser, and verifies every advertised argument exists in the production-discovered tool schema. This converts the Beads query-grade correction into a maintained product contract.

## Harness entry points

Declarations:

```python
from polylogue.product.continuity_scenarios import (
    CONTINUITY_SCENARIOS,
    continuity_scenario,
)
```

Synthetic fixture compiler:

```python
from tests.infra.continuity import (
    load_continuity_catalog,
    seed_continuity_archive,
    validate_continuity_population,
)
```

Programmatic replay:

```python
from devtools.continuity_replay import replay_archive

report = await replay_archive(archive_root, oracle_catalog)
```

Command-line replay:

```bash
python devtools/continuity_replay.py \
  --archive-root /path/to/archive \
  --oracle /path/to/oracle-v2.json \
  --transport stdio \
  --output continuity-receipt.json
```

The `--transport registered` mode exists for controlled in-process mutation tests; it is not the primary black-box acceptance route.

## How `polylogue-z9gh.7` can consume the catalog

The live gate should provide an authorized, privacy-reviewed schema-v2 manifest whose corpus selectors identify the real archive facts and whose oracle section is independently audited from source/repository evidence. It can then invoke the same runner:

```bash
python devtools/continuity_replay.py \
  --archive-root "$POLYLOGUE_ARCHIVE_ROOT" \
  --oracle /authorized/z9gh.7-oracle-v2.json \
  --transport stdio \
  --output z9gh.7-continuity-receipt.json
```

That invocation alone is not sufficient for `z9gh.7`. The terminal gate must additionally capture a cold model's discovery/formulation transcript, real coordinator/run membership evidence, model/material/call/attempt/effect scope distinctions, git/PR/Beads effects with uncertainty, transport cancellation, latency/memory SLOs, and required mechanism-removal mutations.

## Changed files

- `devtools/continuity_replay.py` — schema-v2 loader, stdio MCP route, discovery validation/receipts, exact pagination/count checks, projection, curriculum grading, budgets, diagnostics, CLI, and retained registered mutation adapter.
- `polylogue/mcp/server_prompts.py` — fixes two parser-invalid shipped `query_units` recipes.
- `polylogue/product/continuity_scenarios.py` — all eight structured declarations and typed discovery/result/budget/curriculum contracts.
- `tests/data/continuity/catalog.json` — independent synthetic corpus, known-answer facts, evidence refs, and incident grades.
- `tests/infra/archive_scenarios.py` — established fixture seam extended with structural tool-result linkage/error/exit fields.
- `tests/infra/continuity.py` — corpus compiler and independent direct-SQLite census using existing fixture/storage primitives.
- `tests/infra/continuity_mutations.py` — reusable six-family fault-injection curriculum.
- `tests/integration/test_continuity_replay.py` — official stdio replay, pagination receipts, all named mutations, and two oracle-independence tests.
- `tests/unit/mcp/test_prompt_query_parity.py` — shipped prompt/parser/discovered-schema parity.
- `tests/unit/product/test_continuity_scenarios.py` — declaration structure, schema-v2 independence, classifier, and projector tests.

No existing test or helper is deleted. No production schema migration is introduced. No generated archive or source snapshot is included.

## `polylogue-t8t` acceptance matrix

| Criterion | Status in r02 | Evidence and boundary |
|---|---|---|
| 1. Seven declarations plus incident | Satisfied | Eight executable `ContinuityScenarioSpec` entries; all pass through MCP stdio JSON-RPC |
| 2. Wording, facts/coverage, independent target, discovery, refs, plan equivalence, paging/cancel/resource bounds, stop conditions | Satisfied as declaration/catalog contract | Structured fields and schema-v2 oracle; actual transport cancellation remains criterion 3/terminal-gate work |
| 3. Baseline real-agent transcripts/server receipts classify all failure layers | Partial | Server discovery/call/page receipts and mutations distinguish source coverage, discovery, execution, projection, and reasoning; no cold external model transcript and no runner-issued cancellation transcript |
| 4. Original incident calls graded with exposed curriculum | Satisfied synthetically | Six raw-feature cases are planted separately from six expected grades; prompt parity repair preserves the product-induced sessions-only classification |
| 5. Six named mutation fixtures | Satisfied | All six execute and fail with the expected kind/failure class |
| 6. `z9gh.7` consumes catalog as sole terminal gate | Ready but not executed | Corpus is parameterized and CLI documented; authorized live gate wiring/run remains with `z9gh.7` |

## `polylogue-z9gh.7` boundary

This revision provides reusable support for the terminal gate but does not satisfy the following live requirements:

- recovery of coordinator `cf0c6474-da22-44be-af3e-666037aa5ea4` and run `wf_54d4fb2e-841` from the operator's authorized archive;
- real run-state/journal/metadata/source-ref membership proof rather than synthetic markers;
- git, PR, Beads, and other effect evidence;
- explicit model, material, call, attempt, session, claim, and effect scope distinctions;
- cold-model success from discovery/errors/catalog alone;
- MCP cancellation request/acknowledgment and measured cancellation grace;
- declared live latency and memory SLO measurement;
- live mechanism-removal tests and final mandate-bead disposition.

## Verification performed

Exact commands and results are recorded in `TESTS.md`. The final patch was checked and applied in a fresh detached worktree at the named commit. The continuity suite passed there. On the implementation tree:

- 20 continuity declaration/replay/parity tests passed;
- 82 existing MCP server-surface tests passed;
- 4 existing archive-scenario tests passed;
- 3 existing query execution-control/cancellation tests passed;
- Ruff lint and format checks passed for all nine changed Python files;
- strict mypy passed for all nine changed Python files;
- Python compilation and `git diff --check` passed;
- a fresh command-line replay passed 8/8 scenarios over MCP stdio JSON-RPC;
- direct SQLite census returned 129/91/38, four invocations, six curriculum cases, one final result, and 1,950 total tokens;
- all six named mutation reports failed in the intended failure class.

The fresh machine report was 79,582 bytes with SHA-256 `d1871519f9b6bda7003c1fa7a50ca1dd33c3af439be435a4de3376b51d90453e`. It recorded MCP protocol `2025-11-25`, server `polylogue` `1.28.1`, eight passes, zero failures, and 3,607.369 ms aggregate replay elapsed time. The incident scenario used 25 calls, 276,340 response bytes, and 468.446 ms scenario elapsed time.

## Risks and limitations

- The runner enforces elapsed bounds after/between calls but does not yet place a preemptive per-call deadline around an MCP call. A hung child call can therefore exceed a scenario bound before the runner regains control.
- The runner declares cancellation grace and records `cancellation_exercised=false`; it does not send an MCP cancellation notification or prove transport-level cancellation. The three passing execution-control tests prove the underlying query substrate, not the new replay transport.
- Exact count probes share the production parser/filter/lowering stack with the paginated query. They independently test continuation mechanics and enumeration, but they are not an oracle independent of all query semantics. The fixture/oracle remains independently planted.
- `runs | count` is not supported by current aggregate lowering, so run-row completeness lacks the extra count-probe triangulation used for messages/actions/assertions/files.
- The cost scenario validates exact reported token counters retained in Polylogue's bounded response envelope. It does not prove lossless provider-usage pagination or a currency bill.
- Mutation tests use the registered-handler adapter and FastMCP's current registration table to inject faults. The default acceptance path uses the official SDK/stdio route.
- The child process uses the active Python interpreter and repository checkout. Environment-specific packaging or SDK-version differences may require a small compatibility repair.
- No live daemon, SSE/HTTP transport, operator archive, secrets, Nix closure, deployment, or full repository test matrix was available or claimed.

## Apply order

From a clean checkout at `536a53efac0cbe4a2473ad379e4db49ef3fce74d`:

```bash
git apply --check PATCH.diff
git apply PATCH.diff
```

Then install the repository's existing development/test dependencies and run the commands in `TESTS.md`. `FILES/` is intentionally absent because the unified patch fully and unambiguously represents every change.

## Proposed follow-up beads

These are proposed successor scopes, not existing Bead IDs:

1. **Transport cancellation and hard-deadline continuity receipts** — add per-call deadlines, MCP cancellation request/acknowledgment capture, child cleanup assertions, and cancellation-grace mutations.
2. **Cold-agent continuity discovery transcript** — run a fresh external model from the sparse questions using only MCP discovery/errors/catalog evidence, then compare its selected plan and cited answer with the same oracle.
3. **`z9gh.7` live corpus/effect adapter** — create the authorized real-incident manifest, membership/effect evidence projections, scope distinctions, privacy review, latency/memory measurement, and terminal mandate disposition.
4. **Run-unit aggregate support** — add production aggregate lowering for `runs | count`, then require the same count-probe/enumeration invariant for structural lineage populations.

## Value of another iteration

A small repair iteration would likely address only CI, Python/MCP SDK, or packaging compatibility discovered in the target environment; the implementation is already cohesive and independently apply-tested.

A substantial third pass remains valuable only if it is allowed to exercise the missing terminal-gate mechanisms: transport cancellation, a cold external model, the authorized live incident/effects corpus, and live SLO measurement. That work would materially advance `polylogue-z9gh.7`; another synthetic declaration-only refinement would add little.
