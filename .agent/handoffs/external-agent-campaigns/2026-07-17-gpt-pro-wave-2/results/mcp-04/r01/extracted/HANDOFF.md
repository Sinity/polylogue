# MCP-04 query discovery corpus handoff

## Mission and delivered result

This package implements the parser-truthful discovery slice requested for `polylogue-z9gh.3`: one typed executable corpus, a production-parser anti-lying gate, result-semantics declarations, and rewiring of the highest-risk MCP/CLI/docs teaching routes. The patch is designed to apply directly to the supplied Polylogue snapshot and deliberately makes no grammar change.

The implementation adds 106 positive examples and 18 negative examples in one typed data module. Every positive row carries the expression, parser route, unit source, one-sentence answer, one of six result-semantics classes, projected public columns, selective/corpus-scale cost class, and execution route. Every negative row carries the actual production exception class, exact diagnostic text and field, a parser-valid correction, and any shipped snapshot location. The rows are provider-neutral and privacy-safe.

The completed slice is internally coherent and production-routed. It does not claim to complete all of the broader `z9gh.3` declaration program: typed structured-plan lowering, OpenAPI/JSON schema generation, live coverage/freshness/cardinality, current-value discovery, and the full six-tool `explain` transaction remain separate work.

## Snapshot identity and patch base

- Source snapshot manifest: `/mnt/data/work_polylogue/polylogue/polylogue-manifest.json`
- Manifest generated at: `2026-07-17T180950Z`
- Source recorded by manifest: `/realm/project/polylogue`
- Branch: `master`
- Commit: `536a53efac0cbe4a2473ad379e4db49ef3fce74d`
- Commit subject: `fix(repair): harden raw authority convergence (#3046)`
- Manifest dirty flag: `true`
- Exported branch-delta file list, log, and patch: zero bytes
- Extracted Git worktree before this implementation: no tracked or untracked project delta

Because the manifest's dirty flag had no exported branch delta and the extracted repository matched the named commit, `PATCH.diff` targets commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d` exactly. No unstated operator worktree patch is assumed.

Relevant history inspected:

- `9163d0134f3d334960e4c249c96c5671919a9a06` — `feat(query): bound agent-facing archive reads (#3018)`
- `ed44be18f448c31f9fa5b9289c75da7eee99b131` — `feat(mcp): declare the current tool algebra (#3004)`
- Current `536a53ef...` snapshot and surrounding default-branch history

## Authority inspected

The production route and teaching surfaces were traced through:

- `polylogue/archive/query/expression.py`: real Lark grammar, terminal priorities, `compile_expression`, `parse_unit_source_expression`, and the hand-written quote/parenthesis-aware pipeline splitter.
- `polylogue/cli/query_group.py`: strict bare-root command floor and query-intent rules.
- `polylogue/archive/query/metadata.py`: executable unit descriptors, terminal source inventory, fields, lowerers, aggregate capability, and examples.
- `polylogue/archive/query/transaction.py`: existing bounded read transaction boundary and the natural home for shared result coverage/total vocabulary.
- `polylogue/archive/query/completions.py`: existing executable field/unit completion machinery and public completion payload.
- `polylogue/mcp/declarations/`: the current MCP declaration pilot and `MCPResultSemantics` vocabulary.
- `polylogue/mcp/server_prompts.py` and `server_resources.py`: all six registered cookbook prompts and the query capability resource.
- `docs/search.md`, generated `docs/cli-reference.md`, CLI help declarations, and query-completion surfaces.
- Public row models in `polylogue/surfaces/payloads.py`.
- Existing query, CLI, MCP, generated-surface, topology, and envelope tests.
- Root `CLAUDE.md`, including the requirement to regenerate topology projection/status after adding a module under `polylogue/`.
- Beads `polylogue-z9gh.3`, `polylogue-z9gh.9.1`, and `polylogue-t46.8.1`, including later notes superseding older curriculum assumptions.
- Repository-owned skill/manual candidates. No `SKILL.md` exists in the snapshot; the two campaign manual-install Markdown files contain no query syntax. The separately installed shared skill cited in Beads is outside the supplied snapshot and was not available for verification.

## Mechanism

### One typed corpus

`polylogue/archive/query/discovery.py` is the new executable declaration source. It contains:

- 106 positives, within the required 80–150 range.
- 18 negatives, within the required 10–20 range.
- All six semantics classes: 63 exhaustive, 9 top-k, 6 sample, 16 aggregate, 7 bounded-context, and 5 recursive-page.
- Every terminal source plus sessions: sessions 48, messages 14, actions 8, blocks 5, files 6, assertions 7, runs 4, observed-events 6, context-snapshots 3, and delegations 5.
- Both cost classes: 62 selective and 44 corpus-scale.
- 19 featured examples and 6 safely parameterized prompt examples.
- Compact selectors, Boolean session predicates, structural `exists`, sequence, semantic/FTS forms, every terminal unit, scoped pipelines, sorting/limit/offset, aggregates, projections, bounded context, sampling routes, and recursive lineage seeds/pages.

Dynamic recipe values are rendered through typed parameter declarations. Text is JSON-quoted; simple value/date tokens remain compact; unsafe whitespace or punctuation is quoted; empty/NUL values and multiline non-text values are rejected. Prompt tests use embedded quotes, spaces, and realistic dates and then parse the rendered expression through production code.

### Real parser/splitter gate

`tests/unit/archive/query/test_discovery.py` sends every positive row through one of the actual production entry points:

- session expressions: `compile_expression`
- terminal/scoped/pipeline expressions: `parse_unit_source_expression`

No mock grammar or replica parser exists. The second route necessarily exercises `_split_pipeline_stages`, terminal source parsing, and pipeline-stage application. Every negative row is required to raise the real `ExpressionCompileError` with the exact declared text and field; its correction must then parse through the same production route.

The gate also binds declared exhaustive/aggregate projection columns to the real public Pydantic payload model fields, checks all required units/semantics/costs, verifies provider/privacy neutrality, validates parameter quoting, and proves that catalog/completion payloads project the same rows.

### Shared result-semantics teaching

`polylogue/archive/query/transaction.py` now declares the shared coverage/total/continuation vocabulary and `QueryResultSemanticsContract`. The corpus defines the exact six discovery phrases:

- exhaustive — exact total; physically paged; follow continuation until absent.
- top-k — relevance frontier, not exhaustive; total is qualified.
- sample — not exhaustive; qualified total cannot establish archive-wide coverage.
- aggregate — totals describe buckets over the declared input relation; input coverage is separate.
- bounded-context — orientation context, not exhaustive; omissions are expected.
- recursive-page — recursively related nodes/edges are physically paged; totals remain qualified until all continuations end.

The MCP query capability resource explicitly maps these six classes to the existing MCP declaration enum: `exhaustive_page`, `top_k`, `sample`, `aggregate`, `bounded_context`, and `recursive_graph`. This avoids creating a second MCP semantics taxonomy.

### Discovery and teaching rewiring

- `query_completions(kind="example")` now returns corpus-backed parser-valid examples, semantics language, projections, and cost class.
- `query_completions(kind="error")` now returns real rejected forms, actual diagnostics, and parser-valid insertion corrections.
- The MCP query capability resource is version 2 and publishes grammar forms, corpus counts, completion routes, six semantics contracts, per-unit executable examples, and the MCP declaration mapping while remaining under the 25,000-byte response budget.
- The four cookbook prompts that contain query/search expressions now render from parameterized corpus rows. The other two preserved cookbook prompts contain no query expression.
- Root CLI help resolves its three query examples from stable corpus keys before Click renders help.
- `docs/search.md` contains a generated corpus section with forms, source inventories, semantics wording, 19 featured positives, the shipped-invalid census, and completion routes.
- The new `devtools render query-discovery` command is registered in the command catalog and generated-surface registry. `docs/devtools.md`, `docs/cli-reference.md`, topology projection, and topology status are regenerated.
- A durable docs audit parses 46 concrete `docs/search.md` commands and 41 generated `polylogue find` examples. Non-vacuity thresholds prevent an extraction bug from turning the audit green with no examples.

Some legacy detail examples in `docs/search.md` and command-specific help remain hand-authored; they are now production-parser gated rather than all being string-generated from corpus keys. Completing literal single-source generation for every remaining help line is a meaningful follow-on, not hidden as completed here.

## Shipped-invalid-example census

The following rejected expressions were present at the supplied snapshot boundary. Locations refer to commit `536a53ef...`, not post-patch line numbers.

1. `polylogue/mcp/server_prompts.py:509`
   - Shipped: `actions where session.repo:example-repo since:7d AND output:failed`
   - Actual diagnostic: `invalid query expression near column 27`
   - Cause: missing `AND` and terminal session date fields require the `session.` prefix.
   - Correction: `actions where session.repo:example-repo AND session.since:7d AND output:failed`

2. `polylogue/mcp/server_prompts.py:524`
   - Shipped: `files where repo:example-repo AND path:src/mcp/server.py`
   - Actual diagnostic: `field 'repo' is not supported for file predicates`
   - Cause: repository scope is a session predicate inside terminal queries.
   - Correction: `files where session.repo:example-repo AND path:src/mcp/server.py`

3. `docs/search.md:924`
   - Shipped snapshot form: `text:css {session_id claude-code}: refactor`
   - Privacy-neutral corpus fixture: `text:css {session_id example}: refactor`
   - Actual diagnostic: unknown query field `text`, followed by the production recognized-field list.
   - Cause: raw FTS5 column-query syntax was taught at the strict public command floor even though the DSL does not expose `text` as a session field.
   - Correction: `contains:"css refactor"`

A broad pre-patch shell/prose census examined 75 query-bearing snippets, including variants that are not suitable for direct test parameterization. The durable executable gate retains 46 concrete search-guide commands, all 41 generated CLI `find` commands, every corpus row, and every emitted cookbook query. Only the three failures above were found in the named shipped teaching surfaces.

## Grammar/capability gaps recorded without grammar changes

No intended positive corpus expression remains rejected by the current parser. The following current capability gaps are deliberately represented as diagnostics or handoff findings rather than grammar extensions:

- `sessions where ... | count` has no session terminal aggregate lowerer. Count matching sessions through the CLI analyze route, or scope an executable terminal source before `| count`.
- `runs` and `context-snapshots` have no aggregate lowerer. Their corrections use supported sorting/limit forms.
- Ranked/semantic predicates cannot be nested below Boolean `NOT`; the production diagnostic is pinned.
- Structural projection is limited to declared `with <unit>(<columns>)` units/columns.
- Result classes such as sample, bounded context, and recursive page are execution-route semantics layered over parser-valid selectors. Lark acceptance alone does not establish those coverage claims.

## Changed files

Production declarations and routes:

- `polylogue/archive/query/discovery.py` — typed corpus, parameter rendering, semantics, payload projections.
- `polylogue/archive/query/transaction.py` — shared coverage/total/continuation contract vocabulary.
- `polylogue/archive/query/completions.py` — corpus-backed `example` and `error` completion kinds.
- `polylogue/mcp/server_prompts.py` — corpus-rendered cookbook queries.
- `polylogue/mcp/server_resources.py` — bounded query capability v2 and MCP semantics bridge.
- `polylogue/cli/click_app.py` — corpus-backed root-help examples.

Generation and documentation:

- `devtools/render_query_discovery.py`
- `devtools/command_catalog.py`
- `devtools/generated_surfaces.py`
- `docs/search.md`
- `docs/cli-reference.md`
- `docs/devtools.md`
- `docs/plans/topology-target.yaml`
- `docs/topology-status.md`

Tests:

- `tests/unit/archive/query/test_discovery.py`
- `tests/unit/cli/test_query_discovery_help.py`
- `tests/unit/devtools/test_query_teaching_surfaces.py`
- `tests/unit/devtools/test_render_query_discovery.py`
- `tests/unit/mcp/test_envelope_contracts.py`
- `tests/unit/mcp/test_server_surfaces.py`

`FILES/` is omitted because the binary-capable unified patch carries every addition and modification unambiguously, including the generated topology files.

## Acceptance matrix

| Requirement | Result | Evidence |
|---|---|---|
| 80–150 typed positives with required columns | Complete | 106 typed rows; shape and projection tests |
| 10–20 negatives with actual diagnostics/corrections | Complete | 18 exact `ExpressionCompileError` fixtures; corrections parse |
| Every positive uses real Lark grammar + pipeline splitter | Complete | Parameterized anti-lying test over all 106 rows |
| Every negative fails with documented diagnostic class | Complete | Exact class/text/field assertions over all 18 rows |
| Audit named teaching surfaces | Complete for supplied snapshot | MCP prompts, search guide, generated CLI reference, completions, repo skill/manual candidates |
| Rewire teaching strings to declarations | Substantial, not universal | MCP recipes, central search discovery, root CLI examples, completion/error catalog, capability resource are derived; remaining legacy detail/help examples are parser-gated but hand-authored |
| Exact six-class semantics wording | Complete as static discovery contract | Transaction vocabulary, generated docs, MCP resource, tests |
| Align live page totals/coverage on every read surface | Outside this patch | Full `z9gh.9.1` transaction migration remains open |
| No grammar change | Complete | `expression.py` unchanged |
| Provider-neutral/privacy-safe corpus | Complete | Automated token/email/privacy checks |
| Generated surfaces and topology updated | Complete | Four renderer checks and clean fresh-worktree apply |

## Apply order

From a checkout of commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d`:

```bash
git apply --check PATCH.diff
git apply PATCH.diff
```

Then run the commands in `TESTS.md`. `PATCH.diff` uses Git binary patch records for the topology projection/status files, so use `git apply`, not a text-only patch utility.

## Verification performed

- Binary-capable patch generated against the named commit.
- `git apply --check` and `git apply` succeeded in a fresh detached worktree at the exact commit.
- Four registered generated-surface checks passed in the fresh worktree.
- Ruff formatting, Ruff lint, and `git diff --check` passed in both development and fresh-apply trees.
- The focused production parser/metadata/CLI/docs suite passed: **678 passed, 1 skipped**.
- The affected MCP capability and prompt tests passed: **9 passed**.
- Python bytecode compilation passed for production, generator, and added tests in the development tree.
- Full ZIP validation, SHA-256, and byte size are performed after these documents are written; the final chat report is the authority for those package values.

## Risks, limitations, and next value

Important limitations:

- The supplied project's full locked environment could not be created because the package mirror denied the lock's future-dated `types_dateparser-1.4.1.20260617` wheel. Tests used the container's `/opt/pyvenv` with the project installed and missing test dependencies added.
- Running the complete `test_envelope_contracts.py` plus `test_server_surfaces.py` together showed no failure through 61% but exceeded a 300-second command ceiling. The nine directly affected cases pass; the full two-file result is unverified.
- No live daemon, browser, populated archive, secrets, NixOS deployment, or operator worktree was accessed.
- The separately installed shared Polylogue skill named in Beads was not part of the snapshot, so installed-skill parity remains unverified.
- The semantics declarations make discovery truthful, but the open shared transaction work must still make every live adapter emit the corresponding exact/qualified totals and continuations.
- Remaining hand-written detail examples are parser-gated, not all declaration-rendered.

A small repair iteration could add little beyond resolving the lock/mirror issue and completing the slow full MCP files under the project's intended environment. A substantial second pass remains valuable: migrate every residual help/docs example to corpus keys, generate structured-plan/OpenAPI schemas and valid-value errors, add observed coverage/freshness/cardinality and paged catalog search, wire the same semantics into the live `QueryResultPage`/six-tool `explain` transaction, and parity-check or generate the installed skill. That is a separate declaration/transaction expansion rather than a defect in this apply-ready corpus slice.
