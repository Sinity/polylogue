# Source, Beads, history, and contradiction evidence

## Snapshot authority

The supplied archive manifest records:

- generated: `2026-07-17T180950Z`
- source: `/realm/project/polylogue`
- Git branch: `master`
- Git commit: `536a53efac0cbe4a2473ad379e4db49ef3fce74d`
- dirty: `true`

The same archive contains zero-byte `polylogue-branch-delta-files.txt`, `polylogue-branch-delta-log.txt`, and `polylogue-branch-delta.patch`. The extracted Git worktree was at the recorded commit with no project delta before implementation. The patch therefore names and targets that commit, while explicitly preserving the manifest contradiction rather than inventing missing dirty bytes.

## Parser and command-floor findings

`polylogue/archive/query/expression.py` is the grammar authority:

- `_QUERY_GRAMMAR` accepts compact clauses, Boolean `sessions where` predicates, structural `exists`, sequence predicates, semantic/near and FTS leaves, counts/date/time comparisons, and field clauses.
- `FIELD_CLAUSE` has terminal priority 4. Colon-bearing specialized terminals are declared at higher priority where required.
- `_split_pipeline_stages` is hand-written outside Lark and splits only at top-level `|`, preserving quotes and parentheses such as `origin:(a|b)`.
- `parse_unit_source_expression` first parses scoped/direct terminal sources and then applies hand-parsed pipeline stages.
- `compile_expression` remains the production session expression entry point.

`polylogue/cli/query_group.py` implements the strict bare-root command floor from the #1842 behavior: plain unquoted words can be treated as command mistakes, while quoted multiword or colon-bearing terms signal query intent. This is why raw internal FTS column-query examples cannot be treated as a separate public syntax.

No changes are made to either parser or command floor.

## Existing declaration direction

`polylogue/archive/query/metadata.py` already owns terminal unit descriptors, fields, examples, lowerer kinds, aggregate grouping, stable sorting, and structural capabilities. `polylogue/mcp/declarations/models.py` already owns MCP result classifications including exhaustive page, top-k, sample, aggregate, bounded context, and recursive graph. `registry.py` already declares the bounded target read algebra.

The patch extends this direction:

- It does not replace unit metadata.
- It uses public payload models as projection truth and tests declaration/model parity.
- It declares shared generic coverage/total/continuation words in `archive/query/transaction.py`.
- It maps the six generic classes to existing `MCPResultSemantics` values in the capability resource.
- It registers docs generation through the existing command and generated-surface registries.

## Beads intent and supersession

### `polylogue-z9gh.3`

The current Bead says the failures are one declaration problem: schemas, discovery, errors, completions, projections, docs, and recipes should be generated from executable query declarations. Its later notes establish three controlling facts:

- The installed skill and shipped prompts taught sessions-only/nonterminal or otherwise invalid `query_units` forms, so curriculum parity is a P0 active regression.
- Discovery must classify exhaustive page, top-k, sample, aggregate, bounded context, and recursive graph/page semantics rather than letting a `limit` imply totality.
- The 2026-07-16 GPT-Pro corpus is seed input only; it must share `z9gh` execution semantics and cannot become a second per-surface registry.

This patch implements the parser-gated corpus/static discovery slice while leaving the wider registry/coverage/structured-plan program open.

### `polylogue-z9gh.9.1`

The shared query transaction Bead owns exact versus qualified totals, page/continuation state, and one execution boundary across read surfaces. Its notes document live oversized responses and non-progressing continuation behavior. The semantics declarations in this patch align vocabulary with that work but do not claim to land its executor migration.

### `polylogue-t46.8.1`

The MCP declaration pilot requires every tool/resource/prompt declaration to state result semantics and eventually generate discovery/manual/equivalence artifacts. The existing pilot is present in the snapshot. The patch bridges to it rather than introducing an alternative MCP enum.

## Relevant history

- `ed44be18f448c31f9fa5b9289c75da7eee99b131` — declaration pilot: `feat(mcp): declare the current tool algebra (#3004)`.
- `9163d0134f3d334960e4c249c96c5671919a9a06` — bounded reads: `feat(query): bound agent-facing archive reads (#3018)`.
- `536a53efac0cbe4a2473ad379e4db49ef3fce74d` — supplied snapshot head.

The current snapshot already contains the #3018 bounded-query work and corrected older sessions-only recipe shapes described in some Beads prose, but it still shipped the two concrete terminal-field/conjunction errors listed below and had no corpus-wide real-parser gate.

## Shipped contradiction evidence at the snapshot commit

### MCP prompt 1

Snapshot location: `polylogue/mcp/server_prompts.py:509`.

`git show 536a53ef:polylogue/mcp/server_prompts.py`, line 509:

```text
query_units(expression='actions where session.repo:{repo_name} since:{since} AND output:failed', limit=20)
```

Neutral concrete fixture:

```text
actions where session.repo:example-repo since:7d AND output:failed
```

Production result:

```text
ExpressionCompileError: invalid query expression near column 27
```

Parser-valid correction:

```text
actions where session.repo:example-repo AND session.since:7d AND output:failed
```

### MCP prompt 2

Snapshot location: `polylogue/mcp/server_prompts.py:524`.

`git show 536a53ef:polylogue/mcp/server_prompts.py`, line 524:

```text
query_units(expression='files where {repo_clause}path:{path}', limit=20)
```

With a repository clause, the neutral concrete fixture is:

```text
files where repo:example-repo AND path:src/mcp/server.py
```

Production result:

```text
ExpressionCompileError(field='repo'): field 'repo' is not supported for file predicates
```

Parser-valid correction:

```text
files where session.repo:example-repo AND path:src/mcp/server.py
```

### Search guide

Snapshot location: `docs/search.md:924`.

`git show 536a53ef:docs/search.md`, line 924:

```text
polylogue 'text:css {session_id claude-code}: refactor'
```

A privacy-neutral fixture preserving the same syntax is:

```text
text:css {session_id example}: refactor
```

Production result: `ExpressionCompileError(field='text')` with the production unknown-field diagnostic and recognized-field list.

Parser-valid public correction:

```text
contains:"css refactor"
```

## Teaching-surface audit

Named surfaces inspected:

- all six MCP cookbook prompts; four emit query/search expressions and are now corpus-rendered and parser-tested; two emit no query expression.
- `docs/search.md`; generated corpus block plus a durable gate over 46 concrete commands.
- generated `docs/cli-reference.md`; 41 concrete `polylogue find` expressions parse through production code.
- root CLI help; three high-signal query examples resolve from corpus keys.
- `query_completions`; existing field/unit candidates already derive from executable metadata, while new example/error kinds derive from the corpus.
- MCP query capability resource; grammar/count/semantics/unit examples are declaration-projected and response-budget tested.
- repository skill/manual text; no `SKILL.md`; two campaign manual-install files contain no query syntax.

A temporary broad pre-patch census included 75 query-bearing shell/prose snippets and found the one rejected search-guide form above. The durable tests intentionally use executable command extraction plus non-vacuity assertions; generated corpus rows are covered by their own complete gate.

Residual caveat: many detailed docs and command-specific help lines remain literal strings. They all pass the production parser audits in this patch, but only the central discovery section, MCP recipes, root CLI examples, completion catalog, and capability resource are directly generated/rendered from corpus keys.

## Corpus evidence

Final declaration counts:

```text
positive: 106
negative: 18
featured: 19
parameterized: 6

semantics:
  exhaustive: 63
  top-k: 9
  sample: 6
  aggregate: 16
  bounded-context: 7
  recursive-page: 5

cost:
  selective: 62
  corpus-scale: 44

parser route:
  session: 48
  unit/pipeline: 58
```

All ten sources are represented. Every semantics class has at least five positive examples. The corpus's provider/privacy test rejects model/provider names, local home paths, email addresses, and multi-sentence answers.

## Grammar-gap/capability findings

No grammar was extended. The following are recorded current limitations:

- A sessions scoping stage has no direct aggregate lowerer, so `sessions where ... | count` is rejected with a detailed route correction.
- Run and context-snapshot rows have no aggregate lowerer.
- Semantic/ranked leaves under Boolean `NOT` are not supported.
- `with` projections are limited to declared units/columns.
- Sampling, context construction, and recursive graph paging are execution semantics associated with parser-valid selector declarations; the parser does not itself prove archive coverage or continuation behavior.

These findings are represented in the negative corpus or handoff rather than hidden by grammar changes.

## Repository instructions and generated topology

Root `CLAUDE.md` requires topology projection/status regeneration whenever a module is added under `polylogue/`, and requires new devtools commands to update the command catalog/reference. The patch adds `polylogue/archive/query/discovery.py` and `devtools/render_query_discovery.py`, registers the command and generated surface, and includes regenerated:

- `docs/plans/topology-target.yaml`
- `docs/topology-status.md`
- `docs/devtools.md`
- `docs/cli-reference.md`

All renderer `--check` commands pass after clean patch application.
