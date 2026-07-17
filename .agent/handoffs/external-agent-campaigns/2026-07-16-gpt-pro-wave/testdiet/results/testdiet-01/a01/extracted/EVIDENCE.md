# Source, Bead, test, and history evidence

## Snapshot authority

The supplied project-state manifest identifies `/realm/project/polylogue`, branch `master`, commit `f654480cadb7cc4c194704e24dfd483199547b35`, generated at `2026-07-17T043202Z`, with `dirty: true`.

The supplied branch-delta report identifies `origin/master` with merge base `f654480...`, no commits, no changed files, and an empty patch. Reconstructing a detached worktree at the commit and overlaying the captured `polylogue-working-tree.tar.gz` produced no Git status changes. Thus the exact content authority is the named commit/captured tree, while the dirty flag remains an unexplained snapshot-metadata discrepancy.

The attached Test Diet exact-query dossier identifies the older head `21f78b4db2ba62ff44b5f16dfab96067bc249b4c` and says “Realized artifacts: 0.” Its proposed filenames remain useful, but its paths, readiness assumptions, and API descriptions cannot override the newer source snapshot.

## Repository architecture findings

`AGENTS.md` resolves to `CLAUDE.md`. The repository instructions state that substrate owns meaning and surfaces are leaf adapters; new semantics belong in storage/product before surfaces. They also identify `polylogue/archive/query/expression.py` as the real Lark query DSL and require focused `devtools test` use rather than blanket directory runs. `CONTRIBUTING.md` and `pyproject.toml` establish strict mypy and Ruff obligations.

The current query path is not one function. The inspected chain is:

```text
Lark expression
  -> parsed AST / compiled query spec
  -> archive execution or terminal-unit source
  -> SQLite session/action relation
  -> repository reader/count executor
  -> Polylogue facade and public envelopes
  -> root CLI read/action renderer
```

The daemon session-page fast-path support check rejects a compiled Boolean predicate. Therefore the Boolean root CLI law reaches the local repository path repaired by this patch rather than obtaining a false pass from a daemon adapter.

## Production defect evidence

In `_execute_archive_query_stdout`, the code constructs `_ArchiveFilterKwargs` from all compiled filters and conditionally adds `boolean_predicate`. Search, stats, count, mutation, and other branches consume that map or explicitly forward the predicate. The final bare-list branch instead repeated the individual keyword list and omitted `boolean_predicate`.

Commit `5d5edaf496d70b9372c7c2123c2e70a6ed4d34e6` repaired two count call sites by forwarding `boolean_predicate`; it did not repair the bare `list_summaries` call. This explains the current contradiction with closed Bead `polylogue-70qb`: the Bead title and acceptance criteria require bare explicit Boolean list parity, while its close reason describes only the count-call fix. The new survivor exposed that residual implementation gap directly.

The final patch uses the already-typed map:

```python
summaries = archive.list_summaries(
    limit=fetch_limit,
    offset=page_offset,
    sample=sample_count is not None,
    sort=sort,
    reverse=reverse,
    **filter_kwargs,
)
```

This preserves all previous filter keys, includes `boolean_predicate` only when compiled, and removes a second manual forwarding inventory.

## Relational cardinality evidence

The canonical action relation ranks tool-use and tool-result blocks by `(session_id, tool_id)` and transcript ordering, then pairs equal ranks. It left-preserves tool uses without results. That implementation reflects the historical defect and repair recorded in:

- `7b5a5aa0589772ed9d7eb9632af1591cf22cdd33` — pair actions by transcript rank rather than plain tool-ID equality;
- `9f0e77116280571c14430b47986e7bd5b0e8be1a` — closure evidence for `polylogue-xnkf`.

The planted manifest reproduces the discriminating shape without copying a production answer: two uses plus two results share one tool ID, and one additional use is unmatched. Rank pairing yields three logical Bash rows. Equality joining yields four repeated-pair combinations plus the unmatched row, for five. The test asserts the exact wrong result.

The current action query implementation also supports a bounded exact-session relation that pushes selected session IDs into both ranked branches. Existing C-03 tests prove semantic equality and a work-bound witness against a global-first mutant. The new law reuses that generated archive and receipt rather than inventing a parallel scale framework.

## Test/helper audit

The affected owning cluster was inspected in full. A rough AST/source census used only to understand proof form—not to authorize deletion—found:

| File | Lines | Test functions | Mock/patch/fake signal |
| --- | ---: | ---: | ---: |
| `tests/unit/cli/test_query_expression.py` | 6,494 | 328 | 2 |
| `tests/unit/cli/test_query_exec_laws.py` | 3,552 | 66 | 46 |
| `tests/unit/cli/test_verb_cardinality.py` | 900 | 39 | 26 |
| `tests/unit/storage/test_archive_tiers_archive.py` | 1,483 | 24 | 1 |

The expression suite preserves unique grammar, precedence, quoting, sequence, diagnostics, identity, and execution cases. The execution/cardinality suites contain valuable behavior plus many adapter-forwarding witnesses. The storage file owns the realized C-03 semantic/work-bound tests.

Shared helpers inspected:

- `tests/infra/query_cases.py` — a test-authored query case model;
- `tests/infra/surfaces.py` — surface adapters and `_query_case_to_spec`, a partial shadow lowering path;
- `tests/infra/semantic_facts.py` — shared semantic expected facts;
- `tests/infra/workload_artifacts.py` — content-addressed C-03 artifact, facts, manifest/receipt, and clone support.

The new oracle does not replace those helpers globally. It is deliberately small and local to this responsibility.

## Beads evidence

| Bead | Status | Constraint or intent used |
| --- | --- | --- |
| `polylogue-1xc.14.1` | in progress | Real provider-native generated workloads, selective C-03 query, shared identities/receipts; do not invent another corpus framework. |
| `polylogue-b054.1.1.4` | open | A trustworthy affected gate ultimately needs a real production mutation and actual testmon selection. This package proves the direct mutation but does not claim testmon closure. |
| `polylogue-xnkf` | closed | Plain action joins fan out duplicate provider-emitted tool IDs; one logical use must remain one row. |
| `polylogue-yeq.3` | open | Cross-surface query laws require identity, completeness, partitions, page concatenation, duplicate/missing/late shapes, and mutation sensitivity. |
| `polylogue-z9gh.2` | open | Selective action/delegation relations must avoid archive-wide materialization while preserving Nth-use/Nth-result and missing-result semantics. |
| `polylogue-fnm` | open | One Lark grammar owns query semantics; no parallel Test Diet query language or verb was added. |
| `polylogue-z9gh.9.1` | open | Full bounded shared transactions and lossless continuation across every read surface remain broader work. |
| `polylogue-70qb` | closed | Acceptance says bare explicit Boolean list must agree; close evidence only repaired count calls. Current source retained the bare-list omission, now fixed here. |

Later/current Bead notes and source were treated as stronger than the stale Test Diet prose wherever they conflicted.

## Historical findings

| Commit | Finding relevant to this patch |
| --- | --- |
| `5d5edaf496d70b9372c7c2123c2e70a6ed4d34e6` | Fixed Boolean forwarding for count paths only, leaving the list-path gap. |
| `7b5a5aa0589772ed9d7eb9632af1591cf22cdd33` | Established transcript-rank action pairing. |
| `9f0e77116280571c14430b47986e7bd5b0e8be1a` | Closed the duplicate action fanout issue after the rank-pair repair. |
| `89166362b9aee8c304b27a69f68ec1b74606f634` | Landed the production query evaluator and public query route. |
| `478d6a77cf1cea1d63c180701d49099ad2cfacc3` | Added exact-session selective temporal bounding. |
| `c6495ea29c92d54dae00b302cb17336d80b5162d` | Established the shared receipt-backed query canary. |
| `c20286459cf2c3d1e4c968a8584f13e7cd382ff2` | Added workload profile/seeded artifact infrastructure used by the fixture. |

## Dominated candidates for later certification

No deletion is included. The following are candidates only after independent local route/oracle/sensitivity certification proves diagnostic and compatibility dominance:

- mock-forwarding permutations in `tests/unit/cli/test_query_exec_laws.py` that only restate collaborator keyword plumbing;
- repetitive single-field lowerer examples in `TestLowererFieldMapping`, potentially consolidated into a reviewed decision table while retaining unique diagnostics;
- source-spelling and call-path tests in `TestSharedCardinalityPath` once behavior and removal mutations dominate them;
- shadow query maps/translators in `tests/infra/query_cases.py` and `tests/infra/surfaces.py` after all their represented fields and surfaces migrate to production-expression laws;
- overlapping parts of `tests/infra/semantic_facts.py` only after a broader independent manifest covers the same obligations.

Required certification before any deletion: run the proposed survivor and the candidate locally; remove or mutate the production dependency; prove the survivor fails for the historical defect; confirm the candidate adds no unique parser diagnostic, security, compatibility, payload, or branch evidence. Until that proof exists, retain the tests and helpers.

## Contradictions and unresolved boundaries

1. The snapshot manifest says dirty, but the captured tree equals the named commit.
2. The stale query dossier reports no realized artifact and an older head, while current source contains C-03 workload/receipt support.
3. Closed Bead `polylogue-70qb` names bare-list parity, but its implemented/recorded fix only forwarded Boolean predicates to count paths.
4. The local host's default `pytest-randomly` plugin fails before test setup; managed runs explicitly disable it.
5. Full quick verification did not complete at full-repository mypy; only completed checks are reported as passes.
6. Daemon/HTTP/MCP/browser and live deployment are not represented by this package and remain explicitly unverified.
