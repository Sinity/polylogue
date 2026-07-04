# Query Set-Algebra Design

Status: **design (hold — do not implement yet)** · Owner bead: `polylogue-fnm.13` ·
Enabler: `polylogue-fnm.9` (pipeline-as-subquery) · Anchor: `polylogue-4p1`
(one read algebra: Query × Projection × Render)

This document designs **set operations over query results** — combining the
*result sets* of two independent queries with `union` / `intersect` / `except`.
It is deliberately thorough because the `fnm` DSL is small on purpose and the
ranking/grain semantics have non-obvious corners.

---

## 1. Motivation and the precise gap

The grammar (`polylogue/archive/query/expression.py:623`, `_QUERY_GRAMMAR`)
composes **predicates within one query** via `and` / `or` / `not`. Those are
*row-level boolean* operators over a single retrieval: `auth and test` keeps rows
where both predicates hold.

There is **no way to combine the result sets of two independent queries.** That
is strictly more expressive, because the two operands can be:

- **different retrieval lanes** — `semantic:"db migration"` (vector) vs `~keyword`
  (FTS) vs `origin:codex date>2026-06-01` (structural). A single row-predicate
  cannot mix a vector-similarity set with an FTS set with a structural set.
- **different grains** — a set of sessions vs a set of messages.
- **named cohorts / macros** (`fnm.12`) — `@arm_pack` vs `@arm_raw`.
- **independently expensive computations** — two subqueries each with their own
  pipeline, intersected at the end.

Concrete questions it unlocks, none expressible today:

```
sessions that discussed auth but never tests
  auth  except  test
messages semantically about migrations, excluding week 1
  semantic:"schema migration"  except  week:2026-W01
the symmetric difference of two experiment arms
  (@arm_pack except @arm_raw)  union  (@arm_raw except @arm_pack)
sessions touching file X that also cost > $5
  touches:file.py  intersect  cost:>5
```

---

## 2. The algebra: it is relational, not three ad-hoc mechanisms

The key reframe. Every read construct is a function **relation → relation**:

| Construct | Arity | Signature |
|---|---|---|
| a query (`and/or/not` predicates, a retrieval lane) | base | `() → R` |
| **set-op** `union` / `intersect` / `except` | binary | `R × R → R` |
| pipeline stage (`group by`, `fields`, `read`, `context-image`, aggregates) | unary | `R → R` (or `R → A` for terminal/aggregate) |

So we do not need a new subsystem — we need to admit the **binary** operator into
the one relation algebra that `4p1` already names. A "relation" `R` is a set of
rows at a declared **grain** with an optional **rank/score** column.

### 2.1 Grain and identity

`fnm.13` targets **both** session-grain and message/unit-grain (operator
decision, 2026-07-05).

| Grain | Identity key | Notes |
|---|---|---|
| session | `session_id` | the clean base case |
| message | `(session_id, message_id, variant_index)` | `variant_index` disambiguates content variants (`4smp`) |
| block / unit | `(session_id, message_id, block_index)` | for `block`/`action`/`observed-event` units (`fnm.10`, attached units) |

Set-ops are defined on the **identity key** of the operand grain. Two rows are
"the same element" iff their identity keys are equal.

---

## 3. Operator semantics

Let `A`, `B` be relations at the **same grain** with identity keys `key(·)`.

- **`A union B`** — every element in `A` or `B`, deduplicated by `key`.
- **`A intersect B`** — elements whose `key` appears in both.
- **`A except B`** — elements of `A` whose `key` does *not* appear in `B`.

`union` and `intersect` are commutative and associative; `except` is neither
(left-biased). Parentheses are mandatory to disambiguate mixed chains
(`(A intersect B) except C`), matching SQL's lack of cross-op precedence.

### 3.1 Ranking / scoring — the subtle part

Query results are often **ranked** (FTS bm25, vector similarity, hybrid RRF).
Set-ops must decide the rank of the output, and naive choices produce misleading
orderings because scores from different lanes are **not comparable**.

Policy (chosen for honesty + determinism):

- **`intersect`** — keep the elements' rank from the **left** operand `A`
  (the right operand is a filter, not a ranker). Output order = `A`'s order
  restricted to the intersection. Rationale: the user reads left-to-right; the
  left query is the "primary" ranked search, the right is a membership gate.
- **`except`** — trivially `A`'s order restricted to the complement.
- **`union`** — operands may come from **incomparable score scales** (vector
  cosine vs bm25), so we do **not** compare raw scores. Fuse by **Reciprocal
  Rank Fusion** over the two operands' rank positions (reuse
  `storage/search_providers/hybrid.py:reciprocal_rank_fusion`, `k=60`), with the
  deterministic `(-fused, key)` tie-break already implemented there. This is the
  same principle hybrid search already uses; union of two ranked sets is exactly
  a 2-list RRF. Unranked operands (pure structural filters) contribute rank =
  insertion order.

This means **`union` is rank-honest** (never claims a cosine 0.8 outranks a bm25
12.0) and **`intersect`/`except` preserve the primary search's ranking**. Both
are documented as part of the `SearchEnvelope` contract.

### 3.2 Empty / NULL / self

- Empty operand: `A intersect ∅ = ∅`, `A union ∅ = A`, `A except ∅ = A`,
  `∅ except A = ∅`. No error.
- Self: `A intersect A = A` (dedup), `A except A = ∅`. Allowed; the planner may
  short-circuit but semantics stand.
- A row with a NULL sort key sorts last within its operand (mirrors the existing
  `(occurred_at_ms IS NULL)` ordering) before fusion.

---

## 4. Grain-interaction rules (mixed-grain operands)

What is `sessions-set intersect messages-set`? Two defensible policies:

- **P1 — fail closed (recommended for v1).** Set-ops require **identical grain**
  on both operands; a mismatch is a typed `QueryGrainMismatch` error with a hint
  (`wrap the message query in \`| sessions\` to lift it to session grain`). This
  keeps semantics unambiguous and pushes coercion to an explicit stage.
- **P2 — implicit lift (fast-follow).** Provide an explicit `| sessions` lift
  stage (message-set → the set of distinct owning `session_id`s) and a
  `| messages` explode. Then mixed-grain is a user error the tool can *suggest*
  the fix for, not silently coerce.

v1 ships **P1 + the explicit `| sessions` lift**; never silently coerce.

---

## 5. Syntax — trade space and recommendation

Three designs were evaluated (operator opened the door to changing the pipeline
operator itself).

### Design A — set-ops as subquery-taking **pipeline stages** (RECOMMENDED)

```
find auth | intersect (test) | except (draft) | group by model | read
find semantic:"schema migration" | except (week:2026-W01)
@arm_pack | intersect (@arm_raw)
```

- `| intersect (SUBQUERY)`, `| union (SUBQUERY)`, `| except (SUBQUERY)` are
  **binary pipeline stages**; the operand is a full parenthesized subquery
  (may nest set-ops/pipelines) — this **is `fnm.9` generalized**.
- Operates at the current relation's grain; grain flows through the pipe.
- Strict left-to-right, **no new precedence rules**, parens delimit operands.
- **Preserves `|` = pipeline** — non-breaking; zero migration for existing
  queries, docs, tests, muscle memory.
- Parses at the existing pre-Lark split layer (`_split_pipeline_stages`,
  `expression.py:1436`): add `intersect`/`union`/`except` as stage verbs whose
  single argument is re-fed to `_QUERY_PARSER`.

Cost: the two operands read left-heavy (`A | intersect (B)` rather than the
math-like `A ∩ B`). Acceptable; the pipe already conditions users to
left-to-right dataflow.

### Design B — infix keyword **sugar** at top level (fast-follow, optional)

```
auth intersect week:2026-W01
(A intersect B) except C
```

Reads better for the simple 2-operand case. Requires a precedence decision
relative to the pipeline and to `and/or/not`. **Lower it to Design A** (pure
surface sugar); never make it the primitive. Ship only if user demand appears.

### Design C — repurpose operators (`|`=union, `&`=intersect, `-`=except), move pipeline to `then`/`>>` (REJECTED)

```
(auth | test) - draft then group by model then read
```

Most set-algebra-forward and terse, and it matches set/regex convention
(`|`=union). **But it is breaking**: `|` means pipeline everywhere today.
Design A already delivers the *full* algebra without the break, so C's migration
cost (docs, tests, every saved query, operator habit) is unjustified. Revisit
`then`/`>>` only as an independent readability change, decoupled from set-ops.

**Decision: build Design A. Keep `|` = pipeline. Consider B as sugar later.**

---

## 6. Execution model

1. **Parse** — `_split_pipeline_stages` yields stages; a set-op stage carries its
   raw subquery text. The subquery is parsed by the same `_QUERY_PARSER` and
   planned via the normal path (`plan.py` / `plan_execution.py`).
2. **Materialize operands to keyed, ranked sets.** Each operand runs its own
   plan and yields `[(key, rank)]`. Guard budgets per operand (each honors the
   configured row cap); the set-op does **not** multiply the budget.
3. **Combine** on the identity key per §3, applying the ranking policy (§3.1).
4. **Continue the pipeline** — the combined relation feeds downstream stages
   (`group by`, aggregates, `read`, `context-image`) unchanged.
5. **EXPLAIN** shows **two sub-plans joined by a set-op node**, never a cross
   join. Add a `set_op` node to `plan_description.py` and the `explain` output so
   `explain_query_expression` (MCP) and `--explain` surface it.

Performance:

- Session-grain: the combined set is a bounded list of `session_id`s; downstream
  stages already accept an id set (the lineage composition and filter paths use
  `IN (...)`). Intersect/except are hash-set ops in memory over bounded operand
  caps; no SQL cross-join.
- Message-grain: keys are `(session_id, message_id[, variant_index])`; same
  in-memory hash-set combine after each operand materializes. For very large
  operands, materialize the smaller side as the hash and stream the larger.
- Pagination: set-ops are computed **before** the terminal read/limit; the page
  cap applies to the *combined* ranked relation (consistent with how a single
  query paginates). Document that `except`/`intersect` can make the result
  smaller than either operand's cap — that is correct, not truncation.

---

## 7. Cross-surface parity

CLI, MCP, API, and daemon route the identical query string through the one
parser (the `fnm.11` parity matrix must include set-op strings). Additions:

- **Completions** (`fnm.4`): after `|`, offer `intersect (`, `union (`,
  `except (` alongside the existing stage verbs; inside the parens, resume normal
  query completion.
- **Explain**: the `set_op` plan node appears in every surface's explain output.
- **Macros** (`fnm.12`): a macro expands inside an operand exactly like anywhere
  else — `@a | intersect (@b)`.

---

## 8. Relationship to the wider read algebra

- `fnm.9` (pipeline-as-subquery) is the **enabling primitive**; set-op stages are
  its first concrete consumer. Build `fnm.9`'s subquery-execution seam first.
- `4p1` — this makes the read surface a genuine algebra: **Set(Query) ×
  Projection × Render**. Update the `4p1` record to list the binary set-op layer.
- `rlsb` / `4smp` (variants) — `variant_index` in the message identity key keeps
  variant rows distinct under set-ops; a `semantic` operand and a variant-render
  projection compose without special-casing.
- `9l5` (analytics) — cohort set-ops feed `group by` / aggregates directly,
  enabling `(@arm_pack except @arm_raw) | group by model | count` cohort diffs
  with construct-valid measures.

---

## 9. Open decisions (need operator sign-off before build)

1. **Grain-mismatch policy** — confirm P1 (fail-closed + explicit `| sessions`
   lift) for v1. *(recommended)*
2. **`union` ranking** — confirm RRF fusion (rank-honest across lanes) vs a
   simpler "left-first then right-new" concatenation. *(RRF recommended; matches
   hybrid search)*
3. **Ship infix sugar (Design B) in v1, or defer?** *(defer recommended)*
4. **Terminal semantics** — does `except`/`intersect` apply pre- or
   post-aggregate when both appear? (Spec: set-ops are relation-level and always
   compose *before* a terminal aggregate unless parenthesized otherwise.)

---

## 10. Rollout & testing

- Additive; no breaking change (Design A). No schema change (pure query layer).
- Tests: a parametrized matrix over `{union, intersect, except} × {fts, semantic,
  structural, macro} × {session, message grain}` asserting **exact set identity**
  and the ranking policy; a grain-mismatch error test; an EXPLAIN test showing
  two sub-plans; cross-surface parity via the `fnm.11` harness.
- Docs: fold into `docs/search.md` (Retrieval Lanes) + a set-algebra section;
  regenerate the query cookbook (`pj8`).
