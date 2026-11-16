# Transforms and related rows

What a query can say beyond conditions on one unit: conditions over related rows, transforms after `|`, attached rows and combining queries. The query language itself is the shared one; Polylogue's units, fields and defaults are [query.md](query.md).

## Conditions over related rows

- `exists child(C)`: one child satisfies all of C; three-valued. `exists child()` is true when any related child exists, and false when there is none.
- `count(UNIT [where C]) OP N`: a count is always known, so it and its `NOT` partition the rows.
- `FIELD in (QUERY | group by F)`: the value is one of the other query's grouped values of the same type.
- `seq(A -> [next] B -> [within:5m] C)`: tool calls in session order. Matches are the earliest non-overlapping; a step's own conditions narrow before matches are chosen; `within` needs both times, the later strictly after; `| matches` returns each match as rows. `seq all(…)` (every, overlapping match) and `gap_steps`/`gap_time` on match rows are part of the language.

## Transforms

A transform is a step after `|` inside a query. Transforms apply in the order written; `help transforms` lists them:

- `sort by F [asc|desc], …`: unknowns last in either direction; row identity is the final tie-break. `sort by random[(seed)]` carries its seed in the cursor.
- `group by K1, K2 | count` or `| agg …`: keys are fields and `day()`, `week()`, `month()`. Group keys and `min` or `max` of a reference retain that field's reference metadata. The group or summary has its own row identity and no entity target. Computed aggregate values carry no entity reference.
- `agg count(), sum(E), avg(E), min(E), max(E), pNN(E), count_distinct(E), histogram(E), corr(E, F), covered(start, end)`: expressions are evaluated at the input row's grain; unknown and partial values are reported as `unknown_F` and `partial_F`; a `sum` over no rows is 0, and a `sum` whose members are all unknown stays unknown; percentiles are nearest-rank. `covered` measures the union of intervals, so overlap is counted once. `histogram` returns ordered JSON bins with `value` and `count`; `corr` is Pearson correlation over complete known pairs and stays unknown for fewer than two pairs or constant input. A numeric total or average is unknown only when it has no known inputs; a nonfinite total over known inputs fails rather than returning unknown. Money is integer nano-USD through every condition, expression and aggregate: an amount scaled by a fraction and an average amount round once to the nano-USD, and a total outside the 64-bit range fails. Replies carry an amount as its exact decimal dollars (`0.000123456`), and a kept result reads them back exactly. Default and explicit field selections carry the same kinds, including turn cost, latency and booleans. Arithmetic preserves dimensions; division by zero and operations on unknown inputs stay unknown.
- `text(E)` treats a string value as plain text and clears entity-reference metadata. It propagates unknown values and refuses numbers, times, booleans, lists and objects; a JSON string is decoded as a string. Natural `FIELD:value` matching follows the bound field kind after field selection: plain text uses Unicode lowercase substring matching with accents preserved, references and exact fields use equality, and paths respect whole-component boundaries. Explicit `=` remains equality.
- `| where C` straight after the source narrows its selection as one conjunction with it: `A | where B` selects, ranks and marks snippets as `A B` does, and never drops a default the source bound. It stays a separate step where the conjunction would mean something else: beside an order condition or `latest`, or with a later `| matches`. `| where C` after grouping filters groups. Field selection, filtering, sorting, row selection, grouping and aggregation execute at their written relation boundaries. An output alias may use any accepted name, including `_pk`; internal row keys and ordering slots occupy a separate namespace.
- `| group by K | first N` returns the first N input rows of each stored group, in the input order. It and `| group by K | agg …` read the grouping's input rows once. A key join returns every matching pair, with the right-hand columns under `match.`. An order condition returns an outer row once and exposes its nearest satisfying matching row under `match.`; a later order condition replaces that namespace.
- `limit N`, `offset N` are selections and change the question; there is no maximum. They are not the page size (`--page-size`, [Paging](query.md#paging)).
- `distinct` removes repeated row identities. A field selection changes visible fields while preserving row identity, so equal selected values from different rows remain separate.
- `matches` carries the entering row's visible columns under `parent.` with their types and reference metadata; projected aliases retain their names. Its matched child fields are under `step.`. Parent rows stay in their entering order, followed by each parent's match and step order.
- `| UNIT [where C]` returns rows of another unit within the matched sessions; `fields a,b` projects columns (`--fields` on the command line).
- `| measure WHAT` hands the selection to a measure ([measures.md](measures.md)); it is the last transform.
- Further transforms are part of the language and are refused `not_built` until built: derived keys (`program(command)`, `fingerprint(text)`), `| series`, `| extract number before "MB/s" as rate`, `| cluster by similar [k:20]`, and `let` names within one query.

## Attached rows

`QUERY with messages(id, role, text)[role:assistant, last:2], tool_calls(…)` returns each parent row with one list per child spec. Inside the brackets, conditions filter first, then `first:` and `last:` take that many per parent in the child's order. An empty field list, `with messages()`, attaches the child unit's own columns. A page's attached rows are read in one statement per child spec over that page's parents. Attaching never changes which parents are returned. There is no page-wide ceiling, and attached text is never cut.

## Combining queries

- `Q | union (R)`, `| intersect (R)` and `| except (R)` combine two queries of the same unit; `| fuse (R)` adds R's order as one more ranking without adding rows.
- A chain containing `| fuse` or a ranked operand is one reciprocal-rank fusion over all original operands ([search.md](search.md#fusion)); an unranked operand contributes its captured order. A fuse operand's ranks are taken over its full input before membership restriction. Values come from the first membership operand that supplies a row; compatible stored representations are converted to the output column's units. A member without an optional matching row has unknown matching row values and coordinates. `found_in` names the membership lists that ranked it. Unranked unions remove duplicate identities and order by identity. `intersect` and `except` follow the left order.
- Grains must match; a mismatch is `invalid_query` suggesting the explicit lift (`| sessions`). Set operands with selected fields require compatible field declarations, including reference targets; a reference and `text(reference)` need the same explicit field selection on both sides. Combining applies before a terminal aggregate.
