# Judging

Judgments, comparisons, blind judging, forecasts, and rubrics and passes. Judgments and comparisons are annotations ([annotations.md](annotations.md)), written through the one write path ([annotations.md](annotations.md#the-write-path)) by the `judge` verb; blind reading is the `compare` view ([reading.md](reading.md#compare-and-blind)).

Judgment is optional and never a gate: most agent annotations will never be seen by a person, and nothing waits for one.

## Verdicts on annotations

`judge accept|reject|defer [--reason TEXT]` records a `judgment` on the exact revision selected: `polylogue annotation:01JB7Q2M@2 judge accept`. Any author may judge, and the judgment records its true author. The review queue is the shipped saved query `review` (annotations not by the operator that the counted judges have not judged); `polylogue review pick judge accept` triages several.

`judged:` and `judged_by:` default to the operator's judgments; `judged_by:agent:<…>` and `judged_by:*` widen it. Judging a revision again revises the judge's earlier judgment of it, so which judgment is current follows the revision links, never the writers' clocks; two current operator judgments of one revision that differ (written apart and merged) are a conflict. Operator-authored annotations need none. No score, check result or count of agent judgments accepts anything; there is no promotion.

## The better of several rows

On two or more rows, the verdict is the preferred row and `--on` names the dimension: `polylogue '@1,2' judge @2 --on correctness` records a `comparison` keyed by the dimension, its verdict `prefer:<reference>`.

## Blind

References name their provider (`session:codex:…`), so blind judging never shows or takes a reference. The rows are first kept: `polylogue '@1,2' save` prints `result:01JD7Q4M`. Then `polylogue result:01JD7Q4M read --view compare --blind` shows them labelled `A`, `B` (`1…k` beyond two), with references, origin, model, harness, author, cost, tokens, times, durations, working directories and earlier verdicts removed, and tool names shown by family (shell, edit, read); it prints the seed it used and the ready line `polylogue result:01JD7Q4M judge A --on correctness --seed 7`. `judge LABEL --seed N` on a kept result records the preference: the daemon maps the label back from the kept result and the seed, with no server-side session. The comparison stores the real references, the seed, the label mapping and the list of hidden fields. `--blind` on any selection other than a kept result is refused with the `save` line ([reading.md](reading.md#compare-and-blind)). The text can still name a model; the record says what was hidden, not that identity was unknowable.

## Forecasts

A `forecast` names an event, a probability, a horizon and what resolves it; the resolution is an annotation that `replies_to` it with the outcome. Calibration uses only forecasts written before their event resolved (by `commit_seq` on one archive, otherwise by the resolving record's time) and is reported with its count.

## Rubrics and passes

Systematic agent annotation is a pass of annotation lines governed by a rubric, recording what was covered and where the annotator abstained:

1. A `rubric` declares instructions, a unit query, and fields with types, enums or ranges, plus whether abstaining, a confidence and a rationale are allowed.
2. A pass first keeps its population as `result:<id>` (`save` with no name), then imports one `pass` line, the `origin` of everything that follows, and one `rubric_value` per row and field, each `derived_from` the rubric's exact revision.
3. A value the rubric does not allow is refused and returned, not stored. An abstention is a `rubric_value` keyed `abstain` with its reason.
4. **Coverage** is the kept rows minus the rows with a value or abstention, computed at read.
5. Withdrawing a pass's whole set is one withdrawal.
6. A marker cannot open a pass; `::rubric_value` names a rubric revision and a pass the same author opened.
7. **Shipped rubrics:** Python Polylogue's six annotation schemas (`delegation.discourse`, `seed.activity`, `seed.goal-event`, `seed.outcome-evidence`, `seed.knowledge-artifact`, `seed.reusability`) ship as rubric data, resolved by name like shipped saved queries; a rubric of one's own with the same name takes precedence.
