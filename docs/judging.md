# Judging

Judgments, comparisons, blind and sampled judging, rankings and agreement, forecasts, and rubrics and passes. Judgments and comparisons are annotations ([annotations.md](annotations.md)), written through the one write path ([annotations.md](annotations.md#the-write-path)) by the `judge` verb; blind reading is the `compare` view ([reading.md](reading.md#compare-and-blind)); the computations behind rankings and agreement are measures ([measures.md](measures.md#rankings-and-agreement)).

Judgment is optional and never a gate: most agent annotations will never be seen by a person, and nothing waits for one.

## Verdicts on annotations

`judge accept|reject|defer [--reason TEXT]` records a `judgment` on the exact revision selected: `polylogue annotation:01JB7Q2M@2 judge accept`. Any author may judge, and the judgment records its true author. The review queue is the shipped saved query `review` (annotations not by the operator that the counted judges have not judged); `polylogue review pick judge accept` triages several.

`judged:` reads the operator's judgments and `judged_by:` defaults to them; `judged_by:agent:<…>` and `judged_by:*` widen it. Judging a revision again revises the judge's earlier judgment of it, so which judgment is current follows the revision links, never the writers' clocks; two current operator judgments of one revision that differ (written apart and merged) are a conflict. Operator-authored annotations need none. No score, check result or count of agent judgments accepts anything; there is no promotion.

**The one-key loop.** In the picker over annotations (`polylogue review pick`), `Ctrl-a`, `Ctrl-x` and `Ctrl-d` record accept, reject and defer on the highlighted row and move to the next; moving past a row writes nothing. Each key is one `judge` request with the same reply and undo line as the typed verb. The key table is [terminal.md](terminal.md#keys).

## The better of several rows

On two or more rows, the verdict is the preferred row and `--on` names the dimension: `polylogue '@1,2' judge @2 --on correctness` records a `comparison` keyed by the dimension (the query condition `on:` selects comparisons by it). Other verdicts on several rows:

| Verdict | Means | Stored `verdict` |
|---|---|---|
| a row (`@2`, a reference, or a blind label `A`) | that row is preferred to each of the others | `prefer:<reference>` |
| an order, `'B>A>C'` | each row is preferred to every row after it | `{ranking: [<reference>, …]}`, best first |
| `tie` | none is preferred | `tie` |
| `incomparable` | the rows cannot be compared on this dimension | `incomparable` |
| `abstain` | the judge declines to choose | `abstain` |
| `insufficient-evidence` | the rows do not show enough to choose | `insufficient_evidence` |

`polylogue '@1,2,3' judge '3>1>2' --on correctness` orders three rows by row number; an order names each row at most once (a repeated row is refused) and leaves the rows it does not name unordered. The last three verdicts add no preference to a ranking but are counted beside it. A comparison stores the real references whatever form the judge typed.

## Blind

References name their provider (`session:codex:…`), so blind judging never shows or takes a reference. The rows are first kept: `polylogue '@1,2' save` prints `result:01JD7Q4M`. Then `polylogue result:01JD7Q4M read --view compare --blind` shows them labelled `A`, `B` (`1…k` beyond two), with references, origin, model, harness, author, cost, tokens, times, durations, working directories and earlier verdicts removed, and tool names shown by family (shell, edit, read); it prints the seed it used and the ready line `polylogue result:01JD7Q4M judge A --on correctness --seed 7`. `judge LABEL --seed N` on a kept result records the preference: the daemon maps the label back from the kept result and the seed, with no server-side session. The comparison stores the real references, the seed, the label mapping and the list of hidden fields. `--blind` on any selection other than a kept result is refused with the `save` line ([reading.md](reading.md#compare-and-blind)). The text can still name a model; the record says what was hidden, not that identity was unknowable. Every verdict above works blind, with labels in place of rows: `polylogue result:01JD7Q4M judge 'B>A' --on correctness --seed 7` maps each label back the same way, and `polylogue result:01JD7Q4M judge tie --on correctness --seed 7` records a tie.

## Sampled judging

A reproducible sample is a transform ([transforms.md](transforms.md#transforms)): `| sample N seed S` draws N rows with seed S, and the same seed over the same rows draws the same sample. The sample is kept first, so what is judged stays fixed while the archive grows:

```console
$ polylogue 'sessions where repo:sinex | sample 20 seed 7' save
kept    result:01JD8A2K (20 rows, 16 Nov 15:10)
$ polylogue result:01JD8A2K read --view compare --blind
$ polylogue result:01JD8A2K judge 'C>A>B' --on correctness --seed 3
```

The kept result records the query with its seed, so the sample can be drawn again and checked. A kept sample is also the population of a pass ([Rubrics and passes](#rubrics-and-passes)).

## Rankings

Many comparisons on one dimension give an order: `polylogue 'annotations where kind:comparison on:correctness | measure ranking'`. The reply lists each compared row with its win rate and its Bradley-Terry strength, each rank with the number of comparisons behind it, and the pairs whose order the comparisons do not settle as ready `judge` lines, fewest comparisons first, because choosing between two things is easier than scoring one. The selection decides whose comparisons count (an `author:` condition narrows it). How the figures are computed is [measures.md](measures.md#rankings-and-agreement).

## Agreement

How far a pass agrees with a reference judge: `polylogue 'annotations where pass:01JD9B3C | measure agreement'`. The reference is the operator's judgments unless another judge is named. The reply gives the agreement with its denominator and the counts of excluded, incompatible and conflicting items, so a high figure over few comparable items reads as what it is ([measures.md](measures.md#rankings-and-agreement)). Labelled samples stay private annotations in the archive; only the figures leave it.

## Forecasts

A `forecast` names an event, a probability, a horizon and what resolves it; the resolution is an annotation that `replies_to` it with the outcome. Calibration uses only forecasts written before their event resolved (by the resolving record's time) and is reported with its count.

## Rubrics and passes

Systematic agent annotation is a pass of annotation lines governed by a rubric, recording what was covered and where the annotator abstained:

1. A `rubric` declares instructions, a unit query, and fields with types, enums or ranges, plus whether abstaining, a confidence and a rationale are allowed.
2. A pass first keeps its population as `result:<id>` (`save` with no name), then imports one `pass` line, the `origin` of everything that follows, and one `rubric_value` per row and field, each `derived_from` the rubric's exact revision.
3. A value the rubric does not allow is refused and returned, not stored. An abstention is a `rubric_value` keyed `abstain` with its reason.
4. **Pass coverage** is the kept rows minus the rows with a value or abstention, computed at read.
5. Withdrawing a pass's whole set is one withdrawal.
6. A marker cannot open a pass; `::rubric_value` names a rubric revision and a pass the same author opened.
7. **Shipped rubrics:** Python Polylogue's six annotation schemas (`delegation.discourse`, `seed.activity`, `seed.goal-event`, `seed.outcome-evidence`, `seed.knowledge-artifact`, `seed.reusability`) ship as rubric data, resolved by name like shipped saved queries; a rubric of one's own with the same name takes precedence.
