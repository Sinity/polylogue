# Field Finding: What Happened After a Structured Tool Failure?

## Claim

<!-- public-claim:finding.silent-proceed-lower-bound -->
The historical packet generated on 2026-07-04 reported that, in one bounded private-archive sample, at least 24.1% of sampled structured failures were followed by an assistant turn that proceeded without an acknowledgment marker. Most sampled cases remained ambiguous. It is a historical observation, not a current headline.

## Current construct-validity verdict (2026-07-18)

The old rate is **not currently publishable**. A fresh full-frame audit found only
20 structured failures (all from one Claude Code session), below the report's
minimum of 30. Fourteen apparent `silent-proceed` rows were internal
`<thinking>…</thinking>` protocol content, so the classifier now treats them as
ambiguous rather than evidence of a visible silent follow-up. The current frame
therefore has zero classified rows and no rate.

**Correction to this note's earlier framing:** the n=20, single-origin frame is
not evidence that the real corpus is small or single-origin. At the time this
audit ran (and still, as of this writing), the live archive's derived index was
in a known, actively-tracked degraded state: `readiness_check` reports
`raw_materialization: poisoned`, with only 4 of 73,295 raw source artifacts
materialized into `index.db` (`join_gap_count: 73291`). The real corpus behind
that gap spans `codex-session` (39,638), `claude-code-session` (21,420),
`chatgpt-export` (7,679), `claude-ai-export` (3,922), `hermes-session` (193),
and five smaller origins. This is bead `polylogue-hjpx` / `polylogue-hjpx.2`
(a P0/P1 raw-authority replay fixed-point program, owned by a separate lane,
explicitly not authorized for live-archive repair yet) — not a Lane A finding
and not something this lane fixes. Until that backlog clears, **any live-archive
number this report emits reflects whatever tiny slice happens to be
materialized at run time, not the archive's real shape.** Treat every
"current bounded result" below as provisional and re-run after
`readiness_check.archive_convergence.materialization_ready` is `true`.

Operationally, `silent-proceed` means only: after a structurally failed tool
result, the next visible assistant message contains no configured explicit
failure-acknowledgement marker. It does **not** mean recovery was wrong: a
wordless retry or a successful corrective action may be appropriate. The two
largest validity threats are hidden/protocol-only content being mistaken for
visible prose, and (independently of the materialization gap above) a
single-origin frame if one recurs after re-materialization. The claim would be
falsified by a representative, sufficiently large calibration frame showing
that visible marker absence does not track human labels for this narrow
observable.

This page deliberately labels the number historical. A newly generated result
is not a current public claim merely because the report command completed.

This is a lower-bound field observation from one archive and one method. It is not a prevalence estimate for all agents, models, users, providers, or tasks.

## Corpus

The tracked packet was generated on 2026-07-04 against archive schema v24.

- structured-failure frame: 42,033;
- bounded origin-stratified sample: 5,000;
- acknowledged on next assistant turn: 420;
- silent proceed on next assistant turn: 1,205;
- ambiguous next assistant turn: 3,375;
- acknowledged within the next three assistant turns: 722.

The next-turn silent lower bound is therefore 1,205 / 5,000 = 24.1%.


## Handler-class split

<!-- public-claim:finding.handler-class-split -->
The packet partitioned the same 5,000 inspected failures using explicit tool-name methodology classes:

| Handler class | Inspected failures | Silent proceed | Ambiguous | Silent lower bound |
| --- | ---: | ---: | ---: | ---: |
| Consequential | 4,175 | 930 | 2,842 | 22.3% |
| Benign recovery | 634 | 172 | 455 | 27.1% |
| Other | 191 | 103 | 78 | 53.9% |

These are method-defined groups, not severity labels. The ambiguous remainder remains visible in every class.

## Per-origin inspection counts

<!-- public-claim:finding.per-origin-inspection-counts -->
The bounded origin-stratified allocation was:

| Origin | Inspected | Requested | Complete origin frame |
| --- | ---: | ---: | ---: |
| `claude-code-session` | 3,752 | 3,752 | 31,555 |
| `codex-session` | 1,241 | 1,241 | 10,429 |
| `claude-ai-export` | 7 | 7 | 49 |

These are inspection/frame counts determined by this archive and allocation rule, not provider prevalence estimates.

## Structural oracle

A failure enters the frame from normalized tool-result evidence:

- provider `is_error = true`; or
- a supported nonzero process exit code.

Assistant prose is not used to decide whether the tool failed. Prose is used only by the acknowledgment marker applied to later assistant turns.

## Marker calibration

The tracked calibration set contains 50 labeled rows. The packet reports:

- precision: 100.0%;
- recall: 84.2%.

The calibration is small. The method therefore keeps 3,375 cases ambiguous instead of forcing them into acknowledged or silent classes. Those 50 historical labels do not overlap the 2026-07-18 frame; they are useful historical calibration evidence, not fresh validation of its current rate.

## First-party evidence boundary

The generating command was read-only by default. With explicit
`--materialize-evidence`, it recorded a content-addressed analysis definition,
result-set membership, evaluation receipt, and finding through the archive's
existing user-tier writers, and emitted a public-claim declaration only when
the run's own minimum-sample and classified-outcome gates passed. That
harness was retired 2026-08 (the closed claim-vs-evidence campaign's
private-report/calibration/publishing tooling — see #3950); the surviving
`PublicClaimProjection` (`polylogue/insights/measurement/public_claims.py`)
still applies publication, privacy, freshness, frame, and evidence-integrity
state independently on ordinary production routes. This page remains a frozen
historical record, not a currently regeneratable claim.

## Interpretation

The finding establishes that Polylogue can ask and operationalize a question ordinary transcript search does not naturally answer: after a structurally recorded failure, did the subsequent assistant behavior visibly acknowledge it?

It does not establish why the assistant proceeded, whether the outcome was eventually repaired, whether silence was harmful in every case, or how frequently the behavior occurs outside the sampled archive.

## Reproducing this finding

The generating harness (formerly the devtools workspace claim-vs-evidence
command, now deleted, and its private-report/calibration/publishing
machinery) was retired 2026-08 once the
closed campaign it served (`polylogue-sru`) had its terminal artifacts. This
page is therefore frozen historical text: the numbers above are not
regeneratable through a current command. The reusable query semantics behind
them remain on ordinary production routes via `PublicClaimProjection`
(`polylogue/insights/measurement/public_claims.py`).

Publication of any future finding of this shape still requires the packet's
archive cursor, measure version, commit SHA, sample-frame predicate, and run
date; a finding page should refuse regeneration rather than silently retain
an old number when any of those is missing or stale.
