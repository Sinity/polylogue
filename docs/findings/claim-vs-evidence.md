# Field Finding: What Happened After a Structured Tool Failure?

## Claim

<!-- public-claim:finding.silent-proceed-lower-bound -->
The historical packet generated on 2026-07-04 reported that, in one bounded private-archive sample, at least 24.1% of sampled structured failures were followed by an assistant turn that proceeded without an acknowledgment marker. Most sampled cases remained ambiguous.

The generated [findings-page public-claims view](../generated/public-claims/findings-page.md) is the authority for whether this historical number is currently supported, stale, private-held, or unresolved.

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

The calibration is small. The method therefore keeps 3,375 cases ambiguous instead of forcing them into acknowledged or silent classes.

## Interpretation

The finding establishes that Polylogue can ask and operationalize a question ordinary transcript search does not naturally answer: after a structurally recorded failure, did the subsequent assistant behavior visibly acknowledge it?

It does not establish why the assistant proceeded, whether the outcome was eventually repaired, whether silence was harmful in every case, or how frequently the behavior occurs outside the sampled archive.

## Reproduce the method without private data

```bash
export POLYLOGUE_ARCHIVE_ROOT=/tmp/polylogue-claim-vs-evidence-demo
polylogue demo seed --root "$POLYLOGUE_ARCHIVE_ROOT" --force --with-overlays --format json
polylogue demo verify --root "$POLYLOGUE_ARCHIVE_ROOT" --require-overlays --format json

devtools workspace claim-vs-evidence \
  --archive-root "$POLYLOGUE_ARCHIVE_ROOT" \
  --limit 5000 \
  --out-dir /tmp/polylogue-claim-vs-evidence-repro \
  --json
```

The deterministic corpus reproduces the method and controls. It does not reproduce the private-archive prevalence result.

## Regenerate the private packet locally

Operators with the relevant archive can run:

```bash
devtools workspace claim-vs-evidence \
  --limit 5000 \
  --out-dir .agent/demos/claim-vs-evidence \
  --json

devtools workspace demo-shelf
```

## Evidence and caveats

See:

- [Proof Artifacts](../proof-artifacts.md);
- `devtools/claim_vs_evidence.py`;
- `tests/unit/devtools/test_claim_vs_evidence.py`;
- the local `.agent/demos/claim-vs-evidence/` packet when generated.

Publication requires the packet’s archive cursor, measure version, commit SHA, sample-frame predicate, and run date. If any is missing or stale, the finding page should refuse regeneration rather than silently retain an old number.
