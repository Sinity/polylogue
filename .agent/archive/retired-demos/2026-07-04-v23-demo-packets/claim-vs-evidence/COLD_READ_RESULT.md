# Cold-Read Result

Reader: subagent `019f2735-f872-7ab1-a392-c2f61ee08d57`
Input boundary: only `.agent/demos/claim-vs-evidence/`
Outcome: PASS

## Answer Summary

The reader correctly stated that the packet proves Polylogue can anchor a
failure-follow-up report on normalized structured tool-result outcomes
(`is_error=1` or non-zero `exit_code`), classify the immediately following
assistant turn for explicit acknowledgment markers, and publish aggregate rates
without exposing private transcript text.

The reader correctly did not claim that the private aggregate is reproducible
without the private archive, that seeded-demo rates equal private-corpus rates,
that assistant prose is the failure truth source, or that the classifier is an
LLM judgment of intent or utility.

The reader recovered the sample frame: `/home/sinity/.local/share/polylogue`,
index schema v23, entire archive, 41,886 structured failures, 100 unpaired
failures, 5,000 inspected by origin-stratified bounded sampling, with 3,746
Claude Code, 1,247 Codex, and 7 Claude.ai-export rows inspected.

The reader recovered the main rates and calibration: 420 acknowledged, 1,205
silent-proceed, 3,375 ambiguous; 24.1% next-turn silent lower bound; 74.2%
silent among classified; 37.0% next-3 silent lower bound; 50 calibration labels,
100.0% acknowledged-marker precision, and 84.2% recall.

The reader recovered the private-data-free reproduction commands using
`polylogue demo seed`, `polylogue demo verify`, and
`devtools workspace claim-vs-evidence --archive-root`.

## Full Reader Output

~~~text
PASS

Before quoting any rate, the caveats are: this is aggregate evidence from a private live archive, not a public reproducible corpus rate; the frame is bounded by `--limit 5000`, so it is not a whole-archive rate; ambiguous rows remain in the denominator; and classification is marker-based, not an LLM judgment of intent, utility, or whether the assistant "really handled" the failure.

What it proves: Polylogue can anchor a failure-follow-up report on normalized structured tool-result failures, specifically `is_error=1` or non-zero `exit_code`, classify the immediately following assistant turn for explicit acknowledgment markers, publish aggregate rates, and do so without exposing private transcript text.

What it does not prove: it does not prove the private aggregate is reproducible without the private archive, does not prove seeded-demo corpus rates, does not infer truth from assistant prose, and does not claim all failures were reviewed or semantically judged.

Sample frame used: live archive root `/home/sinity/.local/share/polylogue`, index schema v23, entire archive time window, 41,886 total structured failures, 100 unpaired failures, 5,000 inspected failures selected by origin-stratified bounded sampling. Origin split: 3,746 Claude Code, 1,247 Codex, 7 Claude AI export. Next-turn counts were 420 acknowledged, 1,205 silent-proceed, 3,375 ambiguous. Silent lower bound was 24.1%; silent among classified was 74.2%. Next-3 sensitivity found 722 acknowledgments and a 37.0% silent lower bound. Calibration used 50 labeled rows with acknowledged-marker precision 100.0% and recall 84.2%.

Private-data-free reproduction:

```bash
export POLYLOGUE_ARCHIVE_ROOT=/realm/tmp/polylogue-claim-vs-evidence-demo
polylogue demo seed --root "$POLYLOGUE_ARCHIVE_ROOT" --force --with-overlays --format json
polylogue demo verify --root "$POLYLOGUE_ARCHIVE_ROOT" --require-overlays --format json
devtools workspace claim-vs-evidence --archive-root "$POLYLOGUE_ARCHIVE_ROOT" --limit 5000 --out-dir /realm/tmp/polylogue-claim-vs-evidence-repro --json
```

That reproduces the method and artifact shape, not the private archive's rates.
~~~
