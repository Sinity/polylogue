# Finding: Corpus Reliability and Closure Reliability

<!-- public-claim:finding.corpus-reliability-rates -->

This is a dated evidence record from the 2026-08-24 backlog review. Future rate estimates and backlog-reduction proposals must reference this record and preserve its distinction between assertion-level staleness and record-level invalidity.

## Measured rates

The review found that about 46% of individual checkable assertions in open records were stale. Fifteen of the sixteen stale assertions examined overstated work that had already been fixed. A stale assertion can still belong to a record describing real remaining work, so this is an assertion-level rate.

At the record level, 27 of 197 records verified in depth were fully void, or about 14%. Roughly 95 records were verified live and remained untouched. The two rates describe different units and are therefore not contradictory: stale citations usually require correction, while only a minority of records require closure or removal.

The practical estimate is that citation cleanup removes about one record in seven, once. A materially smaller backlog requires completing or consciously declining real work, rather than relying on citation cleanup.

## Closure reliability

Closures with a written reason were sampled 40 times and produced effectively zero false closures in that sample. Closures without a reason were sampled 10 times, with six false closures, a 60% false-closure rate. The evidence supports requiring a written closure reason and treating a bare closure as unreliable until independently verified.

These figures are sampled observations, not universal prevalence estimates. They describe the reviewed backlog and method as of 2026-08-24. They must not be presented as a current rate without repeating the measurement against a defined population and recording the frame, sample, date, and classification method.

## Staleness modes

Full-record verification found three distinct ways a record can become stale:

- The work landed and the commit cited the record, while an unlinked stale record remained open.
- Code was renamed or moved, so the record's original symbols no longer located the shipped work.
- A sibling record completed the work without linking the records, leaving duplicate open descriptions.

The review also captured a record asserting a defect on three symbols that did not exist at the current head. It was created from an earlier narrative without re-verification and closed after the claim was checked. This is a concrete example of why inherited claims require fresh measurement.

## Reuse contract

Any future rate estimate or backlog-reduction proposal that uses this finding must:

1. State whether its numerator counts assertions or whole records.
2. Define the verified population, sample frame, date, and classification method.
3. Keep stale details separate from fully void records.
4. Report closure-reason presence as a separate dimension from closure status.
5. Recheck referenced symbols, commits, renamed paths, and sibling records at the current head.

The canonical source is bead `polylogue-nr7xw`, whose acceptance criterion requires this record to be referenced by future estimates and reduction proposals. This page preserves the measured distinction and prevents the figures from being re-derived from chat or stale task prose.
