Title: "[testdiet 15] Equivalent-instant temporal behavior"

Job ID: `testdiet-15`
Result ZIP: `testdiet-15-temporal-equivalence-r01.zip`

## Mission

Implement temporal survivor laws proving that equivalent instants and declared
provider timestamps produce consistent Polylogue behavior across parsing,
filtering/query ranges, ordering, freshness, durations, receipts, and public
rendering. Use existing frozen-clock and temporal strategy infrastructure.

Vary timezone offsets, DST folds/gaps, leap-day/month/year boundaries,
subsecond precision, naive/aware provider inputs, absent timestamps, equal
instants with different representations, and start/end inclusivity. Preserve
the semantic distinction between provider-reported durations, observed wall
time, and inferred message gaps; never relabel one as model compute time.

The oracle should normalize through an independent standard instant model and
explicit interval rules, not production helpers under test. Name a timezone
strip, local-time comparison, inclusive-boundary, precision truncation, or
duration-unit mutation. Keep exact unique provider wire witnesses and propose
only dominated repetitive boundary examples.
