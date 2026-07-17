Title: "[analysis 09] Raw-to-query freshness and false-empty audit"

Job ID: `analysis-09`
Result ZIP: `analysis-09-archive-freshness-gap-r01.zip`

Audit the exact source-to-query path for newly browser-captured sessions. The
live incident is concrete: a same-day ChatGPT capture exists in the raw capture
directory while the archive query for recent ChatGPT sessions returns zero.
Trace capture discovery, cursor/acquisition, raw storage, parse/materialize,
index/FTS convergence, and public query diagnostics. Classify the observed
state without mutating the archive.

Adjudicate `polylogue-1xc.13`, `lkrc`, `yla8`, and `z9gh.3`: name the exact
source identity/receipt a model should receive, what an empty result can and
cannot mean, and whether current named-source surfaces actually expose that
truth. Produce an implementation ordering, anti-vacuity fixtures, and any
Bead corrections needed; do not propose archive-wide scans or manual repair as
normal operation.
