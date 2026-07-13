# Analysis rigor for agent claims

Status: ADOPTED 2026-07-13 (operator). Program tracker: polylogue-rxdo.9;
mechanisms materialized as polylogue-rxdo.9.1 – rxdo.9.10.

Operator prompt: "we want to design some more useful abstractions/mechanisms
here. maybe stats adjacent... experimental methods, blinding and such."
This proposal thinks it through against the rxdo object graph
(query:<hash> → query_run → result_set → finding → judgment) currently being
built by the rxdo-substrate/lifecycle/language lanes. Bead: polylogue-rxdo.9.

## 1. The frame: this archive is a population, not a sample

Almost every claim Polylogue makes is an exact count over ALL the data
("4,544,958 FTS rows", "7,383 subagent sessions", "X% of completion claims
lack evidence"). Classical inference — p-values, significance stars — is the
wrong toolkit for exact population counts; importing it would be stats
theater. The failure modes we have actually observed are different, and every
one of them is a *validity* failure, not sampling error:

- **Definition errors**: the 7.69× Codex cost inflation (input tokens
  included cached), the "376.6B tokens" figure (stale rows). The number was
  exactly computed and wrong.
- **Denominator dishonesty**: "user words" counted over protocol rows;
  role=user vs material_origin=human_authored.
- **Fabrication under pattern pressure**: the recovery digest inventing
  "PR #123 merged" from prose regex.
- **Multiple looks**: standing re-evaluation (rxdo.5) will re-test hundreds
  of watched queries per convergence — an alert threshold applied repeatedly
  IS a false-discovery machine even with zero sampling error.
- **Circularity**: agent findings citing agent summaries citing the same
  agent's claims (the evidence-laundering guard in rxdo.4 gestures at this).
- **Temporal confounds**: comparing epochs where the parser/schema changed
  mid-window (v32→v35 content drift is a live example).

So the design principle: **rigor mechanisms are properties of the provenance
graph, not a statistics module.** Each mechanism below is a field, an object
kind, or a planner rule — never a separate analytics engine.

## 2. Mechanisms, ranked by leverage/cost

### A. Metric definitions as content-addressed objects — metric:<hash>  [NEW, highest leverage]
The cost-model incidents were definition bugs. Today "cost_usd" or
"authored user words" is whatever the code path computed that week. Proposal:
a metric definition (unit source, filters, material_origin mask, aggregation,
exclusions) canonicalizes and hashes exactly like query:<hash> (same
machinery the substrate lane is building — near-zero marginal cost). Findings
carry metric_ref; two findings comparing "cost" with different metric hashes
are VISIBLY incomparable, and definition drift becomes a diffable event
rather than a silent re-interpretation. This retroactively explains both cost
fiascos and prevents the third one.

### B. Ratios as derived objects: numerator_ref + denominator_ref
No bare percentages anywhere in finding.v1. A proportion is a derived claim
citing two result-set refs (or one result-set + one cohort). The renderer can
then always answer "% of WHAT" — and the fnm.1 aggregate machinery already
computes explicit denominators/n/unknown-buckets, so this is mostly a schema
convention, not new computation.

### C. Pre-registration with provable ordering
rxdo.4's `expected` field is the hook. Semantics to pin: a REGISTERED
analysis is a finding-candidate whose expected is set while statistic is
null, citing a query hash; running it later fills actuals. Ordering is
provable from the graph (registration row timestamp < query_run timestamp,
and the run's archive_epoch > registration epoch = the claim was tested on
data that arrived after the hypothesis). A claim renders as "confirmed
(pre-registered)" ONLY under that ordering; everything else renders as
"exploratory". Agents don't get moral credit — they get a provable badge or
they don't. Prior art: e5b5 (pre-registered micro-evals).

### D. Holdout cohorts
A persistence class `holdout` on cohorts/result_sets: sessions excluded from
exploratory querying by default. The planner (rxdo.6's RefOperand layer is
exactly the right place) warns-or-fails when an exploratory query references
a holdout cohort; confirmation runs must declare themselves and get marked
in query_runs. Gives before/after demo claims an honest "and it held on data
the analysis never touched" leg.

### E. Standing-query alert discipline (the multiple-looks guard)
rxdo.5 without this becomes a notification firehose with guaranteed false
discoveries. Cheap tier (ship with rxdo.5): per-watch cooldown + minimum
magnitude + the already-designed baseline-then-notify. Proper tier (phase 2):
expected-drift bands learned from a baseline window of epochs; alert only
outside band; a global alerts-per-day budget spent largest-standardized-
deviation-first. No FDR mathematics needed to start — a budget + ordering
gets 90% of the value.

### F. Blinded judgment
When the operator judges finding candidates (p5g interactive judge, MCP
judgment tools), mask provenance — detector/model/agent identity, actor
names — until the verdict is recorded; reveal after. It's a view-profile on
the judge surface, ~a day of work, and it matters precisely when judging
competing agents' findings (Sol-vs-Terra quality arguments, self-reported
lane results). The lifecycle lane's value schema keeps provenance in
separate fields, so masking is a renderer concern — no schema cost.

### G. Negative controls, paired
For a headline claim, the natural falsifier is the same query on a shifted
time window / permuted grouping / unrelated cohort. Mechanism: a finding MAY
carry control_refs (query+result refs); the report renderer shows
claim-vs-control side by side; a registered claim with a passing control
outranks one without. Start as convention + renderer support; automate
control-suggestion later (the query AST makes "same query, shifted window"
mechanically derivable).

### H. Uncertainty only where it is real
exactness (exact/capped/sampled/estimate) already propagates in rxdo.3's
design. Rule: exact counts get NO inferential dressing; sampled/estimated
results MAY carry a bootstrap CI over result-set members (assumption-light,
easy). Explicit anti-goal: p-values on population counts.

### I. Evidence ancestry: circularity + freshness
Extends rxdo.4's laundering guard and x35k's freshness markers: the report
renderer walks refs and flags (a) circular ancestry (claim ultimately cites
its own detector's output), (b) epoch skew (cited result-sets computed
against materially different archive epochs), (c) expired ops-tier refs that
were never promoted. All read-side; no write-path cost.

### J. Session-slice experiments (A/B as cohort pairs)
Two cohorts + one registered metric_ref + a pre-registered comparison = an
experiment object (thin composition over C+A+B). This is the honest version
of "which prompt/model works better" questions over the archive, and the
bridge to the e5b5 eval harness. Phase 3; needs nothing new from the
current lanes beyond what A–C establish.

## 3. What we deliberately do NOT build
- No general statistics library, no scipy dependency for its own sake.
- No p-values/significance on exact population counts (anti-theater rule).
- No auto-injected findings (already the recursive-safety spine).
- No dashboards-first: every mechanism must change what a *claim* looks like
  or when an *alert* fires, or it doesn't get built.

## 4. Sequencing against the live lanes
- **Already compatible, no mid-flight scope change needed**: finding.v1 keeps
  expected extensible (lifecycle prompt requires it); persistence classes
  and value_json are TEXT/additive, so `holdout` and `metric_ref`/
  `numerator_ref`/`control_refs` add cleanly later.
- **Phase 2 (implement rxdo.9 after the three lanes merge)**: metric:<hash>
  (A) riding the substrate canonicalizer; ratio convention (B); registered
  badge logic (C); cheap-tier alert discipline (E) if rxdo.5 didn't ship it;
  holdout class + planner rule (D); blinded judge view (F).
- **Phase 3**: controls automation (G), bootstrap CIs (H), ancestry walker
  (I), experiments (J).

## 5. The demo payoff (why this beats generic "stats")
The show-someone moment: click any headline claim and see — the metric
definition hash, the registration timestamp preceding the run, the evidence
refs, a re-run button, and the holdout confirmation. Agent claims that carry
their own falsification handles. Nobody else's agent-memory product can do
that, and it is exactly the credibility demos need.
