"""Analysis-rigor measurement primitives (polylogue-rxdo.9 program).

Content-addressed, storage-agnostic building blocks for the "measurement
substrate" half of the rigor program (docs/design/analysis-rigor.md):

- :mod:`canon` -- shared content-address canonicalizer (`<kind>:<hash>`).
- :mod:`metric` -- ``metric:<hash>`` canonical metric definitions (rxdo.9.1).
- :mod:`ratio` -- ratios as derived ``metric:<hash>`` objects over a
  numerator/denominator component-ref pair (rxdo.9.2).
- :mod:`registration` -- pre-registration ordering proof (rxdo.9.3).
- :mod:`uncertainty` -- exactness-gated interval rendering; bootstrap CIs
  apply only to sampled/estimated results, never to exact enumeration
  (rxdo.9.8).
- :mod:`alert_budget` -- standing-query alert cooldown/magnitude-floor/
  budget policy, the multiple-looks guard for rxdo.5 (rxdo.9.5).
- :mod:`evidence_ancestry` -- read-side evidence-graph ancestry walker:
  circularity, epoch skew, definition incompatibility, and expired/stale/
  missing/ambiguous/quarantined/private ref detection (rxdo.9.9).

These modules deliberately do not each invent their own persistence layer.
Where a real durable substrate already exists (``queries``/``result_sets``
in :mod:`polylogue.storage.sqlite.query_objects`, rxdo.2), later mechanisms
build directly on it (see holdout policy in that module, rxdo.9.4). Where no
consuming storage exists yet (finding.v1, ``ExperimentDefinition``/``stc``,
the statistics registry in 9l5.7), the mechanism here is the pure,
storage-agnostic identity/ordering/decision function a future storage lane
wires up -- each such module says so explicitly in its docstring.
"""

from __future__ import annotations
