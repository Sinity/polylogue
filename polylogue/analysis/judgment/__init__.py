"""Comparative-judgment mechanisms (rxdo.9.6/.7/.10-.16, Part I/II of docs/design/analysis-rigor.md).

Absolute scoring is unreliable for humans AND agents; the reliable
elicitation primitive is comparison. This package implements the
psychometric-consensus (Bradley-Terry/Thurstone/Plackett-Luce) primitives the
archive needs to answer quality questions the exact-count mechanisms in
:mod:`polylogue.analysis.rigor` cannot:

- :mod:`polylogue.analysis.judgment.types` -- shared ``ComparativeJudgment``
  shape (mechanism K, rxdo.9.11).
- :mod:`polylogue.analysis.judgment.comparative` -- build/serialize
  comparative judgments for assertion-row storage.
- :mod:`polylogue.analysis.judgment.blinding` -- provenance masking until
  verdict (mechanism F, rxdo.9.6).
- :mod:`polylogue.analysis.judgment.calibration` -- judges as actors, measured
  agreement with gold (mechanism L, rxdo.9.12).
- :mod:`polylogue.analysis.judgment.controls` -- paired negative controls on
  findings (mechanism G, rxdo.9.7).

Only the mechanisms a declared route reaches live here. Aggregation models
(M), elicitation sessions (N), judge cascades (O) and experiment-analysis
projections (J) were specified in docs/design/analysis-rigor.md but never
wired to an operation, CLI verb or MCP tool, and were deleted rather than
kept as unreachable code; re-introduce each one with its route.

No new lifecycle store: comparative judgments are stored as
``AssertionKind.COMPARATIVE_JUDGMENT`` rows through the existing assertion
substrate (``polylogue.storage.sqlite.archive_tiers.user_write``), and
calibration reports are a derived, re-runnable computation over that
substrate -- never a second store to keep in sync.
"""

from __future__ import annotations
