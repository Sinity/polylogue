"""Comparative-judgment mechanisms (rxdo.9.6/.7/.10-.16, Part I/II of docs/design/analysis-rigor.md).

Absolute scoring is unreliable for humans AND agents; the reliable
elicitation primitive is comparison. This package implements the
psychometric-consensus (Bradley-Terry/Thurstone/Plackett-Luce) primitives the
archive needs to answer quality questions the exact-count mechanisms in
:mod:`polylogue.insights.rigor` cannot:

- :mod:`polylogue.insights.judgment.types` -- shared ``ComparativeJudgment``
  shape (mechanism K, rxdo.9.11).
- :mod:`polylogue.insights.judgment.comparative` -- build/serialize
  comparative judgments for assertion-row storage.
- :mod:`polylogue.insights.judgment.blinding` -- provenance masking until
  verdict (mechanism F, rxdo.9.6).
- :mod:`polylogue.insights.judgment.calibration` -- judges as actors, measured
  agreement with gold (mechanism L, rxdo.9.12).
- :mod:`polylogue.insights.judgment.rankers` -- ``ranker:<hash>`` aggregation
  models over judgment sets (mechanism M, rxdo.9.13).
- :mod:`polylogue.insights.judgment.controls` -- paired negative controls on
  findings (mechanism G, rxdo.9.7).
- :mod:`polylogue.insights.judgment.elicitation` -- active elicitation
  sessions, the resorter loop (mechanism N, rxdo.9.14).
- :mod:`polylogue.insights.judgment.cascades` -- agent-screen-to-operator-gold
  routing (mechanism O, rxdo.9.15).
- :mod:`polylogue.insights.judgment.experiments` -- experiment analysis
  projection over stc definitions (mechanism J, rxdo.9.10).

No new lifecycle store: comparative judgments are stored as
``AssertionKind.COMPARATIVE_JUDGMENT`` rows through the existing assertion
substrate (``polylogue.storage.sqlite.archive_tiers.user_write``), and every
other object in this package (calibration reports, ranker fits, cascade
decisions) is a derived, re-runnable computation over that substrate -- never
a second store to keep in sync.
"""

from __future__ import annotations
