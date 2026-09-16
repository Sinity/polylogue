"""Analysis-rigor measurement primitives (polylogue-rxdo.9 program).

Content-addressed, storage-agnostic building blocks for the "measurement
substrate" half of the rigor program (docs/design/analysis-rigor.md):

- :mod:`canon` -- shared content-address canonicalizer (`<kind>:<hash>`).
- :mod:`metric` -- ``metric:<hash>`` canonical metric definitions (rxdo.9.1).
- :mod:`registry` -- measure specs and composition.
- :mod:`registered_metrics` / :mod:`registered_measures` -- the process-wide
  default registries, reachable through the MCP ``get`` tool.
- :mod:`outcome_coverage` -- coverage-gated tool-outcome aggregates; a bare
  success-rate scalar is refused below a declared coverage floor and the
  aggregate reports ``degraded`` with the gap named (polylogue-cuxz.4 AC4).
  This is the one route a tool-outcome rate may be produced through, and
  ``tests/unit/architecture/test_tool_outcome_rate_ratchet.py`` makes that
  floor binding across every read path.
- :mod:`public_claims` -- claim/finding provenance.

Only mechanisms a declared route reaches live here. The pre-registration
ordering proof (rxdo.9.3), derived ratio metrics (rxdo.9.2), exactness-gated
interval rendering (rxdo.9.8), the standing-query alert budget (rxdo.9.5) and
the evidence-graph ancestry walker (rxdo.9.9) were specified in
docs/design/analysis-rigor.md and written here, but no operation, CLI verb or
MCP tool ever called them; they were deleted rather than kept as unreachable
code. Re-introduce each one together with its route.
"""

from __future__ import annotations

from polylogue.analysis.measurement.outcome_coverage import (
    COVERAGE_BELOW_FLOOR,
    TOOL_OUTCOME_COVERAGE_FLOOR,
    ToolOutcomeAggregate,
    build_tool_outcome_aggregate,
)
from polylogue.analysis.measurement.registry import (
    MeasurePlan,
    MeasureRegistry,
    MeasureResult,
    MeasureSpec,
    MeasureValidityError,
    compose_measure,
)

__all__ = [
    "COVERAGE_BELOW_FLOOR",
    "TOOL_OUTCOME_COVERAGE_FLOOR",
    "MeasurePlan",
    "MeasureRegistry",
    "MeasureResult",
    "MeasureSpec",
    "MeasureValidityError",
    "ToolOutcomeAggregate",
    "build_tool_outcome_aggregate",
    "compose_measure",
]
