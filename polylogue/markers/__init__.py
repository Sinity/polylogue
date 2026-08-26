"""Provider-neutral, agent-declared structure markers.

Markers are advisory authoring syntax.  This package extracts evidence and
lowers it through existing typed owners; it does not define a marker table or
an enforcement lifecycle.
"""

from polylogue.markers.lowering import candidates_for_block, lower_markers
from polylogue.markers.models import MarkerCandidate, MarkerKindSpec, MarkerMatch, MarkerProvenance
from polylogue.markers.parser import MarkerStreamParser, parse_markers
from polylogue.markers.registry import MARKER_REGISTRY, MarkerRegistry, marker_registry


def scan_block(message_id: str, block_id: str, text: str) -> tuple[MarkerCandidate, ...]:
    """Production marker scan entrypoint for derived block enrichment."""
    return candidates_for_block(message_id, block_id, text)


__all__ = [
    "MARKER_REGISTRY",
    "MarkerCandidate",
    "MarkerKindSpec",
    "MarkerMatch",
    "MarkerProvenance",
    "MarkerRegistry",
    "MarkerStreamParser",
    "candidates_for_block",
    "lower_markers",
    "marker_registry",
    "parse_markers",
    "scan_block",
]
