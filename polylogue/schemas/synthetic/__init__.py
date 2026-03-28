"""Schema-driven synthetic conversation generator.

This package generates realistic provider-format test data from annotated
JSON schemas.

Submodules:
    - ``core`` — SyntheticCorpus facade and batch/report entrypoints
    - ``selection`` — package-aware provider/element selection
    - ``builders`` — conversation builders and wire-format shaping
    - ``runtime`` — recursive schema-driven value emission
    - ``semantic_values`` — Semantic-role-driven value generation
    - ``relations`` — Relational constraint satisfaction
    - ``wire_formats`` — WireFormat, TreeConfig, PROVIDER_WIRE_FORMATS
    - ``showcase`` — ConversationTheme and curated synthetic themes
"""

from polylogue.schemas.synthetic.core import (
    SyntheticCorpus,
    SyntheticGenerationBatch,
    SyntheticGenerationReport,
)
from polylogue.schemas.synthetic.wire_formats import (
    PROVIDER_WIRE_FORMATS,
    TreeConfig,
    WireFormat,
)

__all__ = [
    # Core
    "SyntheticCorpus",
    "SyntheticGenerationBatch",
    "SyntheticGenerationReport",
    # Wire formats
    "PROVIDER_WIRE_FORMATS",
    "TreeConfig",
    "WireFormat",
]
