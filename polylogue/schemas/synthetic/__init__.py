"""Schema-driven synthetic conversation generator.

This package generates realistic provider-format test data from annotated
JSON schemas.  All public symbols from the original monolithic
``synthetic.py`` are re-exported here so that existing imports like
``from polylogue.schemas.synthetic import SyntheticCorpus`` continue to
work unchanged.

Submodules:
    - ``core`` — SyntheticCorpus facade and batch/report entrypoints
    - ``selection`` — package-aware provider/element selection
    - ``builders`` — conversation builders and wire-format shaping
    - ``runtime`` — recursive schema-driven value emission
    - ``semantic_values`` — Semantic-role-driven value generation
    - ``relations`` — Relational constraint satisfaction
    - ``wire_formats`` — WireFormat, TreeConfig, PROVIDER_WIRE_FORMATS
    - ``showcase`` — ConversationTheme, _SHOWCASE_THEMES
"""

from polylogue.schemas.synthetic.core import (
    SyntheticCorpus,
    SyntheticGenerationBatch,
    SyntheticGenerationReport,
)
from polylogue.schemas.synthetic.relations import (
    ForeignKeyGraph,
    MutualExclusionGroup,
    RelationConstraintSolver,
    StringLengthConstraint,
    TimeDeltaConstraint,
)
from polylogue.schemas.synthetic.semantic_values import (
    _ROLE_TEXTS,
    SemanticValueGenerator,
    _text_for_role,
)
from polylogue.schemas.synthetic.showcase import _SHOWCASE_THEMES, ConversationTheme
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
    # Showcase
    "ConversationTheme",
    "_SHOWCASE_THEMES",
    # Semantic values
    "SemanticValueGenerator",
    "_ROLE_TEXTS",
    "_text_for_role",
    # Relations
    "ForeignKeyGraph",
    "MutualExclusionGroup",
    "RelationConstraintSolver",
    "StringLengthConstraint",
    "TimeDeltaConstraint",
]
