"""Schema and extraction infrastructure for polylogue.

Extraction path:
- `unified.py` + `lib/viewports.py` — Full viewport extraction producing
  `HarmonizedMessage` with tool_calls, reasoning_traces, content_blocks, etc.

Supporting modules:
- `schema_inference.py` — Infer JSON schemas from data samples
- `validator.py` — Validate provider exports against schemas with drift detection
- `providers/` — Generated JSON schemas for each provider
"""

# Re-export main types for convenience
from polylogue.schemas.unified import HarmonizedMessage
from polylogue.schemas.validator import SchemaValidator, ValidationResult, validate_provider_export

__all__ = [
    "HarmonizedMessage",
    "SchemaValidator",
    "ValidationResult",
    "validate_provider_export",
]
