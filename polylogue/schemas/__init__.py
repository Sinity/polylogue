"""Schema and extraction infrastructure for polylogue.

This package provides two extraction paths:

1. **Simple extraction** (extractors.py + common.py)
   - glom-based declarative specs
   - Produces `CommonMessage` - flat, minimal structure
   - Good for: validation, schema testing, simple queries

2. **Rich extraction** (unified.py + lib/viewports.py)
   - Manual extractors with full viewport support
   - Produces `HarmonizedMessage` - includes tool_calls, reasoning_traces, etc.
   - Good for: semantic analysis, cross-provider queries, full-featured rendering

Use `unified.py` for most purposes. Use `extractors.py` when you need
declarative glom specs or simpler CommonMessage output.

Additionally:
- `claude_code_records.py` - Metadata record types (progress, file-history-snapshot)
- `schema_inference.py` - Infer JSON schemas from data samples
- `providers/` - Generated JSON schemas for each provider
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
