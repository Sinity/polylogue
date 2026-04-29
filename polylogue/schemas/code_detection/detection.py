"""Public code detection surface."""

from polylogue.schemas.code_detection.extractors import (
    extract_code_block,
    extract_code_block_from_dict,
)
from polylogue.schemas.code_detection.regex import LANGUAGE_PATTERNS
from polylogue.schemas.code_detection.regex import regex_scores as _regex_scores
from polylogue.schemas.code_detection.runtime import detect_language

__all__ = [
    "LANGUAGE_PATTERNS",
    "_regex_scores",
    "detect_language",
    "extract_code_block",
    "extract_code_block_from_dict",
]
