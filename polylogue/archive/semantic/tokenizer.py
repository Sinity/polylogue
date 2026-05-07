"""Client-side token estimation fallback when provider-reported counts are absent.

Uses word-count-based estimation (words * 1.3 ≈ tokens) as a simple
fallback. Provider-reported token counts always take precedence over
this estimator.

Tokenizer provenance is recorded so consumers can distinguish
provider-reported values from estimated ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Provenance = Literal["provider_reported", "tokenizer_estimated", "heuristic_estimated", "unknown"]
Confidence = Literal["reported", "estimated", "unknown"]

TOKENIZER_VERSION = "word-count-1.3-v1"


@dataclass(frozen=True, slots=True)
class TokenEstimate:
    """Token count estimate with provenance tracking."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    provenance: Provenance = "unknown"
    confidence: Confidence = "unknown"
    tokenizer_version: str = TOKENIZER_VERSION

    @property
    def billable_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


_WORDS_PER_TOKEN = 1.3


def estimate_tokens_from_words(word_count: int, *, tokenizer_version: str = TOKENIZER_VERSION) -> TokenEstimate:
    """Estimate token count from word count using the words-to-tokens ratio."""
    estimated = max(int(word_count * _WORDS_PER_TOKEN), 0)
    return TokenEstimate(
        input_tokens=estimated,
        total_tokens=estimated,
        provenance="heuristic_estimated",
        confidence="estimated",
        tokenizer_version=tokenizer_version,
    )


def estimate_tokens_from_words_split(
    input_words: int,
    output_words: int,
    cache_read_words: int = 0,
    cache_write_words: int = 0,
    *,
    tokenizer_version: str = TOKENIZER_VERSION,
) -> TokenEstimate:
    """Estimate token counts from word counts with input/output split."""
    input_tokens = max(int(input_words * _WORDS_PER_TOKEN), 0)
    output_tokens = max(int(output_words * _WORDS_PER_TOKEN), 0)
    cache_read_tokens = max(int(cache_read_words * _WORDS_PER_TOKEN), 0)
    cache_write_tokens = max(int(cache_write_words * _WORDS_PER_TOKEN), 0)
    return TokenEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        total_tokens=input_tokens + output_tokens + cache_read_tokens + cache_write_tokens,
        provenance="heuristic_estimated",
        confidence="estimated",
        tokenizer_version=tokenizer_version,
    )


def token_estimate_from_text(text: str | None, *, tokenizer_version: str = TOKENIZER_VERSION) -> TokenEstimate:
    """Estimate token count from raw text string."""
    if not text:
        return TokenEstimate(
            provenance="heuristic_estimated",
            confidence="estimated",
            tokenizer_version=tokenizer_version,
        )
    word_count = len(text.split())
    return estimate_tokens_from_words(word_count, tokenizer_version=tokenizer_version)


__all__ = [
    "TOKENIZER_VERSION",
    "Confidence",
    "Provenance",
    "TokenEstimate",
    "estimate_tokens_from_words",
    "estimate_tokens_from_words_split",
    "token_estimate_from_text",
]
