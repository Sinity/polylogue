"""The shared prose and serialized-payload token estimator."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

#: Ordinary words (up to ``_ONE_WORD_MAX_CHARS``) count as one estimated word.
#: A longer unbroken run (``"!" * 100000``, a hash, a base64 blob) is weighted
#: by its length at ``_CHARS_PER_ESTIMATED_WORD`` instead of collapsing into one.
_ONE_WORD_MAX_CHARS = 16
_CHARS_PER_ESTIMATED_WORD = 8


def _estimated_words_length(length: int) -> int:
    return 1 if length <= _ONE_WORD_MAX_CHARS else -(-length // _CHARS_PER_ESTIMATED_WORD)


def estimate_tokens(text: str) -> int:
    """Stable prose proxy used by both context and compact renderers."""

    return tokens_from_estimated_words(estimate_token_words(text))


def estimate_token_words(text: str) -> int:
    """Additive word charge for fragments separated by whitespace."""
    return sum(_estimated_words_length(len(match.group())) for match in re.finditer(r"\S+", text))


def tokens_from_estimated_words(words: int) -> int:
    """Apply the shared token conversion after adding fragment charges."""
    return max(1, int(words * 1.3)) if words else 0


def estimate_serialized_tokens(serialized: str) -> int:
    """Estimate a serialized payload, charging for structure as well as words.

    Compact JSON has no whitespace, so a word count sees any number of omission
    objects as roughly one word. The UTF-8 byte component (about four bytes per
    token) keeps punctuation, identifiers and non-Latin text from being free.
    This is a tokenizer-free estimate, not a model-specific count.
    """

    if not serialized:
        return 0
    return max(estimate_tokens(serialized), (len(serialized.encode("utf-8")) + 3) // 4)


@dataclass(frozen=True)
class TextMeasure:
    """Compose the existing byte/word estimator without joining JSON rows."""

    length: int = 0
    byte_count: int = 0
    word_units: int = 0
    head: int = 0
    tail: int = 0
    all_word: bool = True

    @classmethod
    def text(cls, text: str) -> TextMeasure:
        head = re.match(r"\S*", text)
        tail = re.search(r"\S*\Z", text)
        assert head is not None and tail is not None
        return cls(
            len(text),
            len(text.encode("utf-8")),
            estimate_token_words(text),
            len(head.group()),
            len(tail.group()),
            len(head.group()) == len(text),
        )

    def then(self, other: TextMeasure) -> TextMeasure:
        units = self.word_units + other.word_units
        if self.tail and other.head:
            units += (
                _estimated_words_length(self.tail + other.head)
                - _estimated_words_length(self.tail)
                - _estimated_words_length(other.head)
            )
        return TextMeasure(
            self.length + other.length,
            self.byte_count + other.byte_count,
            units,
            self.head + other.head if self.all_word else self.head,
            self.tail + other.tail if other.all_word else other.tail,
            self.all_word and other.all_word,
        )

    @property
    def tokens(self) -> int:
        return max(tokens_from_estimated_words(self.word_units), (self.byte_count + 3) // 4)


def measure(parts: Iterator[str]) -> TextMeasure:
    result = TextMeasure()
    for part in parts:
        result = result.then(TextMeasure.text(part))
    return result
