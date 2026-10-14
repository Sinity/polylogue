"""Finite, injective representations of observed hashed categories.

Measured category shapes preserve their conditional length weights. A schema
with only marginals has no measured category-to-length assignment; its finite
codebook chooses among the representable shapes declared by those marginals.
"""

from __future__ import annotations

import random
import unicodedata
from collections.abc import Callable, Iterator, Mapping
from itertools import islice

from polylogue.schemas.field_stats.distributions import CATEGORICAL_BUCKET_COUNT
from polylogue.schemas.synthetic.models import GeneratedString
from polylogue.schemas.synthetic.text_value import SampledText, joined_text_value, text_equals


def _neutral_characters(character_class: str | None = None) -> Iterator[str]:
    """Printable NFC-stable symbols, with no private or surrogate characters."""
    if character_class == "ascii":
        yield from (chr(point) for point in range(0x21, 0x80) if chr(point).isprintable() and not chr(point).isspace())
        return
    if character_class not in (None, "non_ascii"):
        raise ValueError("unknown anonymous character class")
    for point in range(0x80 if character_class == "non_ascii" else 0x21, 0x110000):
        character = chr(point)
        if character.isprintable() and not character.isspace() and unicodedata.normalize("NFC", character) == character:
            yield character


def _ordinal_codes(length: int, radix: int, *, start: int = 0) -> Iterator[list[int]]:
    """Retain significant digits only, independent of the declared text tail."""
    ordinal = start
    while True:
        digits: list[int] = []
        remaining = ordinal
        while remaining:
            remaining, digit = divmod(remaining, radix)
            digits.append(digit)
        if len(digits) > length:
            return
        yield digits
        ordinal += 1


def _string_shape_candidates(
    length: int,
    newlines: int,
    *,
    prefix: str = "",
    phase: int = 0,
    ordinal: int = 0,
    character_class: str | None = None,
) -> Iterator[tuple[int, int, GeneratedString]]:
    """Borrow the original spelling sequence at an exact native checkpoint."""
    if character_class not in (None, "ascii", "non_ascii"):
        raise ValueError("unknown anonymous character class")
    if character_class == "ascii" and not prefix.isascii():
        return
    body_class = None if character_class == "non_ascii" and not prefix.isascii() else character_class
    body_length = length - newlines - len(prefix)
    if body_length < 0:
        return
    tail = SampledText(newlines, "a", newlines=newlines)
    if body_length == 0:
        if phase == 0 and ordinal == 0 and body_class != "non_ascii":
            yield 0, 0, joined_text_value(prefix, "", tail)
        return
    alphabet = tuple(islice(_neutral_characters(body_class), CATEGORICAL_BUCKET_COUNT))
    alphabet_members = frozenset(alphabet)
    if phase == 0:
        for index, digits in enumerate(_ordinal_codes(body_length, len(alphabet), start=ordinal), ordinal):
            body = joined_text_value(
                "".join(alphabet[digit] for digit in digits), SampledText(body_length - len(digits), alphabet[0])
            )
            if not isinstance(body, str) or unicodedata.normalize("NFC", body) == body:
                yield 0, index, joined_text_value(prefix, body, tail)
        ordinal = 0
    # The fallback is the same complete native shape domain, after the neutral
    # alphabet. Coordinates count candidates before NFC/filter rejection.
    scalar_count = 128 if body_class == "ascii" else 0x110000 - 0x800
    for index, digits in enumerate(_ordinal_codes(body_length, scalar_count, start=ordinal), ordinal):
        body = joined_text_value(
            "".join(chr(point if point < 0xD800 else point + 0x800) for point in digits),
            SampledText(body_length - len(digits), chr(0)),
        )
        if not isinstance(body, str):
            continue
        if (
            "\n" not in body
            and not all(character in alphabet_members for character in body)
            and (body_class != "non_ascii" or not body.isascii())
            and unicodedata.normalize("NFC", body) == body
        ):
            yield 1, index, joined_text_value(prefix, body, tail)


def _strings_of_shape(
    length: int, newlines: int, *, prefix: str = "", character_class: str | None = None
) -> Iterator[GeneratedString]:
    """Spell anonymous classes with exact measured lengths and newline counts."""
    for _, _, value in _string_shape_candidates(length, newlines, prefix=prefix, character_class=character_class):
        yield value


def categorical_codebook(
    buckets: list[tuple[int, int]],
    lengths: Mapping[int, int],
    *,
    seed: str,
    newlines: Mapping[int, int] | None = None,
    category_shapes: Mapping[int, tuple[Mapping[int, int], Mapping[int, int]]] | None = None,
    category_character_shapes: Mapping[int, Mapping[str, tuple[Mapping[int, int], Mapping[int, int], int]]]
    | None = None,
    excluded: frozenset[str] = frozenset(),
    prefix: str = "",
    accepts: Callable[[GeneratedString], bool] | None = None,
) -> list[tuple[GeneratedString, float]]:
    """Allocate each category a distinct value with a supported length.

    Each length iterator supplies its actual capacity (including exactly one
    empty string). Exhausting all of them proves nonrepresentability; a random
    draw can never turn a representable vocabulary into a refusal.
    """
    if category_shapes is not None:
        # Each measured bucket may carry several lengths. Allocate a spelling
        # for each supported shape, using shared iterators to keep them distinct.
        # Bucket hashes select anonymous classes; no observed token is recovered.
        measured_streams: dict[tuple[int, int, str | None], Iterator[GeneratedString]] = {}
        measured_pool: list[tuple[GeneratedString, float]] = []
        for bucket, count in sorted(buckets):
            partitions = (
                category_character_shapes[bucket]
                if category_character_shapes is not None
                else {None: (*category_shapes[bucket], count)}
            )
            shapes: dict[tuple[int, int, str | None], float] = {}
            for character_class, (length_weights, newline_weights, observations) in partitions.items():
                supported = {
                    (length, newline, character_class): length_weight * newline_weight
                    for length, length_weight in length_weights.items()
                    for newline, newline_weight in newline_weights.items()
                    if newline <= length and (character_class != "non_ascii" or newline < length)
                }
                denominator = sum(supported.values())
                if not denominator:
                    raise ValueError("character class cannot represent its original text shape")
                for measured_shape, measured_weight in supported.items():
                    shapes[measured_shape] = observations * measured_weight / denominator
            total = sum(shapes.values())
            if not total:
                raise ValueError("categorical shape cannot represent its observed newlines")
            for measured_shape, partition_weight in sorted(
                shapes.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or "")
            ):
                stream = measured_streams.setdefault(
                    measured_shape,
                    _strings_of_shape(
                        measured_shape[0], measured_shape[1], prefix=prefix, character_class=measured_shape[2]
                    ),
                )
                value = next(
                    (
                        value
                        for value in stream
                        if not any(text_equals(value, item) for item in excluded)
                        and (accepts is None or accepts(value))
                    ),
                    None,
                )
                if value is None:
                    raise ValueError("categorical vocabulary cannot be represented by the observed string lengths")
                measured_pool.append((value, count * partition_weight / total))
        return measured_pool
    newline_weights = newlines or {0: 1}
    weights = {
        (length, count): length_weight * newline_weight
        for length, length_weight in sorted(lengths.items())
        for count, newline_weight in sorted(newline_weights.items())
        if count <= length
    }
    streams = {shape: _strings_of_shape(*shape, prefix=prefix) for shape in weights}
    available = {
        shape: next(
            (
                value
                for value in stream
                if not any(text_equals(value, item) for item in excluded) and (accepts is None or accepts(value))
            ),
            None,
        )
        for shape, stream in streams.items()
    }
    pool: list[tuple[GeneratedString, float]] = []
    rng = random.Random(seed)
    for _, count in sorted(buckets):
        choices = [shape for shape, value in available.items() if value is not None]
        if not choices:
            raise ValueError("categorical vocabulary cannot be represented by the observed string lengths")
        shape = rng.choices(choices, weights=[weights[shape] for shape in choices], k=1)[0]
        value = available[shape]
        assert value is not None
        pool.append((value, count))
        available[shape] = next(
            (
                value
                for value in streams[shape]
                if not any(text_equals(value, item) for item in excluded) and (accepts is None or accepts(value))
            ),
            None,
        )
    return pool
