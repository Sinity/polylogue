"""Structural fingerprint helpers shared by schema sampling and generation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, closing
from functools import cmp_to_key
from itertools import islice
from tempfile import TemporaryFile
from typing import BinaryIO, TypeAlias

from polylogue.core.json import JSONDocument
from polylogue.schemas.field_stats.stats import is_dynamic_key

_FINGERPRINT_MAX_DEPTH = 8
_FINGERPRINT_ARRAY_SAMPLE = 8

FingerprintLeaf: TypeAlias = tuple[str] | tuple[str, str]
FingerprintArray: TypeAlias = tuple[str, tuple["StructureFingerprint", ...]]
FingerprintObjectEntry: TypeAlias = tuple[str, "StructureFingerprint"]
FingerprintObject: TypeAlias = tuple[str, tuple[FingerprintObjectEntry, ...]]
StructureFingerprint: TypeAlias = FingerprintLeaf | FingerprintArray | FingerprintObject


def _structure_fingerprint(
    value: object,
    *,
    depth: int = 0,
    max_depth: int = _FINGERPRINT_MAX_DEPTH,
) -> StructureFingerprint:
    """Build a hashable structural fingerprint for schema-dedup heuristics."""
    if depth >= max_depth:
        return ("depth-limit", type(value).__name__)

    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool",)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ("number",)
    if isinstance(value, str):
        return ("string",)

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        # ``islice`` bounds consumption directly. ``value[:N]`` would resolve
        # slice bounds via ``len(value)`` first -- for a full-corpus record
        # stream (``ReplayableRecordSamples``) wrapping the whole raw session
        # as one top-level array, that forces a complete decode pass of the
        # entire backing file just to fingerprint its first few items.
        item_shapes = {
            _structure_fingerprint(item, depth=depth + 1, max_depth=max_depth)
            for item in islice(value, _FINGERPRINT_ARRAY_SAMPLE)
        }
        return ("array", tuple(sorted(item_shapes, key=repr)))

    if isinstance(value, dict):
        props = _object_fingerprint_entries(value, depth=depth, max_depth=max_depth)
        return ("object", tuple(props))

    return ("other", type(value).__name__)


def _object_fingerprint_entries(
    value: JSONDocument | Mapping[str, object],
    *,
    depth: int,
    max_depth: int,
) -> list[FingerprintObjectEntry]:
    props: list[FingerprintObjectEntry] = []
    for key in sorted(value):
        child = value[key]
        normalized_key = "*" if is_dynamic_key(key) else key
        props.append(
            (
                normalized_key,
                _structure_fingerprint(child, depth=depth + 1, max_depth=max_depth),
            )
        )
    return props


__all__ = ["_structure_fingerprint"]


def ordered_keys(value: Mapping[str, object]) -> Iterator[str]:
    """Order mapping keys using a spill's index when one owns the mapping."""
    from polylogue.schemas.observation_spill import SpilledObject

    return value.sorted_keys() if isinstance(value, SpilledObject) else iter(sorted(value))


def _compare_fingerprints(left: BinaryIO, right: BinaryIO) -> int:
    left.seek(0)
    right.seek(0)
    while True:
        a, b = left.read(65536), right.read(65536)
        if a != b:
            return -1 if a < b else 1
        if not a:
            return 0


def fingerprint_parts(value: object, *, depth: int = 0) -> Iterator[str]:
    """Emit the exact fingerprint repr without retaining its variable key tree."""
    from polylogue.schemas.observation_spill import SpilledArray, SpilledObject

    if depth >= _FINGERPRINT_MAX_DEPTH:
        name = (
            "dict"
            if isinstance(value, SpilledObject)
            else "list"
            if isinstance(value, SpilledArray)
            else type(value).__name__
        )
        yield repr(("depth-limit", name))
    elif isinstance(value, dict):
        yield "('object', ("
        count = 0

        def entries() -> Iterator[tuple[str, object]]:
            if isinstance(value, SpilledObject):
                with closing(value.structure_key_items(sorted_keys=True)) as children:
                    for key, child in children:
                        name = key.small_name
                        yield "*" if name is None or is_dynamic_key(name) else name, child
            else:
                for key in ordered_keys(value):
                    yield "*" if is_dynamic_key(key) else key, value[key]

        with closing(entries()) as children:
            for name, child in children:
                if count:
                    yield ", "
                yield "(" + repr(name) + ", "
                yield from fingerprint_parts(child, depth=depth + 1)
                yield ")"
                count += 1
        if count == 1:
            yield ","
        yield "))"
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        # The existing fingerprint uses exactly the first eight shapes. Their
        # complete repr can itself be large, so sort and deduplicate on disk.
        with ExitStack() as owned:
            files: list[BinaryIO] = []
            children = value.structure_values() if isinstance(value, SpilledArray) else value
            for item in islice(children, _FINGERPRINT_ARRAY_SAMPLE):
                output: BinaryIO = owned.enter_context(TemporaryFile())
                for part in fingerprint_parts(item, depth=depth + 1):
                    output.write(part.encode("utf-8"))
                files.append(output)
            files.sort(key=cmp_to_key(_compare_fingerprints))
            yield "('array', ("
            previous = None
            count = 0
            for output in files:
                if previous is not None and _compare_fingerprints(previous, output) == 0:
                    continue
                if count:
                    yield ", "
                output.seek(0)
                # Repr uses ASCII escapes for surrogate code units; UTF-8
                # chunk boundaries may split an ordinary Unicode key.
                import codecs

                decoder = codecs.getincrementaldecoder("utf-8")()
                while chunk := output.read(65536):
                    yield decoder.decode(chunk)
                yield decoder.decode(b"", final=True)
                previous = output
                count += 1
            if count == 1:
                yield ","
            yield "))"
    else:
        yield repr(_structure_fingerprint(value, depth=depth))
