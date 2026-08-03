"""JSON-structural growth classifier for Drive re-acquisition bytes.

polylogue-1fijp AC (b): ``iter_drive_raw_data``'s live-attachment backfill
(``_inject_live_drive_attachment_bytes``) re-serializes the WHOLE Gemini/
AI-Studio JSON document on every re-acquisition pass that resolves a new
Drive-hosted attachment reference, rather than byte-appending at the
document's end (see ``_bind_drive_revision_lineage`` in
``pipeline/services/ingest_batch/_core.py`` and its citing PR #3656,
polylogue-sp72). The archive's existing revision classifier
(``archive/revision_authority.py``'s ``classify_historical_full_revision_streams``
and the byte-relation arms in ``storage/sqlite/archive_tiers/raw_admission.py``)
is strictly ``bytes.startswith()``-based, so this realistic growth shape is
*never* a byte-prefix superset of its predecessor even when the underlying
conversation genuinely only grew -- it lands ``ambiguous``/``quarantined``
with the byte classifier alone.

This module proves growth at the JSON-structural level instead of the byte
level: one JSON document is a *structural extension* of another when every
piece of information the old document asserts is still present, unchanged,
in the new one -- new dict keys, new trailing list elements (e.g. new
``chunkedPrompt.chunks`` entries for a conversation that grew), and values
that moved from absent/``null`` to populated are all allowed growth; any
value that existed and *changed* (a scalar mutated, a list element removed
or reordered, a dict key's value replaced with something not itself a
structural extension of the old value) breaks the proof and the relation is
``AMBIGUOUS`` -- exactly the same fail-closed posture as the byte-prefix
classifier it complements, just proven a different way.

This is a pure, standalone comparison over two JSON documents. It knows
nothing about ``raw_sessions``, revision envelopes, or SQLite -- callers
(``_bind_drive_revision_lineage``) are responsible for resolving which two
raws to compare and for deciding what typed lineage a ``STRUCTURAL_GROWTH``
verdict earns.
"""

from __future__ import annotations

from enum import StrEnum

from polylogue.core.json import JSONDecodeError, JSONValue, loads


class DriveStructuralRelation(StrEnum):
    """The exhaustive, typed outcome of comparing two JSON documents."""

    #: Byte-identical payloads (also true, vacuously, of identical JSON
    #: structures reached via different byte encodings -- e.g. re-serialized
    #: with different key order -- since this module compares decoded
    #: values, not bytes).
    IDENTICAL = "identical"
    #: ``new`` is a proven structural extension of ``old``: every key/value
    #: and list element ``old`` asserts is still present, unchanged, in
    #: ``new``, which additionally carries new dict keys, new trailing list
    #: elements, and/or populated values where ``old`` had ``null``/absent.
    STRUCTURAL_GROWTH = "structural_growth"
    #: Neither identical nor a proven extension -- some old value changed,
    #: was removed, or the payloads are not both valid JSON. Never treated
    #: as growth; callers must fall back to their existing ambiguity
    #: handling (typed refusal, not a silent accept).
    AMBIGUOUS = "ambiguous"


def _is_structural_extension(old: JSONValue, new: JSONValue) -> bool:
    """Return whether ``new`` structurally contains everything ``old`` asserts.

    Recursive, positional-prefix growth for lists (mirrors the byte-prefix
    classifier's "extends as a prefix" semantics one level up: a genuinely
    new conversation turn is a *trailing* addition, never an insertion or
    reorder), key-superset growth for dicts (new keys are always allowed;
    every existing key's value must itself satisfy this relation), exact
    equality for scalars, and a `None -> anything` escape hatch for the one
    legitimate "field was unpopulated, now resolved" shape this module
    exists to recognize (e.g. a Drive-hosted attachment reference that had
    no fetched bytes yet).
    """
    if old == new:
        return True
    if old is None:
        return True
    if isinstance(old, dict) and isinstance(new, dict):
        return all(key in new and _is_structural_extension(value, new[key]) for key, value in old.items())
    if isinstance(old, list) and isinstance(new, list):
        if len(new) < len(old):
            return False
        return all(_is_structural_extension(old_item, new[index]) for index, old_item in enumerate(old))
    # Differing types (including old being a scalar and new a
    # dict/list, or two different scalar values) are never growth --
    # only a `None -> anything` widening is allowed, and that is
    # already handled above.
    return False


def classify_drive_structural_relation(old_bytes: bytes, new_bytes: bytes) -> DriveStructuralRelation:
    """Classify the structural relation of ``new_bytes`` to ``old_bytes``.

    Both inputs must decode as JSON for anything but ``AMBIGUOUS`` to be
    possible -- a decode failure on either side is itself proof of nothing
    and always yields ``AMBIGUOUS``, never a crash the caller has to guard
    against.
    """
    if old_bytes == new_bytes:
        return DriveStructuralRelation.IDENTICAL
    try:
        old_doc = loads(old_bytes)
        new_doc = loads(new_bytes)
    except JSONDecodeError:
        return DriveStructuralRelation.AMBIGUOUS
    if old_doc == new_doc:
        return DriveStructuralRelation.IDENTICAL
    if _is_structural_extension(old_doc, new_doc):
        return DriveStructuralRelation.STRUCTURAL_GROWTH
    return DriveStructuralRelation.AMBIGUOUS


__all__ = [
    "DriveStructuralRelation",
    "classify_drive_structural_relation",
]
