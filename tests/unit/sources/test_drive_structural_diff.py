"""Contracts for the JSON-structural-diff classifier (polylogue-1fijp AC (b)).

``classify_drive_structural_relation`` proves growth for Drive documents that
the provider re-serializes whole on every save (AI Studio rewrites the entire
JSON when a conversation grows), so a genuinely grown conversation is never a
byte-prefix superset of its predecessor and the byte-prefix classifier
(``archive/revision_authority.py``) can only see it as ambiguous. These tests
prove the classifier's semantics directly: pure JSON in, enum out.
"""

from __future__ import annotations

import json

from polylogue.sources.drive.structural_diff import (
    DriveStructuralRelation,
    classify_drive_structural_relation,
)


def _bytes(doc: object) -> bytes:
    return json.dumps(doc).encode("utf-8")


def test_identical_bytes_are_identical() -> None:
    payload = _bytes({"chunkedPrompt": {"chunks": [{"role": "user", "text": "hi"}]}})
    assert classify_drive_structural_relation(payload, payload) is DriveStructuralRelation.IDENTICAL


def test_identical_structure_different_key_order_is_identical() -> None:
    """Structural comparison decodes both sides -- key order is not byte order."""
    old = _bytes({"a": 1, "b": 2})
    new = _bytes({"b": 2, "a": 1})
    assert old != new
    assert classify_drive_structural_relation(old, new) is DriveStructuralRelation.IDENTICAL


def test_new_trailing_chunk_is_structural_growth() -> None:
    """A conversation that gained a new turn: old chunks list is a strict
    positional prefix of the new one."""
    old = _bytes({"chunkedPrompt": {"chunks": [{"role": "user", "text": "hi"}]}})
    new = _bytes(
        {
            "chunkedPrompt": {
                "chunks": [
                    {"role": "user", "text": "hi"},
                    {"role": "model", "text": "hello"},
                ]
            }
        }
    )
    assert classify_drive_structural_relation(old, new) is DriveStructuralRelation.STRUCTURAL_GROWTH


def test_enriched_existing_chunk_is_structural_growth() -> None:
    """Same chunk count, but an existing chunk's dict gained a key (the real
    attachment-injection shape) without changing anything that was already
    there -- this is the exact case a byte-prefix classifier cannot prove."""
    old = _bytes(
        {
            "chunkedPrompt": {
                "chunks": [
                    {"role": "model", "driveDocument": {"id": "att-1", "name": "doc.txt"}},
                ]
            }
        }
    )
    new = _bytes(
        {
            "chunkedPrompt": {
                "chunks": [
                    {
                        "role": "model",
                        "driveDocument": {"id": "att-1", "name": "doc.txt", "__fetchedData": "aGVsbG8="},
                    },
                ]
            }
        }
    )
    assert not new.startswith(old)  # confirms this is genuinely not a byte-prefix relation
    assert classify_drive_structural_relation(old, new) is DriveStructuralRelation.STRUCTURAL_GROWTH


def test_combined_enrichment_and_new_trailing_chunk_is_structural_growth() -> None:
    old = _bytes(
        {
            "chunkedPrompt": {
                "chunks": [
                    {"role": "user", "text": "hi"},
                    {"role": "model", "driveDocument": {"id": "att-1"}},
                ]
            }
        }
    )
    new = _bytes(
        {
            "chunkedPrompt": {
                "chunks": [
                    {"role": "user", "text": "hi"},
                    {"role": "model", "driveDocument": {"id": "att-1", "__fetchedData": "AA=="}},
                    {"role": "user", "text": "thanks"},
                ]
            }
        }
    )
    assert classify_drive_structural_relation(old, new) is DriveStructuralRelation.STRUCTURAL_GROWTH


def test_null_to_populated_value_is_structural_growth() -> None:
    old = _bytes({"cachedContent": None, "chunkedPrompt": {"chunks": []}})
    new = _bytes({"cachedContent": {"id": "cache-1"}, "chunkedPrompt": {"chunks": []}})
    assert classify_drive_structural_relation(old, new) is DriveStructuralRelation.STRUCTURAL_GROWTH


def test_new_top_level_key_is_structural_growth() -> None:
    old = _bytes({"chunkedPrompt": {"chunks": []}})
    new = _bytes({"chunkedPrompt": {"chunks": []}, "runSettings": {"temperature": 0.5}})
    assert classify_drive_structural_relation(old, new) is DriveStructuralRelation.STRUCTURAL_GROWTH


def test_changed_scalar_value_is_ambiguous() -> None:
    """A real content mutation (not growth) must never be accepted as growth."""
    old = _bytes({"chunkedPrompt": {"chunks": [{"role": "user", "text": "hi"}]}})
    new = _bytes({"chunkedPrompt": {"chunks": [{"role": "model", "text": "hi"}]}})
    assert classify_drive_structural_relation(old, new) is DriveStructuralRelation.AMBIGUOUS


def test_removed_trailing_chunk_is_ambiguous() -> None:
    old = _bytes(
        {
            "chunkedPrompt": {
                "chunks": [
                    {"role": "user", "text": "hi"},
                    {"role": "model", "text": "hello"},
                ]
            }
        }
    )
    new = _bytes({"chunkedPrompt": {"chunks": [{"role": "user", "text": "hi"}]}})
    assert classify_drive_structural_relation(old, new) is DriveStructuralRelation.AMBIGUOUS


def test_populated_to_null_is_ambiguous() -> None:
    """A value regressing from populated to null is a real change, not growth."""
    old = _bytes({"cachedContent": {"id": "cache-1"}})
    new = _bytes({"cachedContent": None})
    assert classify_drive_structural_relation(old, new) is DriveStructuralRelation.AMBIGUOUS


def test_reordered_chunks_is_ambiguous() -> None:
    """Growth is positional-prefix only, matching the byte-prefix classifier's
    'extends as a prefix' semantics one level up -- reordering existing
    entries is never growth, even though the same elements are all present."""
    old = _bytes({"chunks": [{"n": 1}, {"n": 2}]})
    new = _bytes({"chunks": [{"n": 2}, {"n": 1}]})
    assert classify_drive_structural_relation(old, new) is DriveStructuralRelation.AMBIGUOUS


def test_unrelated_documents_are_ambiguous() -> None:
    old = _bytes({"chunkedPrompt": {"chunks": [{"role": "user", "text": "hi"}]}})
    new = _bytes({"totally": "different", "shape": [1, 2, 3]})
    assert classify_drive_structural_relation(old, new) is DriveStructuralRelation.AMBIGUOUS


def test_non_json_bytes_are_ambiguous_not_a_crash() -> None:
    assert classify_drive_structural_relation(b"not json", b"also not json {") is DriveStructuralRelation.AMBIGUOUS
    assert classify_drive_structural_relation(b"", _bytes({"a": 1})) is DriveStructuralRelation.AMBIGUOUS
