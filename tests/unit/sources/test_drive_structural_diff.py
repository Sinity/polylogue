"""Contracts for the JSON-structural-diff classifier (polylogue-1fijp AC (b)).

``classify_drive_structural_relation`` is the fix for the exact gap PR #3656
(polylogue-sp72) documented and deferred: Drive's live-attachment backfill
re-serializes the WHOLE JSON document on every re-acquisition pass that
resolves a Drive-hosted attachment reference, so a genuinely-grown
conversation is never a byte-prefix superset of its predecessor -- the
existing byte-prefix classifier (``archive/revision_authority.py``) can only
ever see it as ambiguous. These tests prove the structural classifier's
semantics directly (no SQLite/archive machinery involved -- pure JSON-in,
enum-out), including one fixture built through the REAL production
attachment-injection code path
(``polylogue.sources.drive.attachment_fetch.fetch_live_drive_attachment_bytes``),
not a hand-rolled approximation of it.
"""

from __future__ import annotations

import json

from polylogue.core.json import JSONValue
from polylogue.sources.drive.attachment_fetch import fetch_live_drive_attachment_bytes
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


def test_real_attachment_injection_fixture_is_structural_growth() -> None:
    """Build the before/after bytes through the REAL production injector
    (``fetch_live_drive_attachment_bytes``), not a hand-rolled dict edit --
    proves the classifier against the actual mutation shape Drive produces,
    matching PR #3656's finding that this shape is not byte-prefix provable."""
    payload: JSONValue = {
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "Hi"},
                {
                    "role": "model",
                    "text": "Here is the file",
                    "driveDocument": {"id": "att-1", "name": "doc.txt", "mimeType": "text/plain"},
                },
            ]
        }
    }
    old_bytes = _bytes(payload)
    resolved, stats = fetch_live_drive_attachment_bytes(payload, lambda file_id: b"the actual attachment bytes")
    assert stats.fetched_count == 1
    new_bytes = json.dumps(resolved).encode("utf-8")

    assert new_bytes != old_bytes
    assert not new_bytes.startswith(old_bytes)  # confirms the real non-byte-prefix shape
    assert classify_drive_structural_relation(old_bytes, new_bytes) is DriveStructuralRelation.STRUCTURAL_GROWTH


def test_two_independent_attachment_fetches_in_sequence_stay_growth() -> None:
    """A document with two Drive-hosted attachments, resolved one at a time
    across two re-acquisition passes (a realistic multi-attachment Drive
    session), stays structural growth at every step."""
    payload: JSONValue = {
        "chunkedPrompt": {
            "chunks": [
                {"role": "model", "driveDocument": {"id": "att-1"}},
                {"role": "model", "driveDocument": {"id": "att-2"}},
            ]
        }
    }
    gen0 = _bytes(payload)

    def fetch_att1_only(file_id: str) -> bytes:
        if file_id == "att-1":
            return b"first attachment bytes"
        raise RuntimeError("not yet fetchable")

    resolved1, stats1 = fetch_live_drive_attachment_bytes(payload, fetch_att1_only)
    assert stats1.fetched_count == 1
    gen1 = json.dumps(resolved1).encode("utf-8")
    assert classify_drive_structural_relation(gen0, gen1) is DriveStructuralRelation.STRUCTURAL_GROWTH

    resolved2, stats2 = fetch_live_drive_attachment_bytes(resolved1, lambda file_id: b"second attachment bytes")
    assert stats2.fetched_count == 1
    gen2 = json.dumps(resolved2).encode("utf-8")
    assert classify_drive_structural_relation(gen1, gen2) is DriveStructuralRelation.STRUCTURAL_GROWTH
    # And transitively across both generations at once.
    assert classify_drive_structural_relation(gen0, gen2) is DriveStructuralRelation.STRUCTURAL_GROWTH
