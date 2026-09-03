"""The declared semantics version gates every route that reads domain bytes,
and it is only readable as an exact integer.

A reader that trusts a record whose shape it does not implement produces
silently wrong domain facts rather than a typed refusal, so the gate must
cover single-anchor resolution as well as the full verification pass, and a
non-integral declaration must not be coerced down into a supported version.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from polylogue.material_protocol.v1 import (
    EncodedRevision,
    RevisionManifest,
    UnsupportedSemanticsVersionError,
    encode_session_revision,
    resolve_anchor,
)
from polylogue.material_protocol.v1.constants import SEMANTICS_VERSION
from tests.unit.material_protocol.v1.fixture import SMALL_SESSION_REVISION_CREATED_AT, build_small_session_material

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "fixtures" / "material_protocol" / "v1" / "small-session"


@pytest.fixture
def encoded() -> EncodedRevision:
    return encode_session_revision(
        build_small_session_material(),
        revision_created_at=SMALL_SESSION_REVISION_CREATED_AT,
        max_records_per_segment=4,
    )


def _any_record_id(manifest: RevisionManifest) -> str:
    return sorted(manifest.anchors)[0]


def test_resolve_anchor_refuses_an_unsupported_semantics_version(encoded: EncodedRevision) -> None:
    """Anchor resolution is version-gated, not merely digest-checked.

    Anti-vacuity: drop the ``require_current_semantics`` call from
    ``resolve_anchor`` and this test fails with the record returned instead of
    raising -- the anchor digest still matches, because only the declared
    version was mutated.
    """
    record_id = _any_record_id(encoded.manifest)
    stale = dataclasses.replace(encoded.manifest, semantics_version=SEMANTICS_VERSION - 1)

    with pytest.raises(UnsupportedSemanticsVersionError):
        resolve_anchor(stale, encoded.segments, record_id)

    # The same call on the unmutated manifest resolves, so the refusal above is
    # attributable to the version and to nothing else.
    assert resolve_anchor(encoded.manifest, encoded.segments, record_id)["record_id"] == record_id


@pytest.mark.parametrize(
    "declared",
    [
        pytest.param(float(SEMANTICS_VERSION) + 0.9, id="non-integral-float"),
        pytest.param(str(SEMANTICS_VERSION), id="string"),
        pytest.param(True, id="bool"),
    ],
)
def test_manifest_refuses_a_non_integer_semantics_version(encoded: EncodedRevision, declared: object) -> None:
    """A declared version is read exactly, never coerced into a supported one.

    Anti-vacuity: restore ``int(payload["semantics_version"])`` in
    ``RevisionManifest.from_dict`` and the non-integral-float case fails with
    no exception raised -- ``int(4.9)`` truncates to the current version and
    passes the later equality gate.
    """
    payload = encoded.manifest.to_dict()
    payload["semantics_version"] = declared  # type: ignore[assignment]

    with pytest.raises(UnsupportedSemanticsVersionError):
        RevisionManifest.from_dict(payload)


def test_attachment_record_fields_are_pinned_to_the_declared_semantics_version(
    encoded: EncodedRevision,
) -> None:
    """The attachment record shape is pinned, so changing it forces a bump.

    Anti-vacuity: add or remove a key in ``records.attachment_record`` without
    touching this list and the key-set assertion fails; change
    ``SEMANTICS_VERSION`` without revisiting the shape and the version
    assertion fails. Together they make a silent record-shape drift -- the
    exact defect that shipped ``direction``/``producer_ref``/``caption`` under
    v3 -- impossible to land.
    """
    records = [
        json.loads(line) for raw in encoded.segments.values() for line in raw.decode("utf-8").splitlines() if line
    ]
    attachments = [record for record in records if record["kind"] == "attachment"]
    assert attachments, "fixture encodes no attachment record to pin"

    assert SEMANTICS_VERSION == 4
    assert set(attachments[0]) == {
        "kind",
        "record_id",
        "seq",
        "session_id",
        "message_id",
        "position",
        "attachment_id",
        "display_name",
        "media_type",
        "byte_count",
        "blob_sha256",
        "acquisition_status",
        "upload_origin",
        "direction",
        "producer_ref",
        "caption",
        "source_url",
    }


def test_checked_in_fixture_declares_the_current_semantics_version() -> None:
    """The cross-repo fixture is regenerated in lockstep with the version.

    Anti-vacuity: bump ``SEMANTICS_VERSION`` without regenerating
    ``tests/fixtures/material_protocol/v1/small-session/`` and this fails,
    which is what leaves the Sinex counterpart pinned to bytes we no longer
    produce.
    """
    payload = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert payload["semantics_version"] == SEMANTICS_VERSION
