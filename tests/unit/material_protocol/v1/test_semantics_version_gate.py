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

    assert SEMANTICS_VERSION == 5
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


def test_message_usage_distinguishes_unknown_from_measured_zero(encoded: EncodedRevision) -> None:
    """v5's value domain: an absent counter is ``null``, a declared zero is ``0``.

    The keys of the message record's ``usage`` object did not change in v5 --
    only what an absent counter serializes to -- so the attachment key-set pin
    above cannot see this drift. A consumer that reads ``input_tokens`` as an
    integer is broken by ``null`` exactly as it would be by a renamed key, so
    the distinction is pinned to the declared version here.

    Anti-vacuity: restore ``input_tokens: int = 0`` (and its three siblings) on
    ``MessageInput`` -- the pre-#5293 defaults -- and the unknown-counter
    assertions fail with ``0 is not None``, because the encoder then fabricates
    a measured zero for a message the fixture never gave counters to.
    """
    assert SEMANTICS_VERSION == 5

    messages = {
        record["record_id"]: record
        for raw in encoded.segments.values()
        for line in raw.decode("utf-8").splitlines()
        if line
        for record in [json.loads(line)]
        if record["kind"] == "message"
    }

    # msg-1 declares no per-message counters at all.
    unknown = messages["claude-code-session:demo-session-1:n:msg-1"]["usage"]
    assert set(unknown) == {
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "duration_ms",
    }
    assert unknown == dict.fromkeys(unknown, None)

    # msg-2 declares input/output but no cache counters: known and unknown
    # coexist inside one usage object.
    partial = messages["claude-code-session:demo-session-1:n:msg-2"]["usage"]
    assert partial["input_tokens"] == 120
    assert partial["output_tokens"] == 40
    assert partial["cache_read_tokens"] is None
    assert partial["cache_write_tokens"] is None


def test_a_declared_zero_counter_is_not_encoded_as_unknown() -> None:
    """The other half of v5: an explicit 0 must survive as ``0``, not ``null``.

    Anti-vacuity: make ``message_record`` coerce falsy counters
    (``message.input_tokens or None``) and this fails with ``None == 0``, which
    is the mirror defect -- a measured zero laundered back into unknown.
    """
    material = build_small_session_material()
    zeroed = dataclasses.replace(material.messages[0], input_tokens=0, output_tokens=0)
    material = dataclasses.replace(material, messages=(zeroed, *material.messages[1:]))

    encoded = encode_session_revision(
        material, revision_created_at=SMALL_SESSION_REVISION_CREATED_AT, max_records_per_segment=4
    )
    record = next(
        json.loads(line)
        for raw in encoded.segments.values()
        for line in raw.decode("utf-8").splitlines()
        if line and json.loads(line)["record_id"] == "claude-code-session:demo-session-1:n:msg-1"
    )
    assert record["usage"]["input_tokens"] == 0
    assert record["usage"]["output_tokens"] == 0
    assert record["usage"]["cache_read_tokens"] is None


def test_checked_in_fixture_declares_the_current_semantics_version() -> None:
    """The cross-repo fixture is regenerated in lockstep with the version.

    Anti-vacuity: bump ``SEMANTICS_VERSION`` without regenerating
    ``tests/fixtures/material_protocol/v1/small-session/`` and this fails,
    which is what leaves the Sinex counterpart pinned to bytes we no longer
    produce.
    """
    payload = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert payload["semantics_version"] == SEMANTICS_VERSION
