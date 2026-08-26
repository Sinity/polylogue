from polylogue.archive.attachment.availability import (
    AttachmentAvailabilityState,
    resolve_attachment_availability,
)


def test_acquisition_status_does_not_certify_availability() -> None:
    result = resolve_attachment_availability(
        blob_hash="ab" * 32,
        acquisition_status="acquired",
        verify=lambda _hash: False,
        exists=lambda _hash: False,
    )
    assert result.state is AttachmentAvailabilityState.MISSING
    assert not result.available
    assert not result.can_fetch


def test_readable_blob_is_available_and_fetchable() -> None:
    result = resolve_attachment_availability(
        blob_hash=bytes.fromhex("ab" * 32),
        acquisition_status="acquired",
        verify=lambda digest: digest == "ab" * 32,
    )
    assert result.state is AttachmentAvailabilityState.AVAILABLE
    assert result.reason == "readable-and-hash-valid"
    assert result.can_fetch


def test_physical_hash_mismatch_is_not_available() -> None:
    result = resolve_attachment_availability(
        blob_hash="ab" * 32,
        acquisition_status="acquired",
        verify=lambda _hash: False,
        exists=lambda _hash: True,
    )
    assert result.state is AttachmentAvailabilityState.HASH_MISMATCH


def test_typed_unfetched_unknown_generation_and_unauthorized() -> None:
    unfetched = resolve_attachment_availability(
        blob_hash=None,
        acquisition_status="unfetched",
        verify=lambda _hash: True,
    )
    wrong_generation = resolve_attachment_availability(
        blob_hash="ab" * 32,
        acquisition_status="acquired",
        verify=lambda _hash: True,
        generation_id="old",
        expected_generation_id="new",
    )
    unauthorized = resolve_attachment_availability(
        blob_hash="ab" * 32,
        acquisition_status="acquired",
        verify=lambda _hash: True,
        authorized=False,
    )
    assert unfetched.state is AttachmentAvailabilityState.UNFETCHED
    assert wrong_generation.reason == "wrong-generation"
    assert unauthorized.state is AttachmentAvailabilityState.UNAUTHORIZED
