"""Publish a membership classification through the canonical prepared route.

Retained publication decides a membership cohort on one original seal: the
Index head reduction and the Source acknowledgements are prepared together,
the Index outcome is applied under that seal's mutation scope, and the staged
Source permit is published afterwards. Laws that supply their own
classification use exactly that sequence here instead of the retired store
write-back, which wrote Source rows inside the Index publication.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from polylogue.storage.blob_store import BlobStore

if TYPE_CHECKING:
    from polylogue.archive.session_revision_membership import MembershipClassification, MembershipDecision
    from polylogue.pipeline.ids import SessionRevisionProjection
    from polylogue.sources.parsers.base import ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.revision_governance import ArchiveRawParsedWriteResult


def publish_prepared_membership_classification(
    archive: ArchiveStore,
    logical_source_key: str,
    classification: MembershipClassification,
    parsed_by_raw_id: Mapping[str, ParsedSession],
    projections_by_raw_id: Mapping[str, SessionRevisionProjection],
    *,
    decided_at_ms: int,
) -> tuple[str | None, dict[str, MembershipDecision]]:
    """Apply one supplied classification on its original seal; return Index outcome."""
    from polylogue.storage.sqlite.archive_tiers.revision_governance import (
        apply_prepared_membership_index,
        membership_decisions_for_head_plan,
        prepare_membership_classification_source,
        prepare_membership_head_plan,
        prepared_raw_revision_file_mtime,
        publish_prepared_revision_source,
    )
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead, prepare_session_write
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    root = archive.archive_root
    store = BlobStore(root / "blob")
    accepted = tuple(classification.accepted_raw_ids)
    blobs: dict[object, tuple[bytes | None, int, str]] = {}
    if accepted:
        for attachment in parsed_by_raw_id[accepted[-1]].attachments:
            if attachment.inline_bytes is None:
                continue
            blob_hash, size = store.write_from_bytes(attachment.inline_bytes)
            blobs[id(attachment)] = (bytes.fromhex(blob_hash), size, "acquired")
    archive.commit()
    with PreparedIndexMutation(archive.index_db_path, archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            read = PreparedSessionSourceRead(seal, blob_store=store)
            head_plan = prepare_membership_head_plan(
                seal.observer("index"),
                read,
                logical_source_key,
                classification,
                before_input=seal.before_index_input,
            )
            decisions = membership_decisions_for_head_plan(classification, head_plan, suppressed=False)
            prepared_write = None
            if accepted and head_plan.yield_to_head_raw_id is None:
                prepared_write = prepare_session_write(
                    seal.observer("index"),
                    parsed_by_raw_id[accepted[-1]],
                    merge_append=False,
                    fallback_timestamp=prepared_raw_revision_file_mtime(seal, accepted[-1]),
                    source_read=read,
                    raw_id=accepted[-1],
                    force_replace=True,
                    before_input=seal.before_index_input,
                )
            prepare_membership_classification_source(
                seal,
                logical_source_key,
                classification,
                decisions=decisions,
                decided_at_ms=decided_at_ms,
            )
        permit = seal.prepare_source_mutation()
        with archive.index_mutation_scope(prepared_seal=seal):
            session_id, actual = apply_prepared_membership_index(
                archive,
                logical_source_key,
                classification,
                parsed_by_raw_id,
                projections_by_raw_id,
                head_plan,
                decided_at_ms=decided_at_ms,
                preacquired_attachment_blobs=blobs,
                prepared_write=prepared_write,
            )
        if actual != decisions:
            raise AssertionError(f"membership Index outcome {actual!r} differs from its Source decisions {decisions!r}")
        publish_prepared_revision_source(seal, permit)
    return session_id, actual


def write_prepared_retained_session(
    archive: ArchiveStore,
    session: ParsedSession,
    *,
    raw_id: str,
    source_index: int = 0,
    revision_authoritative: bool = False,
) -> ArchiveRawParsedWriteResult:
    """Write one retained session through the guarded retained writer.

    The write is prepared on an original seal and published under that seal's
    Index mutation scope, as retained work-event publication does; the
    precedence and membership guards run on that canonical route.
    """
    from polylogue.storage.sqlite.archive_tiers.revision_governance import (
        _index_parsed_for_retained_raw,
        prepared_raw_revision_file_mtime,
    )
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead, prepare_session_write
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    root = archive.archive_root
    archive.commit()
    with PreparedIndexMutation(archive.index_db_path, archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            prepared_write = prepare_session_write(
                seal.observer("index"),
                session,
                merge_append=False,
                fallback_timestamp=prepared_raw_revision_file_mtime(seal, raw_id),
                source_read=PreparedSessionSourceRead(seal, blob_store=BlobStore(root / "blob")),
                raw_id=raw_id,
                force_replace=False,
                before_input=seal.before_index_input,
            )
        with archive.index_mutation_scope(prepared_seal=seal):
            return _index_parsed_for_retained_raw(
                archive,
                session,
                raw_id=raw_id,
                source_index=source_index,
                stage_timings_s=None,
                stage_timing_prefix="test.retained-write",
                manage_transaction=False,
                preacquired_attachment_blobs={},
                finalize_raw_parse=False,
                revision_authoritative=revision_authoritative,
                prepared_required=True,
                prepared_write=prepared_write,
            )
