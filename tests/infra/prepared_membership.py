"""Publish a membership classification through the canonical prepared route.

Retained publication decides a membership cohort's head on an original seal,
applies the Index outcome under that seal's mutation scope, then publishes the
matching Source acknowledgements as one staged mutation on the owner's
dedicated Source writer. Laws that supply their own
classification use exactly that sequence here instead of the retired store
write-back, which wrote Source rows inside the Index publication.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

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
    )
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead, prepare_session_write
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation
    from tests.infra.prepared_replay import publish_prepared_source

    root = archive.archive_root
    store = BlobStore(root / "blob")
    accepted = tuple(classification.accepted_raw_ids)
    parsed_by_raw_id = {raw_id: _parse_bound(session) for raw_id, session in parsed_by_raw_id.items()}
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
    archive.commit()
    # The Source acknowledgements (decisions and any incomplete-cohort parse
    # correction) form one staged mutation published on the owner's writer.
    publish_prepared_source(
        root,
        "test.membership-classification",
        lambda source_seal: prepare_membership_classification_source(
            source_seal,
            logical_source_key,
            classification,
            decisions=decisions,
            decided_at_ms=decided_at_ms,
            projections=projections_by_raw_id,
        ),
    )
    return session_id, actual


def _parse_bound(session: ParsedSession) -> ParsedSession:
    """Bind the semantic digest as retained preparation does before preparing a write.

    ``prepare_retained_jsonl_artifact`` binds every parsed session's content
    hash first, so the prepared write (over timestamp-normalized rows) and the
    membership projection carry the same digest.
    """
    from polylogue.pipeline.ids import bound_session_content_hash, session_content_hash

    if bound_session_content_hash(session) is not None:
        return session
    bound = session.model_copy()
    bound.content_hash = session_content_hash(session)
    return bound


def write_prepared_retained_session(
    archive: ArchiveStore,
    session: ParsedSession,
    *,
    raw_id: str,
    source_index: int = 0,
    revision_authoritative: bool = False,
    preacquired_attachment_blobs: Mapping[object, tuple[bytes | None, int, str]] | None = None,
) -> ArchiveRawParsedWriteResult:
    """Write one retained session through the guarded retained writer.

    The write is prepared on an original seal and published under that seal's
    Index mutation scope, as retained work-event publication does; the
    precedence and membership guards run on that canonical route. The optional
    attachment map is the upstream acquisition result keyed by carrier identity.
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
                preacquired_attachment_blobs=preacquired_attachment_blobs or {},
                finalize_raw_parse=False,
                revision_authoritative=revision_authoritative,
                prepared_required=True,
                prepared_write=prepared_write,
            )


def apply_prepared_aggregate_replay(
    archive: ArchiveStore,
    plan: Any,
    parsed_by_raw_id: dict[str, ParsedSession],
    *,
    acquired_at_ms: int,
) -> tuple[str, tuple[str, ...]]:
    """Publish one byte chain as retained publication does: one prepared aggregate.

    The composed aggregate is prepared on an original seal (its timestamps
    normalized against the tip's retained file time) and the writer consumes
    exactly that carrier: ``prepared_aggregate_rows``/``prepared_aggregate_session``
    and its prepared write, as ``apply_prepared_revision_replay`` passes them in
    production.
    """
    from polylogue.sources.dispatch import merge_parsed_session_chunks
    from polylogue.storage.sqlite.archive_tiers.revision_governance import (
        prepare_revision_replay_outcome,
        prepared_raw_revision_file_mtime,
    )
    from polylogue.storage.sqlite.archive_tiers.write import PreparedSessionSourceRead, prepare_session_write
    from polylogue.storage.sqlite.reference_seal import PreparedIndexMutation

    accepted = tuple(plan.accepted_raw_ids)
    if not accepted:
        raise ValueError("a prepared aggregate replay requires an accepted raw chain")
    chunks = [parsed_by_raw_id[raw_id] for raw_id in accepted]
    composed = chunks if len(chunks) == 1 else merge_parsed_session_chunks(chunks)
    if len(composed) != 1:
        raise ValueError("a prepared aggregate replay must compose exactly one session")
    aggregate = composed[0]
    tip = accepted[-1]
    root = archive.archive_root
    archive.commit()
    with PreparedIndexMutation(archive.index_db_path, archive_root=root) as seal:
        with seal.original_read_snapshot(), seal.source_producer():
            read = PreparedSessionSourceRead(seal, blob_store=BlobStore(root / "blob"))
            adoption = read.prepare_raw_revision_replay_adoption(
                [aggregate], logical_source_key=plan.logical_source_key, raw_ids=accepted
            )
            prepared_write = prepare_session_write(
                seal.observer("index"),
                aggregate,
                merge_append=False,
                fallback_timestamp=prepared_raw_revision_file_mtime(seal, tip),
                source_read=read,
                raw_id=tip,
                force_replace=True,
                before_input=seal.before_index_input,
            )
            outcome = prepare_revision_replay_outcome(
                seal,
                read,
                plan,
                adoption,
                aggregate_session=aggregate,
                aggregate_content_hash=prepared_write.rows.session_content_hash,
                prepared_write=prepared_write,
            )
        with archive.index_mutation_scope(prepared_seal=seal):
            return archive.apply_raw_revision_replay(
                plan,
                parsed_by_raw_id,
                prepared_outcome=outcome,
                acquired_at_ms=acquired_at_ms,
                prepared_aggregate_rows=prepared_write.rows,
                prepared_aggregate_session=aggregate,
                preacquired_aggregate_attachment_blobs={},
                preacquired_attachment_blobs_by_raw_id={raw_id: {} for raw_id in accepted},
                prepared_required_raw_ids=frozenset({tip}),
                prepared_write=prepared_write,
            )
