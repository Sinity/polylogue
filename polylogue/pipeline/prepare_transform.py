"""Pure parsed-session preparation helpers."""

from __future__ import annotations

from pathlib import Path

from polylogue.pipeline.ids import attachment_content_id, session_content_hashes
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.pipeline.prepare_models import (
    AttachmentMaterializationPlan,
    TransformResult,
)
from polylogue.sources.parsers.base import ParsedAttachment, ParsedSession


def plan_attachment_materialization(
    source_path: str | None,
    target_path: str | None,
) -> AttachmentMaterializationPlan:
    if not source_path or not target_path or source_path == target_path:
        return AttachmentMaterializationPlan()

    source = Path(source_path)
    target = Path(target_path)
    if not source.exists():
        return AttachmentMaterializationPlan()
    if target.exists():
        return AttachmentMaterializationPlan(delete_after_save=[source])
    return AttachmentMaterializationPlan(move_before_save=[(source, target)])


def _prepared_attachment(
    source_name: str,
    attachment: ParsedAttachment,
    *,
    archive_root: Path,
    materialization_plan: AttachmentMaterializationPlan,
) -> ParsedAttachment:
    _attachment_id, updated_meta, updated_path = attachment_content_id(
        source_name,
        attachment,
        archive_root=archive_root,
    )
    attachment_plan = plan_attachment_materialization(attachment.path, updated_path)
    materialization_plan.move_before_save.extend(attachment_plan.move_before_save)
    materialization_plan.delete_after_save.extend(attachment_plan.delete_after_save)
    return attachment.model_copy(update={"provider_meta": updated_meta, "path": updated_path})


def transform_to_records(convo: ParsedSession, source_name: str, *, archive_root: Path) -> TransformResult:
    content_hash, _message_hashes = session_content_hashes(convo)
    candidate_cid = make_session_id(convo.source_name, convo.provider_session_id)
    materialization_plan = AttachmentMaterializationPlan()

    prepared_session = convo.model_copy(
        update={
            "attachments": [
                _prepared_attachment(
                    source_name,
                    attachment,
                    archive_root=archive_root,
                    materialization_plan=materialization_plan,
                )
                for attachment in convo.attachments
            ],
        }
    )
    return TransformResult(
        session=prepared_session,
        materialization_plan=materialization_plan,
        content_hash=content_hash,
        candidate_cid=candidate_cid,
    )


__all__ = [
    "transform_to_records",
]
