# mypy: disable-error-code="no-untyped-def,call-arg,arg-type,attr-defined"

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.archive.message.roles import Role
from polylogue.pipeline import prepare as prepare_module
from polylogue.pipeline.prepare_models import (
    AttachmentMaterializationPlan,
    PersistedConversationResult,
    PreparedBundle,
    SaveResult,
)
from polylogue.sources.parsers.base import ParsedConversation, ParsedMessage
from polylogue.types import Provider


def _conversation(*, messages: list[ParsedMessage] | None = None) -> ParsedConversation:
    return ParsedConversation(
        provider_name=Provider.UNKNOWN,
        provider_conversation_id="conv-1",
        title="Conversation",
        created_at="2026-04-23T00:00:00Z",
        updated_at="2026-04-23T00:00:00Z",
        messages=messages if messages is not None else [],
        attachments=[],
    )


@pytest.mark.asyncio
async def test_save_bundle_passes_records_to_repository_and_wraps_counts() -> None:
    bundle = SimpleNamespace(
        conversation="conversation-record",
        messages=["message-record"],
        attachments=["attachment-record"],
        content_blocks=["content-block-record"],
    )
    repository = SimpleNamespace(
        save_conversation=AsyncMock(
            return_value={
                "conversations": 1,
                "messages": 2,
                "attachments": 3,
                "skipped_conversations": 4,
                "skipped_messages": 5,
                "skipped_attachments": 6,
            }
        )
    )

    result = await prepare_module.save_bundle(bundle, repository=repository)

    assert result == SaveResult(
        conversations=1,
        messages=2,
        attachments=3,
        skipped_conversations=4,
        skipped_messages=5,
        skipped_attachments=6,
    )
    repository.save_conversation.assert_awaited_once_with(
        conversation="conversation-record",
        messages=["message-record"],
        attachments=["attachment-record"],
        content_blocks=["content-block-record"],
    )


@pytest.mark.asyncio
async def test_prepare_bundle_requires_context_and_can_construct_repository(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prepare_bundle requires a repository or backend"):
        await prepare_module.prepare_bundle(_conversation(), "source", archive_root=tmp_path)

    backend = SimpleNamespace()
    repository = SimpleNamespace(backend=backend)
    transform = SimpleNamespace(candidate_cid="cid-1")
    prepared = PreparedBundle(
        bundle=SimpleNamespace(),
        materialization_plan=AttachmentMaterializationPlan(),
        cid="cid-1",
        changed=False,
    )
    with patch("polylogue.storage.repository.ConversationRepository", return_value=repository) as repo_cls:
        with patch("polylogue.pipeline.prepare.transform_to_records", return_value=transform) as transform_to_records:
            with patch(
                "polylogue.pipeline.prepare._build_single_cache", new=AsyncMock(return_value="cache")
            ) as build_cache:
                with patch("polylogue.pipeline.prepare.enrich_bundle_from_db", return_value=prepared) as enrich:
                    result = await prepare_module.prepare_bundle(
                        _conversation(messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="hi")]),
                        "source",
                        archive_root=tmp_path,
                        backend=backend,
                    )

    assert result == prepared
    repo_cls.assert_called_once_with(backend=backend)
    transform_to_records.assert_called_once()
    build_cache.assert_awaited_once()
    enrich.assert_called_once()


@pytest.mark.asyncio
async def test_prepare_bundle_uses_repository_backend_and_skips_cache_build_when_cache_is_supplied(
    tmp_path: Path,
) -> None:
    repository = SimpleNamespace(backend="repo-backend")
    transform = SimpleNamespace(candidate_cid="cid-1")
    prepared = PreparedBundle(
        bundle=SimpleNamespace(),
        materialization_plan=AttachmentMaterializationPlan(),
        cid="cid-1",
        changed=True,
    )
    with patch("polylogue.pipeline.prepare.transform_to_records", return_value=transform):
        with patch("polylogue.pipeline.prepare._build_single_cache") as build_cache:
            with patch("polylogue.pipeline.prepare.enrich_bundle_from_db", return_value=prepared) as enrich:
                result = await prepare_module.prepare_bundle(
                    _conversation(messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="hi")]),
                    "source",
                    archive_root=tmp_path,
                    repository=repository,
                    cache="cache",
                    raw_id="raw-1",
                )

    assert result == prepared
    build_cache.assert_not_called()
    assert enrich.call_args.args[3] == "cache"
    assert enrich.call_args.kwargs["raw_id"] == "raw-1"


@pytest.mark.asyncio
async def test_persist_prepared_bundle_rolls_back_failed_moves_and_cleans_duplicates(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_text("source", encoding="utf-8")
    target = tmp_path / "target.bin"
    duplicate = tmp_path / "duplicate.bin"
    duplicate.write_text("dup", encoding="utf-8")
    prepared = PreparedBundle(
        bundle=SimpleNamespace(),
        materialization_plan=AttachmentMaterializationPlan(
            move_before_save=[(source, target)],
            delete_after_save=[duplicate],
        ),
        cid="cid-1",
        changed=True,
    )

    def materialize(source_path: Path, target_path: Path) -> None:
        target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")

    with patch("polylogue.pipeline.prepare.materialize_attachment_path", side_effect=materialize):
        with patch(
            "polylogue.pipeline.prepare.save_bundle",
            new=AsyncMock(
                return_value=SaveResult(
                    conversations=1,
                    messages=2,
                    attachments=1,
                    skipped_conversations=0,
                    skipped_messages=0,
                    skipped_attachments=0,
                )
            ),
        ):
            result = await prepare_module.persist_prepared_bundle(prepared, repository=SimpleNamespace())

    assert result == PersistedConversationResult(
        conversation_id="cid-1",
        save_result=SaveResult(
            conversations=1,
            messages=2,
            attachments=1,
            skipped_conversations=0,
            skipped_messages=0,
            skipped_attachments=0,
        ),
        content_changed=True,
    )
    assert not duplicate.exists()

    with patch("polylogue.pipeline.prepare.materialize_attachment_path", side_effect=materialize):
        with patch("polylogue.pipeline.prepare.save_bundle", new=AsyncMock(side_effect=RuntimeError("save failed"))):
            with patch("polylogue.pipeline.prepare.move_attachment_to_archive") as move_back:
                with pytest.raises(RuntimeError, match="save failed"):
                    await prepare_module.persist_prepared_bundle(prepared, repository=SimpleNamespace())

    move_back.assert_called_once_with(target, source)


@pytest.mark.asyncio
async def test_prepare_records_requires_context_handles_empty_conversations_and_delegates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prepare_records requires a repository or backend"):
        await prepare_module.prepare_records(_conversation(), "source", archive_root=tmp_path)

    empty_repo = SimpleNamespace(backend="repo-backend")
    with patch("polylogue.pipeline.prepare.logger.debug") as debug:
        empty = await prepare_module.prepare_records(
            _conversation(),
            "source",
            archive_root=tmp_path,
            repository=empty_repo,
        )

    assert empty.save_result.skipped_conversations == 1
    assert empty.content_changed is False
    debug.assert_called_once()

    repository = SimpleNamespace(backend="repo-backend")
    backend = SimpleNamespace()
    persisted = PersistedConversationResult(
        conversation_id="cid-1",
        save_result=SaveResult(
            conversations=1,
            messages=1,
            attachments=0,
            skipped_conversations=0,
            skipped_messages=0,
            skipped_attachments=0,
        ),
        content_changed=True,
    )
    with patch("polylogue.storage.repository.ConversationRepository", return_value=repository) as repo_cls:
        with patch(
            "polylogue.pipeline.prepare.prepare_bundle", new=AsyncMock(return_value="prepared")
        ) as prepare_bundle:
            with patch(
                "polylogue.pipeline.prepare.persist_prepared_bundle", new=AsyncMock(return_value=persisted)
            ) as persist:
                result = await prepare_module.prepare_records(
                    _conversation(messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text="hi")]),
                    "source",
                    archive_root=tmp_path,
                    backend=backend,
                    raw_id="raw-1",
                    cache="cache",
                )

    assert result == persisted
    repo_cls.assert_called_once_with(backend=backend)
    prepare_bundle.assert_awaited_once()
    assert prepare_bundle.await_args is not None
    assert prepare_bundle.await_args.kwargs["raw_id"] == "raw-1"
    assert prepare_bundle.await_args.kwargs["cache"] == "cache"
    persist.assert_awaited_once_with("prepared", repository=repository)
