from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.config import Config, get_config
from polylogue.lib.query.spec import ConversationQuerySpec
from polylogue.pipeline.services.parsing import ParsingService
from polylogue.pipeline.tail_overlay import tail_overlay_services
from polylogue.services import RuntimeServices, build_runtime_services
from polylogue.types import Provider


def _write_claude_code_session(
    path: Path,
    *,
    session_id: str = "session-1",
    user_text: str = "hello",
    assistant_text: str = "hi",
) -> Path:
    entries = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": session_id,
            "message": {"content": user_text},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "sessionId": session_id,
            "message": {"content": assistant_text},
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
    return path


async def _configured_services() -> tuple[Config, RuntimeServices]:
    config = get_config()
    services = build_runtime_services(config=config, db_path=config.db_path)
    return config, services


async def _ingest_claude_code_source(config: Config, services: RuntimeServices) -> None:
    claude_source = next(source for source in config.sources if source.name == "claude-code")
    parsing_service = ParsingService(
        repository=services.get_repository(),
        archive_root=config.archive_root,
        config=config,
    )
    await parsing_service.parse_sources([claude_source])


@pytest.mark.asyncio
async def test_tail_overlay_replaces_archived_state_with_newer_source_state(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del workspace_env
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    session_path = _write_claude_code_session(
        home / ".claude" / "projects" / "project-a" / "session.jsonl",
        assistant_text="stable archive answer",
    )

    config, services = await _configured_services()
    try:
        await _ingest_claude_code_source(config, services)
        _write_claude_code_session(session_path, assistant_text="tail only replacement text")

        spec = ConversationQuerySpec(
            providers=(Provider.CLAUDE_CODE,),
            latest=True,
        )

        base_results = await spec.list(services.get_repository())
        assert len(base_results) == 1
        assert "stable archive answer" in base_results[0].to_text(include_role=False)

        async with tail_overlay_services(services) as overlay_services:
            results = await spec.list(overlay_services.get_repository())

        assert len(results) == 1
        assert "tail only replacement text" in results[0].to_text(include_role=False)
        assert results[0].tail_overlay is not None
        assert results[0].tail_overlay.archive_state == "ahead_of_archive"
        assert results[0].tail_overlay.source_path == str(session_path)
    finally:
        await services.close()


@pytest.mark.asyncio
async def test_tail_overlay_reports_unseen_session_state(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del workspace_env
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    session_path = _write_claude_code_session(
        home / ".claude" / "projects" / "project-b" / "session.jsonl",
        assistant_text="unseen tail session",
    )

    config, services = await _configured_services()
    try:
        spec = ConversationQuerySpec(providers=(Provider.CLAUDE_CODE,), latest=True)
        assert await spec.list(services.get_repository()) == []

        async with tail_overlay_services(services) as overlay_services:
            results = await spec.list(overlay_services.get_repository())

        assert len(results) == 1
        assert results[0].tail_overlay is not None
        assert results[0].tail_overlay.archive_state == "unseen"
        assert results[0].tail_overlay.source_path == str(session_path)
    finally:
        await services.close()


@pytest.mark.asyncio
async def test_tail_overlay_falls_back_to_stable_archive_when_source_parse_fails(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del workspace_env
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    session_path = _write_claude_code_session(
        home / ".claude" / "projects" / "project-c" / "session.jsonl",
        assistant_text="stable archive survives",
    )

    config, services = await _configured_services()
    try:
        await _ingest_claude_code_source(config, services)
        session_path.write_text('{"type":"assistant","uuid":"broken"\n', encoding="utf-8")

        spec = ConversationQuerySpec(providers=(Provider.CLAUDE_CODE,), latest=True)

        async with tail_overlay_services(services) as overlay_services:
            results = await spec.list(overlay_services.get_repository())

        assert len(results) == 1
        assert results[0].tail_overlay is None
        assert "stable archive survives" in results[0].to_text(include_role=False)
    finally:
        await services.close()


@pytest.mark.asyncio
async def test_tail_overlay_delta_disappears_after_archive_catchup(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del workspace_env
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    session_path = _write_claude_code_session(
        home / ".claude" / "projects" / "project-d" / "session.jsonl",
        assistant_text="first archived answer",
    )

    config, services = await _configured_services()
    try:
        await _ingest_claude_code_source(config, services)
        _write_claude_code_session(session_path, assistant_text="caught up answer")
        await _ingest_claude_code_source(config, services)

        spec = ConversationQuerySpec(providers=(Provider.CLAUDE_CODE,), latest=True)

        async with tail_overlay_services(services) as overlay_services:
            results = await spec.list(overlay_services.get_repository())

        assert len(results) == 1
        assert results[0].tail_overlay is None
        assert "caught up answer" in results[0].to_text(include_role=False)
    finally:
        await services.close()
