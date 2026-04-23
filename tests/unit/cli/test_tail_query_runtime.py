from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.config import Config, get_config
from polylogue.pipeline.services.parsing import ParsingService
from polylogue.services import RuntimeServices, build_runtime_services
from polylogue.sync_bridge import run_coroutine_sync


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


def test_cli_tail_list_json_exposes_tail_provenance(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del workspace_env
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    session_path = _write_claude_code_session(
        home / ".claude" / "projects" / "project-cli" / "session.jsonl",
        assistant_text="stable cli answer",
    )

    runner = CliRunner()
    config, services = run_coroutine_sync(_configured_services())
    try:
        run_coroutine_sync(_ingest_claude_code_source(config, services))
        _write_claude_code_session(session_path, assistant_text="tail cli answer")

        result = runner.invoke(
            cli,
            [
                "--tail",
                "--plain",
                "--provider",
                "claude-code",
                "--latest",
                "list",
                "--format",
                "json",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert len(payload) == 1
        assert payload[0]["tail"]["archive_state"] == "ahead_of_archive"
        assert payload[0]["tail"]["source_path"] == str(session_path)
    finally:
        run_coroutine_sync(services.close())


def test_cli_tail_errors_when_no_supported_source_exists(
    workspace_env: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    del workspace_env
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    result = CliRunner().invoke(
        cli,
        ["--tail", "--plain", "list"],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "supports configured Claude Code sources only" in result.output
