"""End-to-end CLI coverage for the file-edits/agent-policies read views.

polylogue-nua7: ``file_edits`` (76,105 live rows) and ``session_agent_policies``
had a complete, tested read chain that terminated at the repository layer with
no surface consumer above it. These tests exercise the real ``read --view
file-edits``/``read --view agent-policies`` verbs end to end -- real
``ArchiveStore`` write, real CLI invocation, real JSON render -- not the
storage function in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli as click_cli


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


def test_read_view_file_edits_surfaces_structured_patch(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.core.enums import BlockType, Provider, Role
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedFileEdit, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))

    with ArchiveStore(archive_root) as archive_db:
        parsed = ParsedSession(
            source_name=Provider.CLAUDE_CODE,
            provider_session_id="cli-file-edit-1",
            title="CLI file-edits view session",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.ASSISTANT,
                    position=0,
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TOOL_USE,
                            tool_name="Edit",
                            tool_id="edit-tool-1",
                            tool_input={"file_path": "/tmp/foo.py"},
                        ),
                    ],
                ),
                ParsedMessage(
                    provider_message_id="m2",
                    role=Role.USER,
                    position=1,
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TOOL_RESULT,
                            tool_id="edit-tool-1",
                            text="applied",
                            file_edit=ParsedFileEdit(
                                file_path="/tmp/foo.py",
                                structured_patch=[
                                    {"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 2, "lines": ["+x"]}
                                ],
                                original_file="old contents\n",
                                old_string="old",
                                new_string="new",
                                replace_all=False,
                                user_modified=True,
                            ),
                        ),
                    ],
                ),
            ],
        )
        archive_db.write_raw_and_parsed(
            parsed,
            payload=b'{"raw": "claude payload"}',
            source_path="/tmp/raw.jsonl",
            acquired_at_ms=1735689600000,
        )

    session_id = "claude-code-session:cli-file-edit-1"

    result = cli_runner.invoke(
        click_cli,
        ["--plain", "--id", session_id, "read", "--view", "file-edits", "-f", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["total"] == 1
    edit = payload["file_edits"][0]
    assert edit["file_path"] == "/tmp/foo.py"
    assert edit["original_file"] == "old contents\n"
    assert edit["old_string"] == "old"
    assert edit["new_string"] == "new"
    assert edit["structured_patch"] == [{"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 2, "lines": ["+x"]}]

    missing = cli_runner.invoke(
        click_cli,
        ["--plain", "--id", "claude-code-session:does-not-exist", "read", "--view", "file-edits"],
        catch_exceptions=False,
    )
    assert "not found" in missing.output.lower()


def test_read_view_agent_policies_surfaces_sandbox_facts(
    cli_runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.core.enums import BlockType, Provider, Role
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))

    with ArchiveStore(archive_root) as archive_db:
        parsed = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="cli-agent-policy-1",
            title="CLI agent-policies view session",
            messages=[
                ParsedMessage(
                    provider_message_id="m1",
                    role=Role.USER,
                    text="run it",
                    position=0,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="run it")],
                ),
            ],
            session_events=[
                ParsedSessionEvent(
                    event_type="agent_policy",
                    timestamp="2026-01-01T00:00:01+00:00",
                    payload={
                        "approval_policy": "never",
                        "sandbox_policy": "danger-full-access",
                        "network_policy": "true",
                    },
                ),
            ],
        )
        archive_db.write_raw_and_parsed(
            parsed,
            payload=b'{"raw": "codex payload"}',
            source_path="/tmp/raw.jsonl",
            acquired_at_ms=1735689600000,
        )

    session_id = "codex-session:cli-agent-policy-1"

    result = cli_runner.invoke(
        click_cli,
        ["--plain", "--id", session_id, "read", "--view", "agent-policies", "-f", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["total"] == 1
    policy = payload["agent_policies"][0]
    assert policy["approval_policy"] == "never"
    assert policy["sandbox_policy"] == "danger-full-access"
    assert policy["network_policy"] == "true"
