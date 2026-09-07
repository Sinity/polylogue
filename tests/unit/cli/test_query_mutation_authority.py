"""The root query's tag/metadata writes are the daemon's, never the CLI's.

The daemon is the sole writer. ``--add-tag``/``--set`` lower to the declared
``mutation.session.tag``/``mutation.session.metadata`` operations; with no
daemon answering, the command refuses and leaves ``user.db`` untouched.

Anti-vacuity for every test here: restoring a CLI-side writable ``ArchiveStore``
makes the offline cases write the tag and return exit 0, so
:func:`test_matched_page_mutation_refuses_without_a_daemon` goes red. Sending a
different operation, or dropping the matched selection from the payload, turns
the lowering tests red.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner, Result

from polylogue.cli.click_app import cli
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.storage_records import SessionBuilder

_ORIGIN_FILTER = "origin:claude-ai-export"


@pytest.fixture
def tagged_archive(tmp_path: Path) -> Path:
    """Return an archive root holding three plain sessions."""
    initialize_active_archive_root(tmp_path)
    index_db = tmp_path / "index.db"
    for index in range(3):
        (
            SessionBuilder(index_db, f"conv-{index}")
            .provider("claude-ai")
            .title(f"Session {index}")
            .add_message(f"m{index}", role="user", text="hello alpha bravo")
            .save()
        )
    return tmp_path


def _run(archive_root: Path, *args: str) -> Result:
    env = {
        "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
        "POLYLOGUE_DB_PATH": str(archive_root / "index.db"),
        "POLYLOGUE_NO_DAEMON": "1",
    }
    return CliRunner().invoke(cli, ["--no-daemon", *args], env=env)


def _tagged_sessions(archive_root: Path, tag: str) -> list[object]:
    """Read the tag back through the query surface that would show a write.

    A zero-hit query is its own exit code, so the absence of the tag arrives
    as a diagnostics envelope rather than an empty ``items`` list.
    """
    listed = _run(archive_root, "--format", "json", "find", f"tag:{tag}")
    payload = json.loads(listed.output)
    items = payload.get("items", [])
    assert isinstance(items, list)
    return items


def _user_tier_digest(archive_root: Path) -> str:
    """Digest ``user.db`` and its journal: any local write changes it."""
    digest = hashlib.sha256()
    for path in sorted(archive_root.glob("user.db*")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.mark.parametrize(
    ("args", "operation"),
    [
        (("--add-tag", "triage"), "mutation.session.tag"),
        (("--set", "lane", "triage"), "mutation.session.metadata"),
    ],
)
def test_matched_page_mutation_refuses_without_a_daemon(
    tagged_archive: Path, args: tuple[str, ...], operation: str
) -> None:
    """No daemon means a typed refusal, not a silent second write authority."""
    before = _user_tier_digest(tagged_archive)

    result = _run(tagged_archive, *args, "find", _ORIGIN_FILTER)

    assert result.exit_code != 0, result.output
    assert "daemon is unavailable" in str(result.output) + str(result.exception)
    assert operation in str(result.output) + str(result.exception)
    assert _user_tier_digest(tagged_archive) == before
    assert _tagged_sessions(tagged_archive, "triage") == []


def test_combined_tag_and_metadata_refuses_without_a_daemon(tagged_archive: Path) -> None:
    """A combined mutation refuses on its first operation and writes nothing."""
    before = _user_tier_digest(tagged_archive)

    result = _run(tagged_archive, "--add-tag", "triage", "--set", "lane", "x", "find", _ORIGIN_FILTER)

    assert result.exit_code != 0, result.output
    assert _user_tier_digest(tagged_archive) == before
    assert _tagged_sessions(tagged_archive, "triage") == []


@pytest.mark.parametrize(
    ("args", "operation", "expected_payload_key", "expected_values"),
    [
        (("--add-tag", "triage"), "mutation.session.tag", "tags", ["triage"]),
        (("--set", "lane", "triage"), "mutation.session.metadata", "pairs", [["lane", "triage"]]),
    ],
)
def test_matched_page_mutation_lowers_to_its_declared_operation(
    tagged_archive: Path,
    args: tuple[str, ...],
    operation: str,
    expected_payload_key: str,
    expected_values: list[object],
) -> None:
    """The matched selection and its values reach the daemon as one operation."""
    issued: list[tuple[str, dict[str, object]]] = []

    def _served(_config: object, name: str, payload: dict[str, object]) -> dict[str, object]:
        issued.append((name, payload))
        return {"status": "ok", "affected_count": 3}

    with patch("polylogue.cli.archive_query._submit_mutation_operation", side_effect=_served):
        result = _run(tagged_archive, *args, "find", _ORIGIN_FILTER)

    assert result.exit_code == 0, result.output
    assert [name for name, _payload in issued] == [operation]
    payload = issued[0][1]
    selection = payload["session_ids"]
    assert isinstance(selection, list)
    assert sorted(str(item) for item in selection) == [
        "claude-ai-export:ext-conv-0",
        "claude-ai-export:ext-conv-1",
        "claude-ai-export:ext-conv-2",
    ]
    assert payload[expected_payload_key] == expected_values
    assert json.loads(result.output)["affected_count"] == 3


def test_combined_mutation_reports_both_halves_from_the_daemon(tagged_archive: Path) -> None:
    """Two operations, one combined receipt: metadata first, then tags."""
    issued: list[str] = []

    def _served(_config: object, name: str, _payload: dict[str, object]) -> dict[str, object]:
        issued.append(name)
        return {"status": "ok", "affected_count": 3}

    with patch("polylogue.cli.archive_query._submit_mutation_operation", side_effect=_served):
        result = _run(tagged_archive, "--add-tag", "triage", "--set", "lane", "x", "find", _ORIGIN_FILTER)

    assert result.exit_code == 0, result.output
    assert issued == ["mutation.session.metadata", "mutation.session.tag"]
    payload = json.loads(result.output)
    assert payload["operation"] == "mutate"
    assert payload["tag_count"] == 3
    assert payload["applied_count"] == 3


def test_single_session_tag_route_uses_the_same_operation(tagged_archive: Path) -> None:
    """Tagging one resolved session lowers to the matched-page operation."""
    issued: list[tuple[str, dict[str, object]]] = []

    def _served(_config: object, name: str, payload: dict[str, object]) -> dict[str, object]:
        issued.append((name, payload))
        return {"status": "ok", "affected_count": 1}

    with patch("polylogue.cli.archive_query._submit_mutation_operation", side_effect=_served):
        result = _run(tagged_archive, "--add-tag", "triage", "find", "id:claude-ai-export:ext-conv-1")

    assert result.exit_code == 0, result.output
    assert issued == [
        (
            "mutation.session.tag",
            {"session_ids": ["claude-ai-export:ext-conv-1"], "tags": ["triage"]},
        )
    ]
