"""Where a TAB press is allowed to do work.

Shell completion is the coldest, most latency-sensitive route the CLI has: it
runs in a fresh process on a keystroke. The law these tests pin is that the
archive-backed completers ask the resident daemon, use recent cached values,
or explain a cold cache. They never open the archive themselves or fall through
to the local reader, which costs seconds.

Deliberately not a timing threshold. A latency assertion is flaky on a loaded
machine and proves less than the structural fact: no database was opened.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import click
import pytest

from polylogue.cli.operation_kernel import OperationRequest, OperationUnavailableError, dispatch
from polylogue.cli.shell_completion_classes import MESSAGE_COMPLETION_TYPE, MessageAwareZshComplete
from polylogue.cli.shell_completion_values import (
    DAEMON_REQUIRED_COMPLETION_MESSAGE,
    complete_origin_values,
    complete_repo_values,
    complete_session_ids,
    complete_tag_values,
    complete_tool_values,
)
from polylogue.config import get_config
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.frozen_clock import FrozenClock

pytestmark = pytest.mark.contract

ARCHIVE_BACKED_COMPLETERS = {
    "session_id": complete_session_ids,
    "tag": complete_tag_values,
    "repo": complete_repo_values,
    "tool": complete_tool_values,
}

# Runs the archive-backed completers in a fresh interpreter under an audit hook
# that records every sqlite3 connection, then reports what was opened and what
# the completers returned. An audit hook sees the real ``sqlite3.connect``, so
# it cannot be satisfied by a stub the test itself installed.
_PROBE = """
import json, sys

opened = []
def _hook(event, args):
    if event == "sqlite3.connect":
        opened.append(str(args[0]))
sys.addaudithook(_hook)

import click
from polylogue.cli.shell_completion_values import (
    complete_repo_values, complete_session_ids, complete_tag_values, complete_tool_values,
)

ctx = click.Context(click.Command("polylogue"))
param = click.Option(["--probe"])
returned = {}
for name, fn in (
    ("session_id", complete_session_ids),
    ("tag", complete_tag_values),
    ("repo", complete_repo_values),
    ("tool", complete_tool_values),
):
    returned[name] = [(item.type, item.value) for item in fn(ctx, param, "")]

print("RESULT " + json.dumps({"opened": opened, "returned": returned}))
"""


def _run_probe(archive_root: Path, runtime_dir: Path) -> dict[str, object]:
    process = subprocess.run(
        [sys.executable, "-c", _PROBE],
        check=False,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(runtime_dir),
            "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
            # An empty runtime dir holds no daemon socket, so this is the
            # daemon-off case without needing to stop anything.
            "XDG_RUNTIME_DIR": str(runtime_dir),
            "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
        },
    )
    assert process.returncode == 0, process.stderr
    payload = [line for line in process.stdout.splitlines() if line.startswith("RESULT ")]
    assert payload, process.stdout + process.stderr
    decoded: dict[str, object] = json.loads(payload[-1][len("RESULT ") :])
    return decoded


def test_daemon_off_completion_opens_no_database(tmp_path: Path) -> None:
    """With an archive present and no daemon, a TAB press opens nothing.

    Anti-vacuity is the archive itself: ``bootstrap_archive_root`` leaves a real
    ``index.db`` here, so a completer that reads locally *can* open one and the
    recorded list is the proof that it did not.

    Mutation: drop ``daemon_only=True`` from ``completion_values`` and the
    dispatch falls through to the local reader, which opens ``index.db`` --
    ``opened`` is then non-empty and this fails.
    """

    bootstrap_archive_root(tmp_path / "archive")
    runtime = tmp_path / "runtime"
    runtime.mkdir()

    probe = _run_probe(tmp_path / "archive", runtime)

    assert probe["opened"] == [], f"completion opened databases with no daemon: {probe['opened']}"


def test_cold_cache_completion_says_how_to_populate_values(tmp_path: Path) -> None:
    """A cold cache explains how to populate suggestions, not a false no-match.

    Mutation: return a bare ``[]`` on ``OperationUnavailableError`` and the
    shell shows "no matches" -- which is the difference between an empty cache
    and a query with no matching values.
    """

    bootstrap_archive_root(tmp_path / "archive")
    runtime = tmp_path / "runtime"
    runtime.mkdir()

    returned = _run_probe(tmp_path / "archive", runtime)["returned"]
    assert isinstance(returned, dict)

    for source in ARCHIVE_BACKED_COMPLETERS:
        rows = returned[source]
        assert rows == [[MESSAGE_COMPLETION_TYPE, DAEMON_REQUIRED_COMPLETION_MESSAGE]], (
            f"{source} did not render the cold-cache guidance: {rows}"
        )
    assert "polylogued" in DAEMON_REQUIRED_COMPLETION_MESSAGE, "the refusal must name how to fix it"


def test_daemon_off_completion_uses_recent_values_from_same_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen_clock: FrozenClock
) -> None:
    """A prior daemon answer remains useful offline until its bounded expiry."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive-a"))
    from polylogue.cli import operation_kernel

    monkeypatch.setattr(
        operation_kernel,
        "dispatch",
        lambda *_args, **_kwargs: SimpleNamespace(
            value={"value_completions": {"values": [{"value": "release-tag"}, {"value": "roadmap"}]}}
        ),
    )
    from polylogue.cli.shell_completion_values import completion_values

    assert [item.value for item in completion_values("tag", "", limit=5)] == ["release-tag", "roadmap"]
    monkeypatch.setattr(
        operation_kernel,
        "dispatch",
        lambda *_args, **_kwargs: SimpleNamespace(
            value={
                "value_completions": {
                    "values": [{"value": "claude-code-session:ext-123", "help": "claude-code · Roadmap planning"}]
                }
            }
        ),
    )
    assert [item.value for item in completion_values("session_id", "", limit=5)] == ["claude-code-session:ext-123"]

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise OperationUnavailableError("daemon unavailable")

    monkeypatch.setattr(operation_kernel, "dispatch", unavailable)
    assert [item.value for item in completion_values("tag", "rel", limit=5)] == ["release-tag"]
    assert [item.value for item in completion_values("session_id", "ext-123", limit=5)] == [
        "claude-code-session:ext-123"
    ]
    assert [item.value for item in completion_values("session_id", "Roadmap", limit=5)] == [
        "claude-code-session:ext-123"
    ]
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive-b"))
    assert [item.value for item in completion_values("tag", "rel", limit=5)] == [DAEMON_REQUIRED_COMPLETION_MESSAGE]
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive-a"))
    frozen_clock.advance(24 * 60 * 60 + 1)
    assert [item.value for item in completion_values("tag", "rel", limit=5)] == [DAEMON_REQUIRED_COMPLETION_MESSAGE]


def test_a_declared_vocabulary_still_completes_without_a_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """``origin`` is a declaration, so it answers on a fresh install.

    Mutation: route ``complete_origin_values`` through the ``completion``
    operation and the one always-answerable completion returns the daemon
    refusal instead of the origins.
    """

    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", "/nonexistent/polylogue-archive")
    ctx = click.Context(click.Command("polylogue"))
    items = complete_origin_values(ctx, click.Option(["--origin"]), "")

    assert items, "origin completion must not depend on an archive or a daemon"
    assert all(item.type == "plain" for item in items)


def test_a_message_item_is_displayed_and_never_inserted() -> None:
    """The zsh script renders a ``message`` item with ``_message``.

    Mutation: leave the message branch out of the template and zsh falls
    through, offering nothing at all -- the silent-empty behaviour again. The
    template is derived from Click's, so this also fails loudly if upstream
    changes the loop it splices into.
    """

    template = MessageAwareZshComplete.source_template
    assert '"$type" == "message"' in template
    assert "_message -r" in template
    # A message must never reach the branches that add a completion candidate.
    message_branch = template.split('"$type" == "message"', 1)[1].split("elif", 1)[0]
    assert "compadd" not in message_branch and "_describe" not in message_branch


def test_daemon_only_dispatch_refuses_instead_of_reading_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Archive completion refuses without a daemon and never reads locally."""

    bootstrap_archive_root(tmp_path)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path))
    config = get_config()
    request = OperationRequest("completion", {"source": "tag", "incomplete": "", "limit": 3})

    with pytest.raises(OperationUnavailableError):
        dispatch(config, request, daemon_only=True, archive_root=tmp_path)

    # This operation now has one daemon-owned route even without the explicit
    # flag; a local archive fallback would put seconds back on the TAB path.
    with pytest.raises(OperationUnavailableError):
        dispatch(config, request, archive_root=tmp_path)
