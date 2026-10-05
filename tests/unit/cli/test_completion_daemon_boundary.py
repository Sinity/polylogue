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
from typing import Any

import click
import pytest
from click.shell_completion import CompletionItem

from polylogue.cli.operation_kernel import OperationRequest, OperationUnavailableError, dispatch
from polylogue.cli.shell_completion_classes import MESSAGE_COMPLETION_TYPE, MessageAwareZshComplete
from polylogue.cli.shell_completion_values import (
    DAEMON_REQUIRED_COMPLETION_MESSAGE,
    complete_cwd_prefix_values,
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
    "cwd_prefix": complete_cwd_prefix_values,
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
    complete_cwd_prefix_values, complete_repo_values, complete_session_ids, complete_tag_values, complete_tool_values,
)

ctx = click.Context(click.Command("polylogue"))
param = click.Option(["--probe"])
returned = {}
for name, fn in (
    ("session_id", complete_session_ids),
    ("tag", complete_tag_values),
    ("repo", complete_repo_values),
    ("cwd_prefix", complete_cwd_prefix_values),
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


@pytest.mark.parametrize(
    "source,rows,prefix",
    [
        ("tag", [{"value": "release-tag"}, {"value": "roadmap"}], "rel"),
        ("cwd_prefix", [{"value": "/neutral/release"}, {"value": "/neutral/roadmap"}], "/neutral/rel"),
    ],
)
def test_daemon_off_completion_uses_recent_values_from_same_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_clock: FrozenClock,
    source: str,
    rows: list[dict[str, str]],
    prefix: str,
) -> None:
    """A prior daemon answer remains useful offline until its bounded expiry."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive-a"))
    from polylogue.cli import operation_kernel

    monkeypatch.setattr(
        operation_kernel,
        "dispatch",
        lambda *_args, **_kwargs: SimpleNamespace(value={"value_completions": {"values": rows}}),
    )
    from polylogue.cli.shell_completion_values import completion_values

    assert [item.value for item in completion_values(source, "", limit=5)] == [row["value"] for row in rows]
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
    assert [item.value for item in completion_values(source, prefix, limit=5)] == [rows[0]["value"]]
    assert [item.value for item in completion_values("session_id", "ext-123", limit=5)] == [
        "claude-code-session:ext-123"
    ]
    assert [item.value for item in completion_values("session_id", "Roadmap", limit=5)] == [
        "claude-code-session:ext-123"
    ]
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive-b"))
    assert [item.value for item in completion_values(source, prefix, limit=5)] == [DAEMON_REQUIRED_COMPLETION_MESSAGE]
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive-a"))
    assert [item.value for item in completion_values(source, prefix, limit=5)] == [rows[0]["value"]]
    frozen_clock.advance(24 * 60 * 60 + 1)
    assert [item.value for item in completion_values(source, prefix, limit=5)] == [DAEMON_REQUIRED_COMPLETION_MESSAGE]


@pytest.mark.parametrize("source", ["tag", "cwd_prefix"])
def test_authoritative_empty_refresh_removes_old_cached_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """An empty daemon answer invalidates stale prefix matches immediately.

    Anti-vacuity: first seed a real cache entry, then return a successful empty
    response and take the daemon offline; the removed value must stay absent.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive-a"))
    from polylogue.cli import operation_kernel
    from polylogue.cli.shell_completion_values import completion_values

    monkeypatch.setattr(
        operation_kernel,
        "dispatch",
        lambda *_a, **_k: SimpleNamespace(value={"value_completions": {"values": [{"value": "release-tag"}]}}),
    )
    completion_values(source, "rel", limit=5)
    monkeypatch.setattr(
        operation_kernel, "dispatch", lambda *_a, **_k: SimpleNamespace(value={"value_completions": {"values": []}})
    )
    completion_values(source, "rel", limit=5)

    def unavailable(*_a: object, **_k: object) -> None:
        raise OperationUnavailableError("daemon unavailable")

    monkeypatch.setattr(operation_kernel, "dispatch", unavailable)
    assert [item.value for item in completion_values(source, "rel", limit=5)] == [DAEMON_REQUIRED_COMPLETION_MESSAGE]


def test_completion_cache_global_serialization_stays_under_byte_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The published multi-archive cache remains readable and within its cap.

    Anti-vacuity: populate enough maximum-sized help strings across archives
    to exceed the cap without global eviction, then parse the published file.
    """
    from polylogue.cli import shell_completion_values as cache

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    rows = [{"value": f"value-{index:04d}", "help": "é" * 512} for index in range(256)]
    for archive in range(6):
        cache._remember_completion_values(
            "tag", {"value_completions": {"values": rows}}, archive_root=f"root-{archive}"
        )
    path = cache._completion_cache_path()
    assert path.stat().st_size <= cache._COMPLETION_CACHE_MAX_BYTES
    assert cache._read_completion_cache("tag", "value-", limit=10, archive_root="root-5")


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


def test_completion_cache_contention_does_not_wait_for_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl

    from polylogue.cli import shell_completion_values as cache

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    path = cache._completion_cache_path()
    path.parent.mkdir(parents=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        cache._remember_completion_values(
            "cwd_prefix", {"value_completions": {"values": [{"value": "/neutral"}]}}, archive_root="neutral-root"
        )
        assert not path.exists()
    cache._remember_completion_values(
        "cwd_prefix", {"value_completions": {"values": [{"value": "/neutral"}]}}, archive_root="neutral-root"
    )
    assert [
        item.value
        for item in cache._read_completion_cache("cwd_prefix", "\\neutral", limit=5, archive_root="neutral-root")
    ] == ["/neutral"]


def test_cwd_callback_preserves_literal_prefix_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.cli import shell_completion_values as cache

    requests: list[tuple[str, str, int]] = []

    def values(source: str, incomplete: str, *, limit: int) -> list[CompletionItem]:
        requests.append((source, incomplete, limit))
        return []

    monkeypatch.setattr(cache, "completion_values", values)
    cache.complete_cwd_prefix_values(
        click.Context(click.Command("polylogue")), click.Option(["--cwd-prefix"]), "/neutral/my "
    )
    assert requests == [("cwd_prefix", "/neutral/my ", cache._MAX_VALUE_COMPLETIONS)]


def test_oversized_advisory_cache_is_bounded_on_read_and_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.cli import shell_completion_values as cache

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    path = cache._completion_cache_path()
    path.parent.mkdir(parents=True)
    with path.open("wb") as stream:
        stream.write(b'{"version": 3, "padding": "')
        stream.seek(cache._COMPLETION_CACHE_MAX_BYTES * 2)
        stream.write(b'"}')
    original_open = Path.open
    sizes: list[int] = []

    class BoundedRead:
        def __enter__(self) -> BoundedRead:
            self.stream = original_open(path, "rb")
            return self

        def read(self, size: int = -1) -> bytes:
            assert size == cache._COMPLETION_CACHE_MAX_BYTES + 1
            sizes.append(size)
            return self.stream.read(size)

        def __exit__(self, *_args: object) -> None:
            self.stream.close()

    def open_path(selected: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        return BoundedRead() if selected == path and mode == "rb" else original_open(selected, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)
    assert cache._read_completion_cache("cwd_prefix", "", limit=5, archive_root="neutral-root") == []
    cache._remember_completion_values(
        "cwd_prefix", {"value_completions": {"values": [{"value": "/neutral"}]}}, archive_root="neutral-root"
    )
    assert len(sizes) == 2
    assert path.stat().st_size <= cache._COMPLETION_CACHE_MAX_BYTES
