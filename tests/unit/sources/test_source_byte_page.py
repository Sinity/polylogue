"""A bounded acquisition page owns one byte child through exact request settlement."""

from __future__ import annotations

import array
import errno
import hashlib
import io
import os
import socket
import stat
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.compute_cancel import compute_cancel
from polylogue.sources import source_staging, sqlite_export

pytestmark = pytest.mark.uses_real_clock("byte pages prove actual child process settlement")


class _ObservedChildren(list[subprocess.Popen[bytes]]):
    def __init__(self) -> None:
        super().__init__()
        self.binding_children: list[subprocess.Popen[bytes]] = []


def _children(monkeypatch: pytest.MonkeyPatch) -> _ObservedChildren:
    original = subprocess.Popen
    exchange = sqlite_export._exchange_source_worker
    children = _ObservedChildren()
    binding = False

    def observe_exchange(request: dict[str, Any], handle: Any = None) -> dict[str, Any]:
        nonlocal binding
        previous = binding
        binding = request["operation"] == "binding"
        try:
            return exchange(request, handle)
        finally:
            binding = previous

    def launch(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        child = original(*args, **kwargs)
        (children.binding_children if binding else children).append(child)
        return child

    monkeypatch.setattr(sqlite_export, "_exchange_source_worker", observe_exchange)
    monkeypatch.setattr(subprocess, "Popen", launch)
    return children


def _settled(children: _ObservedChildren) -> None:
    assert len(children) == 1
    for child in [*children.binding_children, *children]:
        assert child.poll() is not None
        assert child.stdin is not None and child.stdin.closed
        assert child.stdout is not None and child.stdout.closed


@pytest.mark.parametrize("count", [0, 1, 16, pytest.param(256, marks=pytest.mark.timeout(0))])
def test_byte_page_launches_once_and_reproves_each_exact_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, record_property: Any
) -> None:
    children = _children(monkeypatch)
    started = time.monotonic()
    with sqlite_export.source_byte_page() as reader:
        for ordinal in range(count):
            parent = tmp_path / f"parent-{ordinal}"
            parent.mkdir()
            source = parent / "input.zip"
            payload = b"PK\x05\x06" + bytes(18) + str(ordinal).encode()
            source.write_bytes(payload)
            with source_staging.bind_source_input(source) as binding:
                output = io.BytesIO()
                result = source_staging.write_bound_input(binding, output, reader=reader)
                assert output.getvalue() == payload
                assert result["content_revision"] == hashlib.sha256(payload).hexdigest()
                assert result["size_bytes"] == len(payload)
                assert result["file_observation"][:2] == list(binding.main_identity)
            assert len(children) == 1 and children[0].poll() is None
            # After request C, the child retains only transport capabilities.
            proc = Path(f"/proc/{children[0].pid}/fd")
            if proc.exists():
                assert not any(stat.S_ISDIR(path.stat().st_mode) for path in proc.iterdir())
    record_property("byte_page_inputs", count)
    record_property("byte_page_elapsed_seconds", time.monotonic() - started)
    assert len(children.binding_children) == count
    if count:
        _settled(children)
        assert children[0].returncode == 0
    else:
        assert children == []


@pytest.mark.parametrize("malformation", ["marker", "missing", "extra", "foreign"])
def test_byte_page_refuses_malformed_or_rebound_ancillary_capabilities_and_reaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformation: str
) -> None:
    children = _children(monkeypatch)
    source = tmp_path / "input.zip"
    source.write_bytes(b"PK\x05\x06" + bytes(18))
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    original = socket.socket.sendmsg

    def transfer(channel: socket.socket, buffers: Any, ancillary: Any = (), flags: int = 0) -> int:
        rights = array.array("i")
        rights.frombytes(bytes(ancillary[0][2]))
        extra: int | None = None
        try:
            if malformation == "marker":
                buffers = [b"X"]
            elif malformation == "missing":
                ancillary = ()
            elif malformation == "extra":
                extra = os.dup(rights[0])
                rights.append(extra)
                ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)]
            else:
                extra = os.open(foreign, os.O_RDONLY | os.O_DIRECTORY)
                rights[0] = extra
                ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)]
            return original(channel, buffers, ancillary, flags)
        finally:
            if extra is not None:
                os.close(extra)

    monkeypatch.setattr(socket.socket, "sendmsg", transfer)
    with pytest.raises(OSError) as stopped:
        with source_staging.bind_source_input(source) as binding, sqlite_export.source_byte_page() as reader:
            source_staging.write_bound_input(binding, io.BytesIO(), reader=reader)
    assert stopped.value.errno == (errno.ESTALE if malformation == "foreign" else errno.EPROTO)
    _settled(children)


def test_second_request_identity_failure_retires_whole_page_without_reusing_first_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    children = _children(monkeypatch)
    first, second, replacement = (tmp_path / name for name in ("first.zip", "second.zip", "replacement.zip"))
    for source in (first, second, replacement):
        source.write_bytes(b"PK\x05\x06" + bytes(18))
    original = socket.socket.sendmsg
    calls = 0

    def transfer(channel: socket.socket, buffers: Any, ancillary: Any = (), flags: int = 0) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            os.replace(replacement, second)
        return original(channel, buffers, ancillary, flags)

    monkeypatch.setattr(socket.socket, "sendmsg", transfer)
    one, two = io.BytesIO(), io.BytesIO()
    with pytest.raises(OSError) as stopped:
        with source_staging.bind_source_input(first) as first_binding:
            with source_staging.bind_source_input(second) as second_binding, sqlite_export.source_byte_page() as reader:
                source_staging.write_bound_input(first_binding, one, reader=reader)
                source_staging.write_bound_input(second_binding, two, reader=reader)
    assert stopped.value.errno == errno.ESTALE
    assert one.getvalue() == first.read_bytes() and two.getvalue() == b""
    _settled(children)


def test_cancellation_between_page_requests_reaps_original_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    children = _children(monkeypatch)
    source = tmp_path / "input.zip"
    source.write_bytes(b"PK\x05\x06" + bytes(18))
    cancelled = threading.Event()
    token = compute_cancel.set(cancelled)
    try:
        with pytest.raises(DaemonOperationCancelled):
            with source_staging.bind_source_input(source) as binding, sqlite_export.source_byte_page() as reader:
                source_staging.write_bound_input(binding, io.BytesIO(), reader=reader)
                cancelled.set()
                source_staging.write_bound_input(binding, io.BytesIO(), reader=reader)
        _settled(children)
    finally:
        compute_cancel.reset(token)


@pytest.mark.parametrize("sink_failure", ["short", "cancel"])
def test_page_ack_requires_complete_sink_delivery_and_preserves_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sink_failure: str
) -> None:
    children = _children(monkeypatch)
    source = tmp_path / "input.zip"
    source.write_bytes(b"PK\x05\x06" + bytes(18))
    cancellation = DaemonOperationCancelled("synthetic original page callback stop")

    class Sink:
        def write(self, payload: bytes) -> int:
            if sink_failure == "cancel":
                raise cancellation
            return len(payload) - 1

    with pytest.raises((OSError, DaemonOperationCancelled)) as stopped:
        with source_staging.bind_source_input(source) as binding, sqlite_export.source_byte_page() as reader:
            source_staging.write_bound_input(binding, Sink(), reader=reader)
    if sink_failure == "cancel":
        assert stopped.value is cancellation
    else:
        assert isinstance(stopped.value, OSError) and stopped.value.errno == errno.EIO
    _settled(children)


def test_page_request_failure_cannot_be_caught_and_published_as_a_successful_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    children = _children(monkeypatch)
    source, replacement = tmp_path / "input.zip", tmp_path / "replacement.zip"
    source.write_bytes(b"PK\x05\x06" + bytes(18))
    replacement.write_bytes(source.read_bytes())
    original_error: OSError | None = None
    with pytest.raises(OSError) as stopped:
        with source_staging.bind_source_input(source) as binding, sqlite_export.source_byte_page() as reader:
            source_staging.write_bound_input(binding, io.BytesIO(), reader=reader)
            os.replace(replacement, source)
            try:
                source_staging.write_bound_input(binding, io.BytesIO(), reader=reader)
            except OSError as error:
                original_error = error
    assert original_error is not None and stopped.value is original_error and stopped.value.errno == errno.ESTALE
    _settled(children)


@pytest.mark.parametrize("frame", [b"X" + bytes(8), b"C" + bytes(8)])
def test_byte_page_malformed_or_unproved_request_completion_reaps_the_same_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frame: bytes
) -> None:
    children = _children(monkeypatch)
    source = tmp_path / "input.zip"
    source.write_bytes(b"PK\x05\x06" + bytes(18))
    function = (
        "def malformed(request, sink):\n"
        f"    sys.stdout.buffer.write({frame!r})\n"
        "    sys.stdout.buffer.flush()\n"
        "    sys.stdin.buffer.read()\n"
    )
    command = (
        "import sys; from polylogue.sources import sqlite_export, source_staging; "
        f"exec({function!r}); "
        "source_staging._read_bound_input_in_worker=malformed; sqlite_export._source_worker_main()"
    )
    monkeypatch.setattr(sqlite_export, "_WORKER_COMMAND", command)
    with pytest.raises(OSError) as stopped:
        with source_staging.bind_source_input(source) as binding, sqlite_export.source_byte_page() as reader:
            source_staging.write_bound_input(binding, io.BytesIO(), reader=reader)
    assert stopped.value.errno == errno.EPROTO
    _settled(children)


def test_page_callback_failure_reaps_before_the_original_binding_releases_anchors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    children = _children(monkeypatch)
    source = tmp_path / "input.zip"
    source.write_bytes(b"PK\x05\x06" + bytes(18))
    cancellation = DaemonOperationCancelled("synthetic original callback")

    class Sink:
        def write(self, payload: bytes) -> int:
            raise cancellation

    with pytest.raises(DaemonOperationCancelled) as stopped:
        with source_staging.bind_source_input(source) as binding, sqlite_export.source_byte_page() as reader:
            try:
                source_staging.write_bound_input(binding, Sink(), reader=reader)
            except DaemonOperationCancelled as error:
                assert error is cancellation
                _settled(children)
                assert stat.S_ISDIR(os.fstat(binding.parent_anchor).st_mode)
                assert stat.S_ISDIR(os.fstat(binding.metadata_anchor).st_mode)
                raise
    assert stopped.value is cancellation
