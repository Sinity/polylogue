from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocumentList
from polylogue.sources.acquisition_boundary import capture_bound_path
from polylogue.sources.sqlite_export import source_byte_page_sequence
from polylogue.storage.blob_store import BlobStore


def _claude_session(session_id: str) -> bytes:
    records: JSONDocumentList = [
        {
            "type": "user",
            "uuid": f"{session_id}-u1",
            "sessionId": session_id,
            "timestamp": "2025-06-13T17:40:00.000Z",
            "cwd": "/home/user/project",
            "message": {"role": "user", "content": f"Question for {session_id}."},
        },
        {
            "type": "assistant",
            "uuid": f"{session_id}-a1",
            "parentUuid": f"{session_id}-u1",
            "sessionId": session_id,
            "timestamp": "2025-06-13T17:40:05.000Z",
            "cwd": "/home/user/project",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Answer."}]},
        },
    ]
    return ("\n".join(json.dumps(record) for record in records) + "\n").encode("utf-8")


def _inputs(tmp_path: Path, count: int) -> list[Path]:
    project = tmp_path / ".claude" / "projects" / "proj"
    project.mkdir(parents=True)
    paths = []
    for index in range(count):
        path = project / f"session-{index}.jsonl"
        path.write_bytes(_claude_session(f"session-{index}"))
        paths.append(path)
    return paths


def test_a_page_of_captures_shares_one_reader_process(tmp_path: Path) -> None:
    """Sequential captures on one page reuse its isolated reader.

    Anti-vacuity: a capture that ignores the lent page opens a reader of its
    own per input; the sequence's page then never starts a process, and every
    input pays a fresh interpreter start.
    """
    store = BlobStore(tmp_path / "blobs")
    paths = _inputs(tmp_path, 3)
    with source_byte_page_sequence() as pages:
        readers = []
        for path in paths:
            page = pages.page()
            capture = capture_bound_path(store, path, Provider.CLAUDE_CODE, byte_page=page)
            with store.open(capture.blob_hash) as retained:
                assert retained.read() == path.read_bytes()
            readers.append(page._process)
        assert readers[0] is not None
        assert all(reader is readers[0] for reader in readers)
        reader = readers[0]
    assert reader.returncode == 0


def test_a_failed_capture_retires_only_its_own_reader(tmp_path: Path) -> None:
    """A rejected page is replaced, so one bad input never strands the next."""
    store = BlobStore(tmp_path / "blobs")
    first, second = _inputs(tmp_path, 2)
    with source_byte_page_sequence() as pages:
        healthy = pages.page()
        capture_bound_path(store, first, Provider.CLAUDE_CODE, byte_page=healthy)
        failed = healthy._process
        # Reject the live page exactly as a failed exchange does: its reader
        # is killed and reaped, and the page refuses further requests.
        healthy.reject(OSError("simulated reader failure"))
        assert failed is not None and failed.returncode is not None
        with pytest.raises(RuntimeError):
            capture_bound_path(store, second, Provider.CLAUDE_CODE, byte_page=healthy)
        replacement = pages.page()
        assert replacement is not healthy
        capture = capture_bound_path(store, second, Provider.CLAUDE_CODE, byte_page=replacement)
        assert replacement._process is not None and replacement._process is not failed
        with store.open(capture.blob_hash) as retained:
            assert retained.read() == second.read_bytes()


def test_a_page_proves_each_binding_without_a_fresh_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The lent page binds its inputs; no capture starts a process of its own.

    Anti-vacuity: binding through ``_exchange_source_worker`` again starts one
    fresh interpreter per input, which the patched exchange refuses here.
    """
    from polylogue.sources import sqlite_export

    def refuse_fresh_reader(request: dict[str, object], handle: object = None) -> dict[str, object]:
        raise AssertionError(f"fresh reader for {request['operation']}")

    monkeypatch.setattr(sqlite_export, "_exchange_source_worker", refuse_fresh_reader)
    store = BlobStore(tmp_path / "blobs")
    paths = _inputs(tmp_path, 3)
    with source_byte_page_sequence() as pages:
        for path in paths:
            capture = capture_bound_path(store, path, Provider.CLAUDE_CODE, byte_page=pages.page())
            assert capture.canonical_source_path == str(path.parent.resolve() / path.name)
            with store.open(capture.blob_hash) as retained:
                assert retained.read() == path.read_bytes()


@pytest.mark.parametrize("operation", ["bytes", "sqlite"])
def test_source_workers_inherit_parent_runtime_import_paths(tmp_path: Path, operation: str) -> None:
    """A packaged parent's injected import paths survive its fresh readers.

    The parent uses the bare interpreter with no Python environment overrides
    and a non-checkout cwd, then injects paths only into its own sys.path, as
    an installed console entry point does. Removing the worker bootstrap makes
    both byte capture and logical SQLite export fail before returning bytes.
    """
    source = _inputs(tmp_path, 1)[0]
    if operation == "sqlite":
        source = tmp_path / "neutral.db"
        with sqlite3.connect(source) as conn:
            conn.execute("CREATE TABLE item (value TEXT)")
            conn.execute("INSERT INTO item VALUES ('neutral')")
    working = tmp_path / "outside-checkout"
    working.mkdir()
    bootstrap = """
import json, sys
from pathlib import Path
sys.path[:] = json.loads(sys.argv[1])
from polylogue.sources.acquisition_boundary import capture_bound_path
from polylogue.sources.sqlite_export import logical_export_bytes
from polylogue.core.enums import Provider
from polylogue.storage.blob_store import BlobStore
source = Path(sys.argv[2])
if sys.argv[3] == 'bytes':
    store = BlobStore(Path(sys.argv[4]))
    capture = capture_bound_path(store, source, Provider.CLAUDE_CODE)
    with store.open(capture.blob_hash) as retained:
        assert retained.read() == source.read_bytes()
    print(json.dumps({'size': capture.blob_size}))
else:
    payload = logical_export_bytes(source, tables=('item',))
    rows = [json.loads(line) for line in payload.splitlines()]
    assert rows[-1][-1] == ['t', 'neutral']
    print(json.dumps({'tables': rows[0]['tables']}))
"""
    paths = [str(Path(path or os.getcwd()).resolve()) for path in sys.path]
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONUSERBASE"}
    }
    result = subprocess.run(
        [
            os.path.realpath(sys.executable),
            "-c",
            bootstrap,
            json.dumps(paths),
            str(source),
            operation,
            str(tmp_path / "blobs"),
        ],
        cwd=working,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == (
        {"size": source.stat().st_size} if operation == "bytes" else {"tables": ["item"]}
    )
