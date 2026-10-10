"""Neutral capture makes one checked private copy of each uncached input."""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, cast

import pytest

from polylogue.core.compute import BoundedComputeAdapter, DaemonOperationCancelled
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import Provider, ValidationMode
from polylogue.sources.revision_backfill import RetainedPreparationRetryableError
from polylogue.storage import blob_store as blob_module
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.derived import raw as raw_module
from polylogue.storage.derived.raw import RawObservationDerivation
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.prepared_replay import run_on_convergence_owner
from tests.unit.storage.test_raw_convergence import _admit, _codex_conversation_bytes


@pytest.mark.parametrize("fault", ["none", "digest", "short", "replace", "read", "cancel"])
def test_uncached_neutral_capture_checks_one_copy_and_retires_faults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, record_property: Callable[[str, object], None]
) -> None:
    bootstrap_archive_root(tmp_path)
    payload = _codex_conversation_bytes("checked-copy", "x" * 1100000)
    raw_id = _admit(tmp_path, (), provider=Provider.CODEX, path="session.jsonl", payload=payload)
    store = BlobStore(tmp_path / "blob")
    blob_path = store.blob_path(hashlib.sha256(payload).hexdigest())
    if fault == "digest":
        blob_path.write_bytes(b"!" + payload[1:])
    elif fault == "short":
        blob_path.write_bytes(payload[:-1])
    read_bytes = opens = 0
    original_open = Path.open
    original_builtin = blob_module.builtins_open
    original_cancel = check_compute_cancelled

    class ObservedReader(io.BufferedReader):
        def read(self, size: int | None = -1) -> bytes:
            nonlocal read_bytes
            if fault == "read":
                raise OSError("neutral read fault")
            chunk = super().read(size)
            read_bytes += len(chunk)
            if chunk and fault == "replace":
                replacement = blob_path.with_suffix(".replacement")
                replacement.write_bytes(payload)
                os.replace(replacement, blob_path)
            return chunk

    def traced_open(
        path: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> BinaryIO:
        nonlocal opens
        handle = original_open(path, mode, buffering, encoding, errors, newline)
        if path == blob_path and mode == "rb":
            opens += 1
            return ObservedReader(cast(io.BufferedReader, handle))
        return cast(BinaryIO, handle)

    def traced_builtin(
        path: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> BinaryIO:
        nonlocal opens
        handle = original_builtin(path, mode, buffering, encoding, errors, newline)
        if path == blob_path and mode == "rb":
            opens += 1
            return ObservedReader(cast(io.BufferedReader, handle))
        return cast(BinaryIO, handle)

    def cancelled() -> None:
        if fault == "cancel" and read_bytes:
            raise DaemonOperationCancelled("neutral cancellation")
        original_cancel()

    monkeypatch.setattr(Path, "open", traced_open)
    monkeypatch.setattr(blob_module, "builtins_open", traced_builtin)
    monkeypatch.setattr(raw_module, "check_compute_cancelled", cancelled)

    def capture(compute: BoundedComputeAdapter) -> None:
        page = RawObservationDerivation(tmp_path, compute_adapter=compute).capture_neutral_raws((raw_id,))
        assert page is not None
        try:
            assert page.captures[raw_id].staged_blob.read_bytes() == payload
        finally:
            page.close()

    if fault == "none":
        run_on_convergence_owner(tmp_path, "test.neutral.checked_copy", capture)
        record_property("source_opens", opens)
        record_property("logical_source_bytes", read_bytes)
        assert (opens, read_bytes) == (1, len(payload))
    else:
        expected = (
            DaemonOperationCancelled
            if fault == "cancel"
            else OSError
            if fault == "read"
            else RetainedPreparationRetryableError
        )
        with pytest.raises(expected):
            run_on_convergence_owner(tmp_path, "test.neutral.checked_copy", capture)
    assert not list(store.staging_root.glob(".raw-prepared-*"))


def test_neutral_worker_charges_only_its_declared_sidecar_scope(tmp_path: Path) -> None:
    bootstrap_archive_root(tmp_path)
    selected = []
    expected: dict[str, int] = {}
    for ordinal, size in enumerate((2000, 3000)):
        session_id = f"10000000-0000-4000-8000-{ordinal:012d}"
        owner_path = tmp_path / "projects" / "neutral" / f"{session_id}.jsonl"
        sidecar_path = owner_path.parent / session_id / "tool-results" / "toolu_owned.txt"
        _admit(tmp_path, (), provider=Provider.UNKNOWN, path=str(sidecar_path), payload=b"x" * size)
        payload = (
            json.dumps(
                {
                    "type": "user",
                    "uuid": "message",
                    "sessionId": session_id,
                    "message": {"role": "user", "content": "neutral"},
                }
            )
            + "\n"
        ).encode()
        selected.append(_admit(tmp_path, (), provider=Provider.CLAUDE_CODE, path=str(owner_path), payload=payload))
        expected[str(owner_path)] = size
    codex = _admit(tmp_path, (), provider=Provider.CODEX, path="codex.jsonl", payload=_codex_conversation_bytes())
    selected.append(codex)

    def capture(compute: BoundedComputeAdapter) -> None:
        page = RawObservationDerivation(tmp_path, compute_adapter=compute).capture_neutral_raws(selected)
        assert page is not None
        try:
            assert dict(page.sidecar_input_bytes) == expected
            jobs = list(page.parser_jobs(ValidationMode.ADVISORY))
            assert len(jobs) == 3
            for key, charge, _operation in jobs:
                parser_identity = key[0]
                assert isinstance(parser_identity, tuple)
                raw_identity = parser_identity[0]
                assert isinstance(raw_identity, tuple) and raw_identity[0] == "raw-id"
                raw_id = str(raw_identity[1])
                provider, _hash, source_path, _kind, raw_size = page.captures[raw_id].descriptor
                snapshot = page.schema_snapshots[provider]
                schema_bytes = sum(len(payload) for _root, files in snapshot.roots for _name, payload in files)
                assert charge == raw_size + schema_bytes + expected.get(source_path, 0)
                if raw_id == codex:
                    assert charge == raw_size + schema_bytes
        finally:
            page.close()

    run_on_convergence_owner(tmp_path, "test.neutral.exact_scope_charge", capture)
