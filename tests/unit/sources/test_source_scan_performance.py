"""Acquisition counters preserve complete member and ownership semantics."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.core.raw_coordinates import zip_member_source_index
from polylogue.sources import source_acquisition_components as components
from polylogue.sources.live.production_baseline import capture_production_source_baseline
from polylogue.sources.live.source_selection import deepest_source_for_path
from polylogue.sources.live.watcher import WatchSource
from polylogue.sources.source_acquisition import iter_source_acquisition_records
from polylogue.sources.source_layout import export_drop_layout
from polylogue.storage.blob_store import BlobStore


def _bundle(path: Path) -> tuple[bytes, bytes]:
    first = json.dumps({"id": "first", "mapping": {"node": {"message": {"author": {"role": "user"}}}}}).encode()
    second = json.dumps(
        {"uuid": "second", "name": "Neutral", "chat_messages": [{"sender": "human", "text": "hi"}]}
    ).encode()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("conversations.json", first)
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("conversations.json", second)
        archive.writestr("unknown.json", b"{}")
    return first, second


def test_production_zip_baseline_and_acquisition_inspect_each_ordinal_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "account"
    root.mkdir()
    bundle = root / "export.zip"
    first, second = _bundle(bundle)
    inspections: Counter[tuple[str, int]] = Counter()
    original = components._zip_entry_detected_provider

    def inspected(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> Provider | None:
        inspections[(info.filename, info.header_offset)] += 1
        return original(archive, info)

    monkeypatch.setattr(components, "_zip_entry_detected_provider", inspected)
    baseline = capture_production_source_baseline(
        (WatchSource("account", root, layout=export_drop_layout((".zip",))),), operation_id="neutral"
    )
    assert len(inspections) == 3
    assert set(inspections.values()) == {1}
    members = {row.source_index: row.revision for row in baseline.accepted}
    assert members[zip_member_source_index(entry_ordinal=0, split_index=0)] == hashlib.sha256(first).hexdigest()
    assert members[zip_member_source_index(entry_ordinal=1, split_index=0)] == hashlib.sha256(second).hexdigest()
    acquisitions = list(
        iter_source_acquisition_records(Source(name="account", path=bundle), blob_store=BlobStore(tmp_path / "blob"))
    )
    assert set(inspections.values()) == {2}
    raws = [item.data for item in acquisitions if item.data is not None]
    assert {raw.source_index: raw.blob_hash for raw in raws} == members
    by_ordinal = {
        raw.captured_zip_coordinate.entry_ordinal: raw.provider_hint for raw in raws if raw.captured_zip_coordinate
    }
    assert by_ordinal[0] is Provider.CHATGPT
    assert by_ordinal[1] is Provider.CLAUDE_AI


def test_zip_inspection_reuses_negative_result_and_refuses_wrong_ordinal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "neutral.zip"
    _bundle(bundle)
    calls = 0
    original = components._zip_entry_detected_provider

    def inspected(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> Provider | None:
        nonlocal calls
        calls += 1
        return original(archive, info)

    monkeypatch.setattr(components, "_zip_entry_detected_provider", inspected)
    with zipfile.ZipFile(io.BytesIO(bundle.read_bytes())) as archive:
        entries = archive.infolist()
        with components.zip_member_admission(
            archive,
            bundle,
            entries,
            Provider.UNKNOWN,
            container_blob_hash=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        ) as admission:
            assert calls == 3
            negative = admission.entry_provider_hint(entries[2], entry_ordinal=2)
            assert admission.entry_provider_hint(entries[2], entry_ordinal=2) is negative
            assert calls == 3
            with pytest.raises(ValueError, match="central-directory ordinal"):
                admission.entry_provider_hint(entries[0], entry_ordinal=1)
        assert admission.detected_members is not None
        with pytest.raises(ValueError):
            admission.entry_provider_hint(entries[2], entry_ordinal=2)


def test_lexical_ownership_does_not_resolve_unrelated_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "captures"
    root.mkdir()
    path = root / "one.json"
    path.write_bytes(b"{}")
    owner = WatchSource("inbox", root)
    unrelated = WatchSource("inbox", tmp_path / "other")
    original = Path.resolve
    calls: list[Path] = []

    def resolved(path: Path, strict: bool = False) -> Path:
        calls.append(path)
        return original(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolved)
    assert deepest_source_for_path(path, (unrelated, owner)) is owner
    assert calls == []


def test_physical_ownership_rechecks_retargeted_alias(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(first, target_is_directory=True)
    source = WatchSource("inbox", alias)
    path = first / "one.json"
    path.write_bytes(b"{}")
    assert deepest_source_for_path(path, (source,)) is source
    alias.unlink()
    alias.symlink_to(second, target_is_directory=True)
    assert deepest_source_for_path(path, (source,)) is None


@pytest.mark.parametrize("corrupt", [False, True])
def test_malformed_zip_member_inspection_finishes_crc_before_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt: bool
) -> None:
    bundle = tmp_path / "malformed.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("conversations.json", b'{"unfinished":' + b" " * 100000)
    encoded = bytearray(bundle.read_bytes())
    if corrupt:
        # Stored member's final byte changes without updating its declared CRC.
        with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
            info = archive.infolist()[0]
            encoded[info.header_offset + 30 + len(info.filename.encode()) + info.file_size - 1] ^= 1
    inspections = 0
    original = components._zip_entry_detected_provider

    def inspected(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> Provider | None:
        nonlocal inspections
        inspections += 1
        return original(archive, info)

    monkeypatch.setattr(components, "_zip_entry_detected_provider", inspected)
    with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
        entries = archive.infolist()
        context = components.zip_member_admission(
            archive, bundle, entries, Provider.UNKNOWN, container_blob_hash=hashlib.sha256(encoded).hexdigest()
        )
        if corrupt:
            with pytest.raises(zipfile.BadZipFile, match="CRC"):
                with context:
                    pytest.fail("corrupt inspection cannot become reusable")
        else:
            with context as admission:
                assert admission.entry_provider_hint(entries[0], entry_ordinal=0) is Provider.UNKNOWN
                assert admission.entry_provider_hint(entries[0], entry_ordinal=0) is Provider.UNKNOWN
        assert inspections == 1


def test_zip_explain_keeps_its_independent_diagnostic_container_hint(tmp_path: Path) -> None:
    from polylogue.sources.import_explain import explain_import_path

    bundle = tmp_path / "diagnostic.zip"
    raw_only = {
        "id": "neutral",
        "mapping": {
            "n1": {"message": {"author": {"role": "user"}, "content": {"content_type": "text", "parts": ["x" * 1000]}}}
        },
    }
    session = {"uuid": "neutral-claude", "name": "Neutral", "chat_messages": [{"sender": "human", "text": "hi"}]}
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("history.jsonl", json.dumps(raw_only))
        archive.writestr("conversations.json", json.dumps(session))
    payload = explain_import_path(bundle)
    assert payload.entries[0].detected_provider == Provider.CHATGPT.value
