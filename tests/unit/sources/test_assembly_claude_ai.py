"""Tests for ClaudeAIAssemblySpec — bd polylogue-4zqh3 sole-copy attachment recovery.

Exercises the real ``sources/assembly.py`` protocol wiring end to end: an
``attachment-recovery.json`` sidecar next to a recovery source is discovered,
its referenced local file is streamed into the blob store, and a matching
attachment is enriched with ``precomputed_blob`` -- the same field
``ingest_batch/_core.py``'s ``preacquired_attachment_blobs`` loop records as
``acquired`` without re-hashing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.sources.assembly import SidecarData, get_assembly_spec
from polylogue.sources.assembly_claude_ai import ClaudeAIAssemblySpec
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession
from polylogue.storage.blob_store import BlobStore


def _session(attachments: list[ParsedAttachment], *, source_name: Provider = Provider.CLAUDE_AI) -> ParsedSession:
    return ParsedSession(
        source_name=source_name,
        provider_session_id="conv-1",
        title="t",
        messages=[ParsedMessage(provider_message_id="m1", role=Role.ASSISTANT, text="hi")],
        attachments=attachments,
    )


def test_get_assembly_spec_returns_claude_ai_spec() -> None:
    assert isinstance(get_assembly_spec(Provider.CLAUDE_AI), ClaudeAIAssemblySpec)


def test_discover_sidecars_finds_manifest_next_to_source_and_streams_bytes(tmp_path: Path) -> None:
    recovered = tmp_path / "recovered.md"
    recovered.write_bytes(b"recovered attachment payload")
    (tmp_path / "attachment-recovery.json").write_text(
        json.dumps({"attachments": [{"native_id": "att-1", "path": "recovered.md"}]}),
        encoding="utf-8",
    )
    (tmp_path / "claude-ai-browser-capture.json").write_text("{}", encoding="utf-8")
    store = BlobStore(tmp_path / "blobs")

    sidecar_data = ClaudeAIAssemblySpec().discover_sidecars(
        [tmp_path / "claude-ai-browser-capture.json"], blob_store=store
    )

    recovered_blobs = sidecar_data["claude_ai_recovered_blobs"]
    blob_hash, size = recovered_blobs["att-1"]
    assert blob_hash == hashlib.sha256(b"recovered attachment payload").hexdigest()
    assert size == len(b"recovered attachment payload")
    assert store.exists(blob_hash)


def test_discover_sidecars_skips_missing_manifest(tmp_path: Path) -> None:
    (tmp_path / "claude-ai-browser-capture.json").write_text("{}", encoding="utf-8")

    sidecar_data = ClaudeAIAssemblySpec().discover_sidecars([tmp_path / "claude-ai-browser-capture.json"])

    assert sidecar_data == {}


def test_discover_sidecars_warns_and_skips_missing_referenced_file(tmp_path: Path) -> None:
    (tmp_path / "attachment-recovery.json").write_text(
        json.dumps({"attachments": [{"native_id": "att-1", "path": "does-not-exist.md"}]}),
        encoding="utf-8",
    )
    (tmp_path / "claude-ai-browser-capture.json").write_text("{}", encoding="utf-8")

    sidecar_data = ClaudeAIAssemblySpec().discover_sidecars(
        [tmp_path / "claude-ai-browser-capture.json"], blob_store=BlobStore(tmp_path / "blobs")
    )

    assert sidecar_data == {}


def test_enrich_session_sets_precomputed_blob_for_matching_attachment() -> None:
    attachment = ParsedAttachment(provider_attachment_id="att-1", message_provider_id="m1", name="recovered.md")
    conv = _session([attachment])
    sidecar_data: SidecarData = {"claude_ai_recovered_blobs": {"att-1": ("deadbeef" * 8, 29)}}

    enriched = ClaudeAIAssemblySpec().enrich_session(conv, sidecar_data)

    assert enriched.attachments[0].precomputed_blob == ("deadbeef" * 8, 29)
    assert enriched.attachments[0].size_bytes == 29


def test_enrich_session_ignores_non_matching_attachment() -> None:
    attachment = ParsedAttachment(provider_attachment_id="att-other", message_provider_id="m1", name="unrelated.md")
    conv = _session([attachment])
    sidecar_data: SidecarData = {"claude_ai_recovered_blobs": {"att-1": ("deadbeef" * 8, 29)}}

    enriched = ClaudeAIAssemblySpec().enrich_session(conv, sidecar_data)

    assert enriched is conv


def test_enrich_session_does_not_overwrite_already_acquired_attachment() -> None:
    attachment = ParsedAttachment(
        provider_attachment_id="att-1",
        message_provider_id="m1",
        name="already-acquired.md",
        precomputed_blob=("cafebabe" * 8, 10),
    )
    conv = _session([attachment])
    sidecar_data: SidecarData = {"claude_ai_recovered_blobs": {"att-1": ("deadbeef" * 8, 29)}}

    enriched = ClaudeAIAssemblySpec().enrich_session(conv, sidecar_data)

    assert enriched is conv
    assert enriched.attachments[0].precomputed_blob == ("cafebabe" * 8, 10)


def test_enrich_session_ignores_non_claude_ai_sessions() -> None:
    attachment = ParsedAttachment(provider_attachment_id="att-1", message_provider_id="m1", name="recovered.md")
    conv = _session([attachment], source_name=Provider.CHATGPT)
    sidecar_data: SidecarData = {"claude_ai_recovered_blobs": {"att-1": ("deadbeef" * 8, 29)}}

    enriched = ClaudeAIAssemblySpec().enrich_session(conv, sidecar_data)

    assert enriched is conv
