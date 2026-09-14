"""Sidecar and manifest lookups stay inside the source tree that declared them.

Both routes derive a filesystem location from a value an untrusted export
controls: a Gemini CLI snapshot's ``sessionId`` names a ``tool-outputs/``
subdirectory, and a Claude AI ``attachment-recovery.json`` entry names a file
beside the manifest. Neither value was contained, so an import could pull bytes
from anywhere the daemon can read into the archive as session evidence.

Anti-vacuity: revert either containment check and the escaping case below
resolves to (and, for the manifest, ingests) the outside-the-tree target
instead of being refused -- ``escape_target`` appears in the resolved path, or
its bytes appear in the returned blob map.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from polylogue.sources.assembly_claude_ai import ClaudeAIAssemblySpec
from polylogue.sources.live.gemini_tool_output_sidecars import resolve_tool_outputs_dir
from polylogue.storage.blob_store import BlobStore


def test_gemini_tool_outputs_dir_resolves_an_ordinary_session_id(tmp_path: Path) -> None:
    snapshot = tmp_path / "project" / "chats" / "session-a.json"

    resolved = resolve_tool_outputs_dir(snapshot, "abc-123")

    assert resolved == tmp_path / "project" / "tool-outputs" / "session-abc-123"


def test_gemini_tool_outputs_dir_refuses_a_traversing_session_id(tmp_path: Path) -> None:
    snapshot = tmp_path / "project" / "chats" / "session-a.json"

    assert resolve_tool_outputs_dir(snapshot, "hop/../../../../escape_target") is None
    assert resolve_tool_outputs_dir(snapshot, "a/b") is None
    assert resolve_tool_outputs_dir(snapshot, "/escape_target") is None


def test_claude_ai_recovery_manifest_refuses_a_path_outside_its_directory(tmp_path: Path) -> None:
    outside = tmp_path / "escape_target.md"
    outside.write_bytes(b"private bytes the manifest must not reach")
    import_root = tmp_path / "import"
    import_root.mkdir()
    (import_root / "attachment-recovery.json").write_text(
        json.dumps(
            {
                "attachments": [
                    {"native_id": "att-escape-relative", "path": "../escape_target.md"},
                    {"native_id": "att-escape-absolute", "path": str(outside)},
                ]
            }
        ),
        encoding="utf-8",
    )
    (import_root / "claude-ai-browser-capture.json").write_text("{}", encoding="utf-8")
    store = BlobStore(tmp_path / "blobs")

    sidecar_data = ClaudeAIAssemblySpec().discover_sidecars(
        [import_root / "claude-ai-browser-capture.json"], blob_store=store
    )

    assert sidecar_data == {}
    assert not store.exists(hashlib.sha256(outside.read_bytes()).hexdigest())


def test_claude_ai_recovery_manifest_still_accepts_a_contained_path(tmp_path: Path) -> None:
    import_root = tmp_path / "import"
    (import_root / "nested").mkdir(parents=True)
    payload = b"recovered attachment payload"
    (import_root / "nested" / "recovered.md").write_bytes(payload)
    (import_root / "attachment-recovery.json").write_text(
        json.dumps({"attachments": [{"native_id": "att-1", "path": "nested/recovered.md"}]}),
        encoding="utf-8",
    )
    (import_root / "claude-ai-browser-capture.json").write_text("{}", encoding="utf-8")
    store = BlobStore(tmp_path / "blobs")

    sidecar_data = ClaudeAIAssemblySpec().discover_sidecars(
        [import_root / "claude-ai-browser-capture.json"], blob_store=store
    )

    blob_hash, size = sidecar_data["claude_ai_recovered_blobs"]["att-1"]
    assert blob_hash == hashlib.sha256(payload).hexdigest()
    assert size == len(payload)
