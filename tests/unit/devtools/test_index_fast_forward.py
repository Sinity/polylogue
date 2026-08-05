from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

import devtools.index_fast_forward as forward
from polylogue.core.enums import Provider
from polylogue.sources.dispatch import parse_payload
from polylogue.storage.index_generation import IndexGenerationStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


@dataclass(frozen=True)
class _Report:
    blocking: bool = False
    checks: tuple[object, ...] = ()


def _archive(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    root.mkdir()
    for tier in (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.EMBEDDINGS, ArchiveTier.OPS):
        initialize_archive_database(root / f"{tier.value}.db", tier)
    storage = tmp_path / "storage"
    active_root = storage / ".index-generations" / "v36"
    active_root.mkdir(parents=True)
    active = active_root / "index.db"
    initialize_archive_database(active, ArchiveTier.INDEX)
    (storage / "index.db").symlink_to(active)
    (root / "index.db").symlink_to(storage / "index.db")

    payload_object = {
        "id": "source-backed-session",
        "conversation_id": "source-backed-session",
        "title": "source-backed proof",
        "mapping": {
            "message": {
                "id": "message",
                "parent": None,
                "children": [],
                "message": {
                    "id": "message",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["retained raw proof"]},
                    "create_time": 1_700_000_000,
                },
            }
        },
    }
    payload = json.dumps(payload_object, sort_keys=True).encode()
    parsed = parse_payload(
        Provider.CHATGPT,
        payload_object,
        "source-backed-session",
        source_path="source-backed-session.json",
    )
    assert len(parsed) == 1
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        archive.write_raw_and_parsed(
            parsed[0],
            payload=payload,
            source_path="source-backed-session.json",
            acquired_at_ms=1,
        )
    with sqlite3.connect(active) as conn:
        conn.execute("CREATE TABLE session_runs(id TEXT)")
        conn.execute("CREATE TABLE session_observed_events(id TEXT)")
        conn.execute("CREATE TABLE session_context_snapshots(id TEXT)")
        conn.execute(
            "INSERT INTO attachment_native_ids(ref_id, id_kind, native_id) VALUES ('orphan-ref', 'url', 'orphan')"
        )
        conn.execute("PRAGMA user_version = 36")
        conn.commit()
    return root


@pytest.fixture
def _patch_v37(monkeypatch: pytest.MonkeyPatch) -> None:
    import polylogue.storage.sqlite.lifecycle as lifecycle
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER

    monkeypatch.setattr(
        lifecycle,
        "INDEX_DELTA_DECLARATIONS",
        tuple(declaration for declaration in lifecycle.INDEX_DELTA_DECLARATIONS if 36 <= declaration.version <= 37),
    )
    monkeypatch.setitem(ARCHIVE_VERSION_BY_TIER, ArchiveTier.INDEX, 37)


def _no_corpus_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(forward, "verify_archive", lambda *args, **kwargs: _Report())


def test_prepare_receipt_proves_source_replay_equivalence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _patch_v37: None
) -> None:
    root = _archive(tmp_path)
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: None)

    receipt_path = tmp_path / "transition.json"
    prepared = forward.prepare_forward(archive_root=root, receipt_path=receipt_path)

    proof = prepared["proof"]
    assert isinstance(proof, dict)
    assert proof["verdict"] == "equivalent"
    assert proof["mismatch_details"] == []
    assert proof["replayed_session_ids"]
    assert {"sessions", "messages", "blocks", "fts", "scoped"} <= set(proof["fast_forward_hashes"])
    assert "session_links" in proof["fast_forward_hashes"]["scoped"]
    assert prepared["sample_manifest"]
    fingerprints = prepared["fingerprints"]
    assert isinstance(fingerprints, dict)
    assert fingerprints["parser"]
    assert fingerprints["lowering"]
    assert fingerprints["materializer"]

    _no_corpus_failure(monkeypatch)
    activated = forward.activate_forward(receipt_path=receipt_path)
    assert activated["status"] == "activated"
    assert IndexGenerationStore.for_archive_root(root).active_pointer.resolve().parent.name.startswith("gen-")
    with sqlite3.connect(IndexGenerationStore.for_archive_root(root).active_pointer.resolve()) as conn:
        assert conn.execute("SELECT 1 FROM attachment_native_ids WHERE ref_id = 'orphan-ref'").fetchone() is None


def test_activation_refuses_parser_or_materializer_fingerprint_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _patch_v37: None
) -> None:
    root = _archive(tmp_path)
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: None)
    receipt_path = tmp_path / "transition.json"
    forward.prepare_forward(archive_root=root, receipt_path=receipt_path)
    monkeypatch.setattr(forward, "_materializer_fingerprint", lambda: "changed-materializer")
    _no_corpus_failure(monkeypatch)

    with pytest.raises(forward.IndexFastForwardError, match="fingerprints changed"):
        forward.activate_forward(receipt_path=receipt_path)

    assert IndexGenerationStore.for_archive_root(root).active_pointer.resolve().parent.name == "v36"


def test_activation_refuses_source_metadata_mutation_after_candidate_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _patch_v37: None
) -> None:
    root = _archive(tmp_path)
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: None)
    receipt_path = tmp_path / "transition.json"
    forward.prepare_forward(archive_root=root, receipt_path=receipt_path)

    def mutate_source(*args: object, **kwargs: object) -> None:
        with sqlite3.connect(root / "source.db") as conn:
            conn.execute("UPDATE raw_sessions SET source_path = 'mutated-after-proof'")

    monkeypatch.setattr(forward, "_require_candidate_corpus_fidelity", mutate_source)
    with pytest.raises(forward.IndexFastForwardError, match="immediately before promotion"):
        forward.activate_forward(receipt_path=receipt_path)

    assert IndexGenerationStore.for_archive_root(root).active_pointer.resolve().parent.name == "v36"


def test_bypassing_replay_cannot_create_an_activatable_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _patch_v37: None
) -> None:
    root = _archive(tmp_path)
    monkeypatch.setattr(forward, "running_daemon_pid", lambda _config: None)

    def removed_replay(*args: object, **kwargs: object) -> dict[str, object]:
        # Mutation: deleting the production replay/canonical comparison would
        # otherwise be paper-covered by a receipt that only says equivalent.
        return {
            "fast_forward_hashes": {},
            "canonical_replay_hashes": {},
            "replayed_session_ids": [],
            "mismatch_details": [],
            "verdict": "equivalent",
        }

    monkeypatch.setattr(forward, "_replay_sample", removed_replay)
    with pytest.raises(forward.IndexFastForwardError, match="proof"):
        forward.prepare_forward(archive_root=root, receipt_path=tmp_path / "transition.json")

    assert not list(IndexGenerationStore.for_archive_root(root).generations_root.glob("gen-*/index.db"))
