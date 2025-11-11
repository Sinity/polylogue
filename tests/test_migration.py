from __future__ import annotations

import json

from polylogue.migration import perform_legacy_migration
from polylogue.persistence.state import ConversationStateRepository
from polylogue.persistence.database import ConversationDatabase
from polylogue import util
from tests.conftest import _configure_state


def _write_legacy_files(state_dir, state_payload, runs_payload):
    state_path = state_dir / "state.json"
    runs_path = state_dir / "runs.json"
    state_path.write_text(json.dumps(state_payload), encoding="utf-8")
    runs_path.write_text(json.dumps(runs_payload), encoding="utf-8")
    return state_path, runs_path


def test_perform_legacy_migration_imports_state_and_runs(tmp_path, monkeypatch):
    state_dir = _configure_state(monkeypatch, tmp_path)
    state_payload = {
        "conversations": {
            "render": {
                "conv-1": {
                    "slug": "legacy-render",
                    "outputPath": str(tmp_path / "out" / "conversation.md"),
                    "contentHash": "abc",
                }
            }
        }
    }
    runs_payload = [
        {"cmd": "render", "count": 1, "attachments": 2, "attachmentBytes": 1024, "timestamp": "2024-01-01T00:00:00Z"}
    ]
    state_path, runs_path = _write_legacy_files(state_dir, state_payload, runs_payload)

    report = perform_legacy_migration(state_path=state_path, runs_path=runs_path)

    assert report.conversations_migrated == 1
    assert report.runs_migrated == 1

    repo = ConversationStateRepository(database=ConversationDatabase(path=state_dir / "polylogue.db"))
    entry = repo.get("render", "conv-1")
    assert entry is not None
    assert entry.get("slug") == "legacy-render"

    runs = util.load_runs()
    assert runs and runs[-1]["cmd"] == "render"


def test_perform_legacy_migration_dry_run(tmp_path, monkeypatch):
    state_dir = _configure_state(monkeypatch, tmp_path)
    state_payload = {"conversations": {"render": {"conv-dry": {"outputPath": "dry.md"}}}}
    runs_payload = [{"cmd": "render", "count": 1}]
    state_path, runs_path = _write_legacy_files(state_dir, state_payload, runs_payload)

    report = perform_legacy_migration(state_path=state_path, runs_path=runs_path, dry_run=True)
    assert report.conversations_migrated == 1
    assert report.runs_migrated == 1
    assert not util.load_runs()


def test_perform_legacy_migration_force_runs(tmp_path, monkeypatch):
    state_dir = _configure_state(monkeypatch, tmp_path)
    util.add_run({"cmd": "render", "count": 1})
    _, runs_path = _write_legacy_files(
        state_dir,
        {"conversations": {}},
        [
            {"cmd": "render", "count": 1, "timestamp": "2024-01-01T00:00:00Z"},
            {"cmd": "render", "count": 2, "timestamp": "2024-01-02T00:00:00Z"},
        ],
    )

    report_skip = perform_legacy_migration(runs_path=runs_path)
    assert report_skip.runs_skipped

    report_force = perform_legacy_migration(runs_path=runs_path, force_runs=True)
    assert report_force.runs_migrated == 2
    runs = util.load_runs()
    assert len(runs) == 2


def test_perform_legacy_migration_idempotent(tmp_path, monkeypatch):
    state_dir = _configure_state(monkeypatch, tmp_path)
    state_payload = {
        "conversations": {
            "render": {
                "conv-1": {
                    "slug": "legacy-render",
                    "outputPath": str(tmp_path / "out" / "conversation.md"),
                }
            }
        }
    }
    runs_payload = [
        {"cmd": "render", "provider": "render", "count": 1, "timestamp": "2024-02-01T00:00:00Z"}
    ]
    state_path, runs_path = _write_legacy_files(state_dir, state_payload, runs_payload)

    first = perform_legacy_migration(state_path=state_path, runs_path=runs_path)
    assert first.conversations_migrated == 1
    assert first.runs_migrated == 1

    second = perform_legacy_migration(state_path=state_path, runs_path=runs_path)
    assert second.conversations_migrated == 1  # still scanned
    assert second.runs_migrated == 0
    assert second.runs_skipped is True

    repo = ConversationStateRepository(database=ConversationDatabase(path=state_dir / "polylogue.db"))
    entry = repo.get("render", "conv-1")
    assert entry and entry.get("slug") == "legacy-render"
    runs = util.load_runs()
    assert len(runs) == 1


def test_perform_legacy_migration_overwrites_existing_state(tmp_path, monkeypatch):
    state_dir = _configure_state(monkeypatch, tmp_path)
    repo = ConversationStateRepository(database=ConversationDatabase(path=state_dir / "polylogue.db"))
    repo.upsert("render", "conv-1", {"slug": "old-slug"})

    state_payload = {
        "conversations": {
            "render": {
                "conv-1": {
                    "slug": "new-slug",
                    "outputPath": str(tmp_path / "updated" / "conversation.md"),
                }
            }
        }
    }
    runs_payload = []
    state_path, runs_path = _write_legacy_files(state_dir, state_payload, runs_payload)

    report = perform_legacy_migration(state_path=state_path, runs_path=runs_path)
    assert report.conversations_migrated == 1
    entry = repo.get("render", "conv-1")
    assert entry is not None
    assert entry.get("slug") == "new-slug"
    assert entry.get("outputPath").endswith("conversation.md")
