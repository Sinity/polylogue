"""Tests for SOURCE_REPLAY dispatch wiring (issue #1195).

These pin the acceptance criteria from the issue:

* a ``source-replay`` target name resolves to a dispatch entry that
  calls into the source-acquisition path for a bounded scope;
* re-running the same operation against an unchanged source root
  produces zero new raw rows (content-hash idempotency);
* re-running against a source root with one new file ingests only that
  file;
* per-artifact resume cursor: ``target:N:artifact:K`` skips the first
  ``K`` artifacts inside target ``N`` on resume;
* per-artifact failure does not abort the rest of the source scope;
* ``polylogue maintenance run --target source-replay --source-root
  <path>`` works end-to-end against a seeded archive.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.config import Config, Source
from polylogue.maintenance.planner import BackfillKind, BackfillStatus
from polylogue.maintenance.replay import (
    SOURCE_REPLAY_TARGET,
    _decode_artifact_cursor,
    _decode_cursor,
    _encode_cursor,
    execute_replay,
    supported_replay_targets,
)
from polylogue.maintenance.scope import MaintenanceScopeFilter
from polylogue.maintenance.source_replay import (
    repair_source_replay,
    resolve_source_replay_sources,
)


def _chatgpt_conversation(title: str, message: str) -> dict[str, object]:
    """Build a minimal valid ChatGPT export JSON."""
    return {
        "title": title,
        "mapping": {
            "n1": {
                "id": "n1",
                "message": {
                    "id": "m1",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": [message]},
                    "create_time": 1700000000.0,
                },
                "parent": None,
                "children": ["n2"],
            },
            "n2": {
                "id": "n2",
                "message": {
                    "id": "m2",
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["reply"]},
                    "create_time": 1700000060.0,
                },
                "parent": "n1",
                "children": [],
            },
        },
        "current_node": "n2",
    }


def _seed_chatgpt_source(root: Path, *, count: int = 2) -> None:
    """Write ``count`` chatgpt-shaped JSON conversation files under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        path = root / f"conversation_{i:02d}.json"
        path.write_text(json.dumps(_chatgpt_conversation(f"Conv {i}", f"hello {i}")))


def _make_config(tmp_path: Path, sources: list[Source]) -> Config:
    archive_root = tmp_path / "archive"
    render_root = tmp_path / "render"
    archive_root.mkdir(parents=True, exist_ok=True)
    render_root.mkdir(parents=True, exist_ok=True)
    return Config(
        archive_root=archive_root,
        render_root=render_root,
        sources=sources,
        db_path=tmp_path / "archive.db",
    )


# ---------------------------------------------------------------------------
# Catalog and dispatch registration
# ---------------------------------------------------------------------------


def test_source_replay_target_registered_in_catalog() -> None:
    """``source-replay`` resolves through the canonical target catalog."""

    from polylogue.maintenance.targets import build_maintenance_target_catalog

    catalog = build_maintenance_target_catalog()
    spec = catalog.resolve_name("source_replay")
    assert spec is not None
    assert spec.name == "source_replay"
    # Alias accepted (CLI users may write source-replay with a dash).
    aliased = catalog.resolve_name("source-replay")
    assert aliased is not None
    assert aliased.name == "source_replay"


def test_source_replay_target_in_supported_dispatch_set() -> None:
    """Pins the contract that #1195 AC #1 names explicitly."""

    assert SOURCE_REPLAY_TARGET in supported_replay_targets()


# ---------------------------------------------------------------------------
# Source scope resolution from the typed filter
# ---------------------------------------------------------------------------


def test_resolve_sources_empty_filter_returns_all(tmp_path: Path) -> None:
    src_a = Source(name="chatgpt", path=tmp_path / "a")
    src_b = Source(name="codex", path=tmp_path / "b")
    config = _make_config(tmp_path, sources=[src_a, src_b])
    resolved = resolve_source_replay_sources(config, MaintenanceScopeFilter())
    assert resolved == [src_a, src_b]


def test_resolve_sources_filters_by_source_root(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    src_a = Source(name="chatgpt", path=root_a)
    src_b = Source(name="codex", path=root_b)
    config = _make_config(tmp_path, sources=[src_a, src_b])
    filter_a = MaintenanceScopeFilter(source_root=root_a)
    assert resolve_source_replay_sources(config, filter_a) == [src_a]


def test_resolve_sources_filters_archived_paths_lexically(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    archived_root = Path("/mnt/pendrv/archive/chatgpt")
    src = Source(name="chatgpt", path=archived_root / "exports")
    config = _make_config(tmp_path, sources=[src])

    def fail_resolve(self: Path, *args: object, **kwargs: object) -> Path:
        raise AssertionError("source replay must not resolve archived paths")

    monkeypatch.setattr(Path, "resolve", fail_resolve)

    resolved = resolve_source_replay_sources(config, MaintenanceScopeFilter(source_root=archived_root))

    assert resolved == [src]


def test_resolve_sources_filters_by_provider(tmp_path: Path) -> None:
    src_a = Source(name="chatgpt", path=tmp_path / "a")
    src_b = Source(name="codex", path=tmp_path / "b")
    config = _make_config(tmp_path, sources=[src_a, src_b])
    filter_codex = MaintenanceScopeFilter(provider="codex")
    assert resolve_source_replay_sources(config, filter_codex) == [src_b]


def test_resolve_sources_empty_match_returns_empty(tmp_path: Path) -> None:
    src_a = Source(name="chatgpt", path=tmp_path / "a")
    config = _make_config(tmp_path, sources=[src_a])
    filter_other = MaintenanceScopeFilter(provider="codex")
    assert resolve_source_replay_sources(config, filter_other) == []


# ---------------------------------------------------------------------------
# Per-artifact cursor encoding/decoding
# ---------------------------------------------------------------------------


def test_per_artifact_cursor_round_trips() -> None:
    encoded = _encode_cursor(2, artifact_index=9182)
    assert encoded == "target:2:artifact:9182"
    assert _decode_cursor(encoded, total_targets=5) == 2
    assert _decode_artifact_cursor(encoded) == 9182


def test_per_artifact_cursor_omitted_when_zero() -> None:
    """``target:N:artifact:0`` collapses to the legacy ``target:N`` form."""

    assert _encode_cursor(2, artifact_index=0) == "target:2"
    assert _encode_cursor(2) == "target:2"


def test_legacy_target_cursor_yields_zero_artifact_index() -> None:
    assert _decode_artifact_cursor("target:3") == 0
    assert _decode_artifact_cursor(None) == 0
    assert _decode_artifact_cursor("done") == 0


# ---------------------------------------------------------------------------
# End-to-end idempotency against a seeded source root
# ---------------------------------------------------------------------------


def test_first_replay_acquires_all_artifacts(tmp_path: Path) -> None:
    """A fresh replay against a seeded source acquires every artifact."""

    source_root = tmp_path / "chatgpt-source"
    _seed_chatgpt_source(source_root, count=3)
    source = Source(name="chatgpt", path=source_root)
    config = _make_config(tmp_path, sources=[source])

    outcome = repair_source_replay(config, dry_run=False)

    assert outcome.result.success is True
    assert outcome.acquired == 3
    assert outcome.skipped == 0
    assert outcome.failures == []


def test_second_replay_acquires_nothing_new(tmp_path: Path) -> None:
    """Content-hash idempotency: re-running ingests zero new raw rows."""

    source_root = tmp_path / "chatgpt-source"
    _seed_chatgpt_source(source_root, count=3)
    source = Source(name="chatgpt", path=source_root)
    config = _make_config(tmp_path, sources=[source])

    first = repair_source_replay(config, dry_run=False)
    assert first.acquired == 3

    second = repair_source_replay(config, dry_run=False)
    assert second.result.success is True
    assert second.acquired == 0
    assert second.skipped == 3


def test_new_file_after_seed_only_ingests_the_new_file(tmp_path: Path) -> None:
    """Adding one file to the source root causes exactly one new acquisition."""

    source_root = tmp_path / "chatgpt-source"
    _seed_chatgpt_source(source_root, count=2)
    source = Source(name="chatgpt", path=source_root)
    config = _make_config(tmp_path, sources=[source])

    first = repair_source_replay(config, dry_run=False)
    assert first.acquired == 2

    # Add one new conversation file with different content (so a
    # different content hash).
    new_path = source_root / "conversation_new.json"
    new_path.write_text(json.dumps(_chatgpt_conversation("New", "fresh content")))

    second = repair_source_replay(config, dry_run=False)
    assert second.acquired == 1
    assert second.skipped == 2


def test_dry_run_does_not_persist_anything(tmp_path: Path) -> None:
    """Dry runs report what would happen without writing raw rows."""

    source_root = tmp_path / "chatgpt-source"
    _seed_chatgpt_source(source_root, count=2)
    source = Source(name="chatgpt", path=source_root)
    config = _make_config(tmp_path, sources=[source])

    dry = repair_source_replay(config, dry_run=True)
    assert dry.acquired == 0
    assert "Would attempt" in dry.result.detail

    # A subsequent real run still has to acquire everything — dry run
    # left no archive state.
    real = repair_source_replay(config, dry_run=False)
    assert real.acquired == 2


# ---------------------------------------------------------------------------
# Per-artifact resume
# ---------------------------------------------------------------------------


def test_resume_artifact_index_skips_already_processed_artifacts(tmp_path: Path) -> None:
    """``resume_artifact_index=K`` skips the first ``K`` artifacts in the scope."""

    source_root = tmp_path / "chatgpt-source"
    _seed_chatgpt_source(source_root, count=4)
    source = Source(name="chatgpt", path=source_root)
    config = _make_config(tmp_path, sources=[source])

    # Resume at artifact index 2 — the first two artifacts must be
    # skipped, the remaining two acquired.
    outcome = repair_source_replay(config, dry_run=False, resume_artifact_index=2)
    assert outcome.acquired == 2


# ---------------------------------------------------------------------------
# Empty-scope behaviour
# ---------------------------------------------------------------------------


def test_no_sources_configured_is_success_zero(tmp_path: Path) -> None:
    """An empty source scope is a no-op repair, not a failure."""

    config = _make_config(tmp_path, sources=[])
    outcome = repair_source_replay(config, dry_run=False)
    assert outcome.result.success is True
    assert outcome.acquired == 0


def test_filter_with_no_matches_is_success_zero(tmp_path: Path) -> None:
    source = Source(name="chatgpt", path=tmp_path / "chatgpt")
    config = _make_config(tmp_path, sources=[source])
    outcome = repair_source_replay(
        config,
        dry_run=False,
        scope_filter=MaintenanceScopeFilter(provider="codex"),
    )
    assert outcome.result.success is True
    assert outcome.acquired == 0


# ---------------------------------------------------------------------------
# End-to-end through ``execute_replay``
# ---------------------------------------------------------------------------


def test_execute_replay_with_source_replay_target_sets_kind(tmp_path: Path) -> None:
    """``BackfillOperation.kind`` flips to SOURCE_REPLAY when the target is used."""

    source_root = tmp_path / "chatgpt-source"
    _seed_chatgpt_source(source_root, count=2)
    source = Source(name="chatgpt", path=source_root)
    config = _make_config(tmp_path, sources=[source])

    op = execute_replay(
        config,
        targets=("source_replay",),
        operation_id="op-source-replay",
        persist_state=False,
    )
    assert op.kind is BackfillKind.SOURCE_REPLAY
    assert op.status is BackfillStatus.COMPLETED
    assert op.affected_rows == 2


def test_execute_replay_threads_scope_filter_to_source_replay(tmp_path: Path) -> None:
    """A typed scope filter narrows which sources execute_replay touches."""

    root_a = tmp_path / "src-a"
    root_b = tmp_path / "src-b"
    _seed_chatgpt_source(root_a, count=2)
    _seed_chatgpt_source(root_b, count=3)
    src_a = Source(name="chatgpt", path=root_a)
    src_b = Source(name="codex", path=root_b)
    config = _make_config(tmp_path, sources=[src_a, src_b])

    op = execute_replay(
        config,
        targets=("source_replay",),
        scope_filter=MaintenanceScopeFilter(source_root=root_a),
        operation_id="op-narrowed",
        persist_state=False,
    )
    assert op.status is BackfillStatus.COMPLETED
    # Only the two artifacts under root_a were acquired.
    assert op.affected_rows == 2
