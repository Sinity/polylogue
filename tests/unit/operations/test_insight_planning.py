"""Insight planning freezes exact targets from one supplied index reader."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.operations.insight_acceptance import (
    MAX_INSIGHT_ACCEPTED_PARTS,
    MAX_INSIGHT_PART_TARGETS,
    accepted_part_from_plan,
)
from polylogue.operations.insight_planning import insight_page_plan, prepare_insight_manifest
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root


def _seed_sessions(root: Path, count: int, *, profile_orphan: bool = False) -> None:
    bootstrap_archive_root(root)
    with sqlite3.connect(root / "index.db") as conn:
        conn.executemany(
            "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, 'codex-session', zeroblob(32))",
            ((f"target-{index:05d}",) for index in range(count)),
        )
        if profile_orphan:
            # Historical derived rows can outlive their source session.  The
            # full sweep's UNION must retain that evidence as an excess target.
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("INSERT INTO session_profiles (session_id) VALUES (?)", ("codex-session:orphan",))


@contextmanager
def _reader(tmp_path: Path, count: int, *, profile_orphan: bool = False) -> Iterator[ArchiveStore]:
    root = tmp_path / "archive"
    _seed_sessions(root, count, profile_orphan=profile_orphan)
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        yield archive


def _prepare(archive: ArchiveStore, session_ids: list[str] | None = None):
    return prepare_insight_manifest(
        archive,
        session_ids,
        index_generation="index-generation:synthetic",
        recipe_version="insight-recipe-v1",
        check_stop=lambda: None,
    )


@pytest.mark.parametrize("count", [257, 10_001])
def test_full_manifest_is_exactly_paged_and_digest_stable(tmp_path: Path, count: int) -> None:
    with _reader(tmp_path, count) as archive:
        manifest = _prepare(archive)
        repeated = _prepare(archive)

    assert manifest.scope_kind == "full"
    assert len(manifest.pages) == (count + MAX_INSIGHT_PART_TARGETS - 1) // MAX_INSIGHT_PART_TARGETS
    assert len(manifest.pages) <= MAX_INSIGHT_ACCEPTED_PARTS
    assert all(len(page) <= MAX_INSIGHT_PART_TARGETS for page in manifest.pages)
    assert sum(len(page) for page in manifest.pages) == count
    assert manifest.pages == repeated.pages
    assert manifest.digest == repeated.digest
    assert len(manifest.digest) == 64
    assert all(char in "0123456789abcdef" for char in manifest.digest)
    assert [target.target_ref for page in manifest.pages for target in page] == [
        f"session:codex-session:target-{index:05d}" for index in range(count)
    ]


def test_full_manifest_keeps_profile_orphan_as_excess_target(tmp_path: Path) -> None:
    with _reader(tmp_path, 1, profile_orphan=True) as archive:
        manifest = _prepare(archive)

    targets = [target for page in manifest.pages for target in page]
    assert [(target.target_ref, target.disposition) for target in targets] == [
        ("session:codex-session:orphan", "excess"),
        ("session:codex-session:target-00000", "required"),
    ]


def test_post_pin_index_insert_is_not_added_to_manifest(tmp_path: Path) -> None:
    with _reader(tmp_path, 257) as archive:
        manifest = _prepare(archive)
        with sqlite3.connect(tmp_path / "archive" / "index.db") as inserted:
            inserted.execute(
                "INSERT INTO sessions (native_id, origin, content_hash) VALUES (?, 'codex-session', zeroblob(32))",
                ("inserted-after-pin",),
            )

        assert "session:codex-session:inserted-after-pin" not in {
            target.target_ref for page in manifest.pages for target in page
        }
        assert sum(len(page) for page in manifest.pages) == 257


def test_explicit_alias_dedup_and_no_match_match_canonical_selection(tmp_path: Path) -> None:
    with _reader(tmp_path, 2) as archive:
        canonical = _prepare(archive, ["codex-session:target-00000"])
        aliases = _prepare(
            archive,
            [
                "target-00000",
                "codex-session:target-00000",
                "target-00000",
                "does-not-exist",
            ],
        )

    assert aliases.scope_kind == canonical.scope_kind == "explicit"
    assert aliases.pages == canonical.pages
    assert aliases.digest == canonical.digest
    assert [target.target_ref for target in aliases.pages[0]] == ["session:codex-session:target-00000"]


def test_previous_preview_reference_changes_page_plan_hash_not_manifest_digest(tmp_path: Path) -> None:
    with _reader(tmp_path, 257) as archive:
        manifest = _prepare(archive)
        first = insight_page_plan(
            manifest,
            1,
            previous_preview_ref="preview:previous-a",
            archive_instance_id="archive-instance",
            archive_identity_digest="a" * 64,
            now_ms=10,
            expires_at_ms=20,
        )
        second = insight_page_plan(
            manifest,
            1,
            previous_preview_ref="preview:previous-b",
            archive_instance_id="archive-instance",
            archive_identity_digest="a" * 64,
            now_ms=10,
            expires_at_ms=20,
        )

    assert first.plan_hash != second.plan_hash
    assert first.context["manifest_digest"] == second.context["manifest_digest"] == manifest.digest
    assert first.context["targets"] == second.context["targets"]


def test_unknown_or_mutated_target_is_rejected_by_shared_plan_context(tmp_path: Path) -> None:
    with _reader(tmp_path, 1) as archive:
        manifest = _prepare(archive)
        plan = insight_page_plan(
            manifest,
            0,
            previous_preview_ref=None,
            archive_instance_id="archive-instance",
            archive_identity_digest="b" * 64,
            now_ms=10,
            expires_at_ms=20,
        )

    mutated_targets = [dict(target) for target in plan.context["targets"]]
    mutated_targets[0]["target_ref"] = "session:unknown"
    mutated_context = dict(plan.context)
    mutated_context["targets"] = mutated_targets
    mutated = replace(plan, context=mutated_context)

    with pytest.raises(ValueError, match="target sequence"):
        accepted_part_from_plan(mutated, preview_ref="preview:one", authorization_ref="authorization:one")
