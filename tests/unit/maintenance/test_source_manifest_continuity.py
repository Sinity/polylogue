from __future__ import annotations

import os
from pathlib import Path

import pytest

from polylogue.maintenance.source_manifest_continuity import (
    ConsumptionReceipt,
    MemberState,
    SourceContinuityError,
    SourceDeclaration,
    SourceRole,
    build_source_manifest,
    canonical_source_declarations,
    recheck_source_manifest,
    validate_backup_evidence,
)


def _source(tmp_path: Path, name: str = "source") -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "one.jsonl").write_text("one\n", encoding="utf-8")
    (root / "two.json").write_text("two", encoding="utf-8")
    return root


def test_canonical_declaration_contains_all_source_roles_once(tmp_path: Path) -> None:
    roots = [tmp_path / name for name in ("hooks", "legacy", "restored", "queue", "attachments", "exports", "live")]
    for root in roots:
        root.mkdir()
    declarations = canonical_source_declarations(
        hook_primary=roots[0],
        hook_legacy=[roots[1]],
        restored_spools=[roots[2]],
        browser_queue=roots[3],
        attachments=roots[4],
        exports=[roots[5]],
        live_sources=[roots[6]],
    )
    assert {declaration.source_id for declaration in declarations} == {
        "hooks-primary",
        "hooks-legacy-0",
        "restored-spool-0",
        "browser-queue",
        "attachments",
        "export-0",
        "live-source-0",
    }
    assert sum(declaration.role is SourceRole.SPOOL for declaration in declarations) == 3


def test_duplicate_roots_and_symlinks_fail_closed(tmp_path: Path) -> None:
    root = _source(tmp_path)
    with pytest.raises(SourceContinuityError, match="duplicate root"):
        canonical_source_declarations(
            configured=[
                SourceDeclaration("a", SourceRole.DIRECTORY, root, True),
                SourceDeclaration("b", SourceRole.DIRECTORY, root, True),
            ]
        )
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(SourceContinuityError, match="real directory|unreadable"):
        build_source_manifest([SourceDeclaration("link", SourceRole.DIRECTORY, link, True)])


def test_non_regular_members_fail_closed(tmp_path: Path) -> None:
    root = _source(tmp_path)
    fifo = root / "unreadable-pipe"
    os.mkfifo(fifo)
    with pytest.raises(SourceContinuityError, match="not a regular file"):
        build_source_manifest([SourceDeclaration("live", SourceRole.DIRECTORY, root, True)])


def test_file_backed_source_is_manifested(tmp_path: Path) -> None:
    source = tmp_path / "export.json"
    source.write_text('{"export": true}', encoding="utf-8")
    manifest = build_source_manifest([SourceDeclaration("export", SourceRole.IMMUTABLE_EXPORT, source)])
    assert [(member.relative_path, member.size) for member in manifest.members] == [("export.json", 16)]


def test_member_loss_is_not_hidden_by_equal_aggregate_replacement(tmp_path: Path) -> None:
    root = _source(tmp_path)
    baseline = build_source_manifest([SourceDeclaration("live", SourceRole.DIRECTORY, root, True)])
    (root / "one.jsonl").unlink()
    (root / "replacement.json").write_text("one\n", encoding="utf-8")
    result = recheck_source_manifest(baseline)
    assert result.safe is False
    assert any(state is MemberState.BLOCKED for state in result.states.values())
    assert any(item.startswith("missing:live:one.jsonl") for item in result.blocked)


def test_append_and_rewrite_are_declared_transitions(tmp_path: Path) -> None:
    append_root = _source(tmp_path, "append")
    append = build_source_manifest([SourceDeclaration("append", SourceRole.APPEND_JSONL, append_root, True)])
    with (append_root / "one.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("later\n")
    assert recheck_source_manifest(append).states["append:one.jsonl"] is MemberState.DECLARED_APPEND

    rewrite_root = _source(tmp_path, "rewrite")
    rewrite = build_source_manifest([SourceDeclaration("rewrite", SourceRole.REWRITE_JSONL, rewrite_root, True)])
    (rewrite_root / "one.jsonl").write_text("new\n", encoding="utf-8")
    assert recheck_source_manifest(rewrite).states["rewrite:one.jsonl"] is MemberState.DECLARED_REWRITE


def test_consumed_requires_exact_authenticated_acquisition(tmp_path: Path) -> None:
    root = _source(tmp_path)
    baseline = build_source_manifest([SourceDeclaration("spool", SourceRole.SPOOL, root, True)])
    member = next(member for member in baseline.members if member.relative_path == "one.jsonl")
    (root / member.relative_path).unlink()
    key = "spool:one.jsonl"
    assert (
        recheck_source_manifest(
            baseline,
            consumption_receipts={key: ConsumptionReceipt(member.content_sha256, "sealed-generation-1")},
        ).states[key]
        is MemberState.CONSUMED_AND_ACQUIRED
    )
    assert not recheck_source_manifest(baseline, consumed={key: member.content_sha256}).safe


def test_append_requires_the_original_prefix(tmp_path: Path) -> None:
    root = _source(tmp_path, "append-prefix")
    baseline = build_source_manifest([SourceDeclaration("append", SourceRole.APPEND_JSONL, root, True)])
    (root / "one.jsonl").write_text("changed\nlater\n", encoding="utf-8")
    result = recheck_source_manifest(baseline)
    assert not result.safe
    assert "append:one.jsonl" in result.states
    assert result.states["append:one.jsonl"] is MemberState.BLOCKED


def test_sqlite_uses_transactional_logical_evidence_not_file_bytes(tmp_path: Path) -> None:
    root = _source(tmp_path, "sqlite")
    declaration = SourceDeclaration("archive", SourceRole.MUTABLE_SQLITE, root, True)
    logical = {"one.jsonl": "relation-hash-1", "two.json": "relation-hash-2"}
    baseline = build_source_manifest([declaration], logical_snapshot=lambda _: logical)
    (root / "one.jsonl").write_text("page-layout-changed", encoding="utf-8")
    assert recheck_source_manifest(baseline, logical_snapshot=lambda _: logical).safe
    assert not recheck_source_manifest(
        baseline, logical_snapshot=lambda _: {**logical, "one.jsonl": "relation-hash-new"}
    ).safe


def test_replacement_with_same_bytes_is_not_unchanged(tmp_path: Path) -> None:
    root = _source(tmp_path)
    baseline = build_source_manifest([SourceDeclaration("live", SourceRole.DIRECTORY, root, True)])
    content = (root / "one.jsonl").read_bytes()
    (root / "one.jsonl").unlink()
    (root / "one.jsonl").write_bytes(content)
    assert not recheck_source_manifest(baseline).safe


def test_baseline_cannot_self_refresh(tmp_path: Path) -> None:
    root = _source(tmp_path)
    baseline = build_source_manifest([SourceDeclaration("live", SourceRole.DIRECTORY, root, True)])
    object.__setattr__(baseline, "manifest_sha256", "refreshed")
    with pytest.raises(SourceContinuityError, match="integrity"):
        recheck_source_manifest(baseline)


def test_backup_evidence_is_external_authenticated_and_fresh() -> None:
    validate_backup_evidence(
        {"authenticated": True, "reference": "backup:42", "observed_at_ms": 900}, now_ms=1000, max_age_ms=200
    )
    with pytest.raises(SourceContinuityError, match="authenticated"):
        validate_backup_evidence({"reference": "backup:42", "observed_at_ms": 900}, now_ms=1000, max_age_ms=200)
    with pytest.raises(SourceContinuityError, match="stale"):
        validate_backup_evidence(
            {"authenticated": True, "reference": "backup:42", "observed_at_ms": 1}, now_ms=1000, max_age_ms=200
        )
