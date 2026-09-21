from __future__ import annotations

import os
from pathlib import Path

import pytest

from polylogue.maintenance.source_manifest_continuity import (
    ConsumptionReceipt,
    FrontierState,
    MemberState,
    SourceContinuityError,
    SourceDeclaration,
    SourceRole,
    WantedSourceReceiptError,
    build_source_frontier,
    build_source_manifest,
    build_wanted_source_receipt,
    campaign_default_wanted_source_policy,
    canonical_source_declarations,
    load_wanted_source_receipt,
    preflight_rebuild,
    recheck_source_manifest,
    validate_backup_evidence,
    write_wanted_source_receipt,
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
    with pytest.raises(SourceContinuityError, match="duplicate roots"):
        build_source_frontier(
            [
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


def test_frontier_retains_missing_roots_and_valid_empty_roots(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    frontier = build_source_frontier(
        [
            SourceDeclaration("empty", SourceRole.DIRECTORY, empty, True),
            SourceDeclaration("missing", SourceRole.DIRECTORY, tmp_path / "missing", True),
        ]
    )
    assert frontier.root_states["empty"] is FrontierState.VALID_EMPTY
    assert frontier.root_states["missing"] is FrontierState.UNAVAILABLE
    assert frontier.complete is False
    assert any(item.startswith("unavailable:missing:") for item in frontier.blockers)
    frontier.verify_integrity()


def test_frontier_digest_binds_captured_member_after_path_mutation(tmp_path: Path) -> None:
    root = _source(tmp_path, "captured")
    frontier = build_source_frontier([SourceDeclaration("captured", SourceRole.DIRECTORY, root, True)])
    (root / "one.jsonl").write_text("changed", encoding="utf-8")
    # The captured denominator remains verifiable without rereading mutable
    # source bytes; a changed live root is a new observation, not a refreshed
    # frontier.
    frontier.verify_integrity()
    assert frontier.members[0].content_sha256 != ""


def test_wanted_source_receipt_round_trip_and_operator_preflight_are_private(tmp_path: Path) -> None:
    source = _source(tmp_path, "wanted")
    archive = tmp_path / "archive"
    archive.mkdir()
    declaration = SourceDeclaration("configured-0", SourceRole.DIRECTORY, source, True)

    receipt = write_wanted_source_receipt(archive, [declaration])
    loaded = load_wanted_source_receipt(archive, declarations=[declaration])
    preflight = preflight_rebuild(archive, declarations=[declaration])

    assert loaded.receipt_sha256 == receipt.receipt_sha256
    assert loaded.item_count == 2
    assert preflight.as_dict() == {
        "outcome": "ok",
        "receipt_sha256": receipt.receipt_sha256,
        "policy_identity": receipt.policy_identity,
        "declaration_sha256": receipt.declaration_sha256,
        "frontier_sha256": receipt.frontier_sha256,
        "item_count": 2,
        "byte_count": 7,
    }
    # The operator-facing projection carries proof identities, never roots or
    # member coordinates.
    assert "root" not in preflight.as_dict()
    assert "relative_path" not in preflight.as_dict()


def test_wanted_source_receipt_rejects_tampering_and_policy_revision(tmp_path: Path) -> None:
    source = _source(tmp_path, "tamper")
    archive = tmp_path / "archive"
    archive.mkdir()
    declaration = SourceDeclaration("configured-0", SourceRole.DIRECTORY, source, True)
    receipt = write_wanted_source_receipt(archive, [declaration])
    path = archive / ".maintenance-state" / "wanted-sources" / "selected.json"
    payload = path.read_text(encoding="utf-8").replace(receipt.frontier_sha256, "0" * 64)
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(WantedSourceReceiptError, match="integrity"):
        load_wanted_source_receipt(archive, declarations=[declaration])

    write_wanted_source_receipt(
        archive,
        [declaration],
        policy=campaign_default_wanted_source_policy().__class__(revision="2"),
    )
    with pytest.raises(WantedSourceReceiptError, match="policy mismatch"):
        load_wanted_source_receipt(archive, declarations=[declaration])


def test_wanted_source_receipt_refuses_missing_root_and_duplicate_members(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    missing = SourceDeclaration("missing", SourceRole.DIRECTORY, tmp_path / "absent", True)
    incomplete = build_wanted_source_receipt([missing])
    assert incomplete.blockers and not incomplete.complete
    write_wanted_source_receipt(archive, [missing])
    with pytest.raises(WantedSourceReceiptError, match="missing or unavailable"):
        load_wanted_source_receipt(archive)

    source = _source(tmp_path, "duplicate")
    declaration = SourceDeclaration("duplicate", SourceRole.DIRECTORY, source, True)
    receipt = build_wanted_source_receipt([declaration])
    object.__setattr__(receipt, "members", receipt.members + (receipt.members[0],))
    with pytest.raises(WantedSourceReceiptError, match="duplicate"):
        receipt.verify_integrity()


def test_same_byte_replacement_is_a_new_wanted_source_identity(tmp_path: Path) -> None:
    source = _source(tmp_path, "replacement")
    declaration = SourceDeclaration("replacement", SourceRole.DIRECTORY, source, True)
    first = build_wanted_source_receipt([declaration])
    original = (source / "one.jsonl").read_bytes()
    (source / "one.jsonl").unlink()
    (source / "one.jsonl").write_bytes(original)
    second = build_wanted_source_receipt([declaration])
    assert first.members[0].content_sha256 == second.members[0].content_sha256
    assert first.members[0].identity != second.members[0].identity
    assert first.receipt_sha256 != second.receipt_sha256


def test_campaign_policy_excludes_non_standalone_kinds_but_keeps_raw_evidence(tmp_path: Path) -> None:
    wanted = _source(tmp_path, "wanted-kind")
    discovery = _source(tmp_path, "discovery-kind")
    declarations = [
        SourceDeclaration("standalone", SourceRole.DIRECTORY, wanted, True),
        SourceDeclaration("discovery", SourceRole.DIRECTORY, discovery, True),
    ]
    receipt = build_wanted_source_receipt(
        declarations,
        source_kinds={"discovery": "discovery-only"},
    )

    assert [member.source_id for member in receipt.members] == ["standalone", "standalone"]
    assert receipt.excluded_source_ids == ("discovery",)
    assert receipt.complete


def test_policy_classification_is_bound_when_receipt_is_loaded_and_preflighted(tmp_path: Path) -> None:
    wanted = _source(tmp_path, "wanted-classified")
    discovery = _source(tmp_path, "discovery-classified")
    declarations = [
        SourceDeclaration("standalone", SourceRole.DIRECTORY, wanted, True),
        SourceDeclaration("discovery", SourceRole.DIRECTORY, discovery, True),
    ]
    source_kinds = {"discovery": "discovery-only"}
    archive = tmp_path / "archive-classified"
    archive.mkdir()

    receipt = write_wanted_source_receipt(archive, declarations, source_kinds=source_kinds)
    loaded = load_wanted_source_receipt(archive, declarations=declarations, source_kinds=source_kinds)
    preflight = preflight_rebuild(archive, declarations=declarations, source_kinds=source_kinds)

    assert loaded.receipt_sha256 == receipt.receipt_sha256
    assert preflight.receipt_sha256 == receipt.receipt_sha256
    with pytest.raises(WantedSourceReceiptError, match="declaration mismatch"):
        load_wanted_source_receipt(archive, declarations=declarations)


#: The receipt ``write_wanted_source_receipt`` produced for an empty
#: declaration set before this guard existed -- byte-for-byte, and carrying no
#: path or source identity. Keeping the real artifact makes the load-side
#: refusal a regression against a receipt an operator could already hold, not
#: against a hand-built approximation.
_PRE_GUARD_EMPTY_RECEIPT = """{
  "blockers": [],
  "byte_count": 0,
  "complete": true,
  "declaration_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
  "declarations": [],
  "excluded_source_ids": [],
  "frontier_sha256": "8e13d2edfadffdfd8e6c99d9f21083f35bd347abc5b329ebc3d4b4ee6adc3153",
  "item_count": 0,
  "members": [],
  "policy": {
    "excluded_kinds": [
      "discovery-only",
      "experimental",
      "optional",
      "native-sinex"
    ],
    "identity": "6e526aba9b3e717874318e6c5c22bad801debe1801b6ea32302c404ac127d5f4",
    "included_roles": [
      "immutable-export",
      "archive-member",
      "append-jsonl",
      "rewrite-leading-jsonl",
      "mutable-sqlite",
      "spool",
      "queue",
      "attachment",
      "sidecar",
      "provider-cache",
      "directory"
    ],
    "name": "campaign-default",
    "revision": "1"
  },
  "receipt_sha256": "7f4e8852955b3b34047a72e21c451373128be463c35041a181b71a017e39787b",
  "root_states": {},
  "schema": "polylogue.wanted-source.v1"
}"""


def test_an_empty_denominator_can_neither_be_frozen_nor_authorize_a_rebuild(tmp_path: Path) -> None:
    """A receipt that selects no declaration is not a denominator.

    Anti-vacuity: remove the empty-declaration guard from
    ``WantedSourceReceipt.verify_integrity`` and the frozen artifact below
    -- which this code really produced, and which a runtime with no
    configured source root still produces -- loads as ``complete`` and
    preflights the final rebuild against zero items and zero bytes. Every
    conservation ratio measured against it is then vacuously satisfied.
    """
    archive = tmp_path / "archive-empty"
    archive.mkdir()

    with pytest.raises(WantedSourceReceiptError, match="no source is declared"):
        write_wanted_source_receipt(archive, [])
    assert not (archive / ".maintenance-state" / "wanted-sources" / "selected.json").exists()

    excluded = _source(tmp_path, "discovery-everything")
    with pytest.raises(WantedSourceReceiptError, match="excluded every declaration"):
        build_wanted_source_receipt(
            [SourceDeclaration("discovery", SourceRole.DIRECTORY, excluded, True)],
            source_kinds={"discovery": "discovery-only"},
        )

    receipt_path = archive / ".maintenance-state" / "wanted-sources" / "selected.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(_PRE_GUARD_EMPTY_RECEIPT, encoding="utf-8")
    with pytest.raises(WantedSourceReceiptError, match="selects no declaration"):
        load_wanted_source_receipt(archive)
    with pytest.raises(WantedSourceReceiptError, match="selects no declaration"):
        preflight_rebuild(archive)
