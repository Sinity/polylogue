from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from devtools import verify_backlog_hygiene

# Fixed anchor instant for F3 (stale-claim) tests -- passed explicitly as
# collect_findings' `now` param rather than mocking datetime.now, so these
# tests are deterministic regardless of the host wall clock.
_ANCHOR = datetime(2026, 7, 21, 12, 0, 0, tzinfo=timezone.utc)


def _write_jsonl(path: Path, issues: list[dict[str, object]]) -> None:
    lines = []
    for issue in issues:
        payload = {"_type": "issue", **issue}
        lines.append(json.dumps(payload))
    path.write_text("\n".join(lines) + "\n")


def _write_receipt(receipts_dir: Path, payload: dict[str, object], *, name: str) -> Path:
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts_dir / name
    receipt_path.write_text(json.dumps(payload))
    return receipt_path


def _clean_receipt(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "created_at": "2026-07-20T00:00:00Z",
        "source": "test",
        "is_clean": True,
        "outcomes": [],
    }
    payload.update(overrides)
    return payload


def _issue(
    id: str,
    *,
    title: str = "A bead",
    description: str = "",
    design: str = "",
    acceptance_criteria: str = "",
    notes: str = "",
    status: str = "open",
    priority: int = 3,
    issue_type: str = "task",
    labels: list[str] | None = None,
    dependencies: list[dict[str, str]] | None = None,
    metadata: dict[str, object] | None = None,
    updated_at: str | None = None,
) -> dict[str, object]:
    return {
        "id": id,
        "title": title,
        "description": description,
        "design": design,
        "acceptance_criteria": acceptance_criteria,
        "notes": notes,
        "status": status,
        "priority": priority,
        "issue_type": issue_type,
        "labels": labels if labels is not None else [],
        "dependencies": dependencies if dependencies is not None else [],
        "metadata": metadata if metadata is not None else {},
        "updated_at": updated_at,
    }


def test_clean_backlog_has_no_findings(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [
            _issue(
                "polylogue-clean1",
                title="A clean bead",
                acceptance_criteria="Verify: run the thing and check the output.",
                labels=["area:ops"],
            ),
        ],
    )
    findings = verify_backlog_hygiene.collect_findings(
        path=path, allow_path=tmp_path / "no-allowlist.txt", receipts_path=tmp_path / "no-receipts"
    )
    assert findings == []


def test_seeded_violation_per_check_class(tmp_path: Path) -> None:
    """One deliberately-seeded violation per check code; assert every code fires.

    This is the bead's AC clause: "seed one violation per class, assert
    non-zero exit" -- exercised here at the collect_findings layer (main()'s
    exit-code contract is covered separately below).
    """
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [
            # D1: dangling dependency ref.
            _issue(
                "polylogue-d1",
                acceptance_criteria="Verify: x.",
                labels=["area:ops"],
                dependencies=[{"depends_on_id": "polylogue-ghost", "type": "blocks"}],
            ),
            # D2: a blocks-cycle between two open beads.
            _issue(
                "polylogue-cyca",
                acceptance_criteria="Verify: x.",
                labels=["area:ops"],
                dependencies=[{"depends_on_id": "polylogue-cycb", "type": "blocks"}],
            ),
            _issue(
                "polylogue-cycb",
                acceptance_criteria="Verify: x.",
                labels=["area:ops"],
                dependencies=[{"depends_on_id": "polylogue-cyca", "type": "blocks"}],
            ),
            # H1: tech-tree bead without a horizon label.
            _issue(
                "polylogue-h1",
                acceptance_criteria="Verify: y.",
                labels=["tech-tree", "area:ops"],
            ),
            # H2: horizon:vision bead at P1 (should be P3/P4).
            _issue(
                "polylogue-h2",
                acceptance_criteria="Verify: z.",
                labels=["horizon:vision", "area:ops"],
                priority=1,
            ),
            # H3: horizon:frontier bead without acceptance criteria.
            _issue(
                "polylogue-h3",
                labels=["horizon:frontier", "area:ops"],
            ),
            # H4: horizon:frontier bead with AC but no design content.
            _issue(
                "polylogue-h4",
                acceptance_criteria="Verify: q.",
                description="short and vague",
                labels=["horizon:frontier", "area:ops"],
            ),
            # P1 (+ R1): P0 bead without acceptance criteria.
            _issue(
                "polylogue-p1",
                labels=["area:ops"],
                priority=0,
            ),
            # A1: non-epic bead without an area:* label.
            _issue(
                "polylogue-a1",
                acceptance_criteria="Verify: r.",
            ),
            # B1: decision bead that declares adopted/decided but is still open.
            _issue(
                "polylogue-b1",
                acceptance_criteria="Verify: s.",
                description="Status: adopted.",
                labels=["area:ops"],
                issue_type="decision",
            ),
            # E1: epic with no children, no dep edges, no named members.
            _issue(
                "polylogue-e1",
                description="An epic that groups nothing yet.",
                labels=["area:ops"],
                issue_type="epic",
            ),
            # E2: epic without a description.
            _issue(
                "polylogue-e2",
                labels=["area:ops"],
                issue_type="epic",
            ),
            # T1: ephemeral path cited without provenance framing.
            _issue(
                "polylogue-t1",
                acceptance_criteria="Verify: t.",
                description="See /tmp/scratch-notes for details.",
                labels=["area:ops"],
            ),
            # X1: duplicate open titles (case-folded).
            _issue(
                "polylogue-x1a",
                title="Duplicate Title Example",
                acceptance_criteria="Verify: u.",
                labels=["area:ops"],
            ),
            _issue(
                "polylogue-x1b",
                title="duplicate title example",
                acceptance_criteria="Verify: u.",
                labels=["area:ops"],
            ),
            # X2: names a bead id that does not exist.
            _issue(
                "polylogue-x2",
                acceptance_criteria="Verify: v.",
                description="See polylogue-9zk2 for prior discussion.",
                labels=["area:ops"],
            ),
        ],
    )

    # S1: seed an unclean sync receipt (a conflicted row) alongside the
    # per-issue violations, so every check fires from one collect_findings call.
    receipts_dir = tmp_path / "receipts"
    _write_receipt(
        receipts_dir,
        _clean_receipt(
            is_clean=False,
            outcomes=[
                {
                    "id": "polylogue-s1",
                    "outcome": "conflicted",
                    "current_revision": None,
                    "candidate_revision": "2026-07-20T00:00:00Z",
                }
            ],
        ),
        name="20260720T000000Z-test.json",
    )

    findings = verify_backlog_hygiene.collect_findings(
        path=path,
        allow_path=tmp_path / "no-allowlist.txt",
        receipts_path=receipts_dir,
    )
    fired = {f.check for f in findings}
    expected = {"D1", "D2", "H1", "H2", "H3", "H4", "P1", "A1", "B1", "E1", "E2", "T1", "X1", "X2", "R1", "S1"}
    assert expected <= fired, f"missing checks: {expected - fired}"


def test_allowlist_suppresses_matching_finding(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(path, [_issue("polylogue-noarea", acceptance_criteria="Verify: x.")])
    allow_path = tmp_path / "allow.txt"
    allow_path.write_text("A1\tpolylogue-noarea\taccepted exception\n")

    findings = verify_backlog_hygiene.collect_findings(
        path=path, allow_path=allow_path, receipts_path=tmp_path / "no-receipts"
    )
    assert findings == []


def test_check_filter_limits_findings_to_requested_classes(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [
            _issue(
                "polylogue-filtered",
                labels=[],
                dependencies=[{"depends_on_id": "polylogue-missing", "type": "blocks"}],
            )
        ],
    )

    findings = verify_backlog_hygiene.collect_findings(
        path=path,
        allow_path=tmp_path / "no-allowlist.txt",
        checks={"D1", "D2"},
        receipts_path=tmp_path / "no-receipts",
    )
    assert [(finding.check, finding.bead_id) for finding in findings] == [("D1", "polylogue-filtered")]


def test_external_request_id_is_not_parsed_as_truncated_bead_ref(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [
            _issue(
                "polylogue-request-ref",
                acceptance_criteria="Verify the receiver receipt.",
                description="Receiver request polylogue-ext-mrhjgnkn-hzbd33l3 was acknowledged.",
                labels=["area:capture"],
            )
        ],
    )

    findings = verify_backlog_hygiene.collect_findings(
        path=path,
        allow_path=tmp_path / "no-allowlist.txt",
        receipts_path=tmp_path / "no-receipts",
    )
    assert not [finding for finding in findings if finding.check == "X2"]


def test_main_json_reports_findings_and_nonzero_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(path, [_issue("polylogue-noarea", acceptance_criteria="Verify: x.")])
    monkeypatch.setattr(verify_backlog_hygiene, "_get_root", lambda: tmp_path)

    exit_code = verify_backlog_hygiene.main(["--json", str(path)])
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert any(f["check"] == "A1" and f["id"] == "polylogue-noarea" for f in payload["findings"])


def test_main_reports_ok_with_clean_backlog(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [_issue("polylogue-clean1", acceptance_criteria="Verify: x.", labels=["area:ops"])],
    )
    monkeypatch.setattr(verify_backlog_hygiene, "_get_root", lambda: tmp_path)

    assert verify_backlog_hygiene.main([str(path)]) == 0
    assert "zero unhandled findings" in capsys.readouterr().out


def test_main_skips_missing_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(verify_backlog_hygiene, "_get_root", lambda: tmp_path)
    assert verify_backlog_hygiene.main([]) == 0
    assert "does not exist" in capsys.readouterr().out


# --- S1: bd JSONL sync receipt consumption (polylogue-8jg9.1 / polylogue-gxjh.1) ---


def _clean_backlog(tmp_path: Path) -> Path:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [_issue("polylogue-clean1", acceptance_criteria="Verify: x.", labels=["area:ops"])],
    )
    return path


def test_sync_receipt_absent_is_not_a_finding(tmp_path: Path) -> None:
    path = _clean_backlog(tmp_path)
    findings = verify_backlog_hygiene.collect_findings(
        path=path,
        allow_path=tmp_path / "no-allowlist.txt",
        receipts_path=tmp_path / "receipts-that-do-not-exist",
    )
    assert findings == []


def test_sync_receipt_clean_is_not_a_finding(tmp_path: Path) -> None:
    path = _clean_backlog(tmp_path)
    receipts_dir = tmp_path / "receipts"
    _write_receipt(
        receipts_dir,
        _clean_receipt(
            outcomes=[
                {
                    "id": "polylogue-a",
                    "outcome": "new",
                    "current_revision": None,
                    "candidate_revision": "2026-07-20T00:00:00Z",
                },
                {
                    "id": "polylogue-b",
                    "outcome": "updated",
                    "current_revision": "2026-07-19T00:00:00Z",
                    "candidate_revision": "2026-07-20T00:00:00Z",
                },
            ]
        ),
        name="20260720T000000Z-test.json",
    )
    findings = verify_backlog_hygiene.collect_findings(
        path=path, allow_path=tmp_path / "no-allowlist.txt", receipts_path=receipts_dir
    )
    assert findings == []


def test_sync_receipt_corrupt_json_is_a_finding(tmp_path: Path) -> None:
    path = _clean_backlog(tmp_path)
    receipts_dir = tmp_path / "receipts"
    receipts_dir.mkdir()
    (receipts_dir / "20260720T000000Z-test.json").write_text("{not valid json")
    findings = verify_backlog_hygiene.collect_findings(
        path=path, allow_path=tmp_path / "no-allowlist.txt", receipts_path=receipts_dir
    )
    assert [f.check for f in findings] == ["S1"]
    assert "corrupt sync receipt" in findings[0].message


def test_sync_receipt_missing_fields_is_a_finding(tmp_path: Path) -> None:
    path = _clean_backlog(tmp_path)
    receipts_dir = tmp_path / "receipts"
    _write_receipt(receipts_dir, {"created_at": "2026-07-20T00:00:00Z"}, name="20260720T000000Z-test.json")
    findings = verify_backlog_hygiene.collect_findings(
        path=path, allow_path=tmp_path / "no-allowlist.txt", receipts_path=receipts_dir
    )
    assert [f.check for f in findings] == ["S1"]
    assert "incomplete sync receipt" in findings[0].message


def test_sync_receipt_conflicted_row_is_a_finding(tmp_path: Path) -> None:
    path = _clean_backlog(tmp_path)
    receipts_dir = tmp_path / "receipts"
    _write_receipt(
        receipts_dir,
        _clean_receipt(
            is_clean=False,
            outcomes=[
                {
                    "id": "polylogue-conflicted-one",
                    "outcome": "conflicted",
                    "current_revision": None,
                    "candidate_revision": "2026-07-20T00:00:00Z",
                }
            ],
        ),
        name="20260720T000000Z-test.json",
    )
    findings = verify_backlog_hygiene.collect_findings(
        path=path, allow_path=tmp_path / "no-allowlist.txt", receipts_path=receipts_dir
    )
    assert [(f.check, f.bead_id) for f in findings] == [("S1", "polylogue-conflicted-one")]
    assert "conflicted" in findings[0].message


def test_sync_receipt_skipped_downgrade_is_a_finding(tmp_path: Path) -> None:
    path = _clean_backlog(tmp_path)
    receipts_dir = tmp_path / "receipts"
    _write_receipt(
        receipts_dir,
        _clean_receipt(
            is_clean=False,
            outcomes=[
                {
                    "id": "polylogue-downgraded-one",
                    "outcome": "skipped_downgrade",
                    "current_revision": "2026-07-20T00:00:00Z",
                    "candidate_revision": "2026-07-01T00:00:00Z",
                }
            ],
        ),
        name="20260720T000000Z-test.json",
    )
    findings = verify_backlog_hygiene.collect_findings(
        path=path, allow_path=tmp_path / "no-allowlist.txt", receipts_path=receipts_dir
    )
    assert [(f.check, f.bead_id) for f in findings] == [("S1", "polylogue-downgraded-one")]
    assert "downgrade" in findings[0].message


def test_sync_receipt_uses_latest_by_filename(tmp_path: Path) -> None:
    path = _clean_backlog(tmp_path)
    receipts_dir = tmp_path / "receipts"
    # Older receipt is unclean; newest receipt (lexicographically last) is clean.
    _write_receipt(
        receipts_dir,
        _clean_receipt(
            is_clean=False,
            outcomes=[
                {
                    "id": "polylogue-stale-conflict",
                    "outcome": "conflicted",
                    "current_revision": None,
                    "candidate_revision": "2026-07-19T00:00:00Z",
                }
            ],
        ),
        name="20260719T000000Z-test.json",
    )
    _write_receipt(receipts_dir, _clean_receipt(), name="20260720T000000Z-test.json")
    findings = verify_backlog_hygiene.collect_findings(
        path=path, allow_path=tmp_path / "no-allowlist.txt", receipts_path=receipts_dir
    )
    assert findings == []


def test_sync_receipt_allowlist_suppresses_finding(tmp_path: Path) -> None:
    path = _clean_backlog(tmp_path)
    receipts_dir = tmp_path / "receipts"
    _write_receipt(
        receipts_dir,
        _clean_receipt(
            is_clean=False,
            outcomes=[
                {
                    "id": "polylogue-allowed-conflict",
                    "outcome": "conflicted",
                    "current_revision": None,
                    "candidate_revision": "2026-07-20T00:00:00Z",
                }
            ],
        ),
        name="20260720T000000Z-test.json",
    )
    allow_path = tmp_path / "allow.txt"
    allow_path.write_text("S1\tpolylogue-allowed-conflict\taccepted exception\n")
    findings = verify_backlog_hygiene.collect_findings(path=path, allow_path=allow_path, receipts_path=receipts_dir)
    assert findings == []


# --- F1-F4: active-set / execution-focus / program-grouping invariants
# (polylogue-8jg9.1 remaining scope) ---


def test_f1_active_epic_leaf_is_a_finding(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [
            # Program-marker beads need not be issue_type="epic" for this
            # module's checks (only F1 cares about the *leaf's* type); using
            # "task" here avoids incidentally tripping the unrelated
            # E1/E2 epic-membership checks in this minimal fixture.
            _issue(
                "polylogue-prog",
                title="Owning program",
                description="Groups active work.",
                labels=["area:ops"],
                issue_type="task",
                metadata={"frontier_program": "active"},
            ),
            # F1: an epic itself carries frontier=active as a leaf.
            _issue(
                "polylogue-f1epic",
                title="Epic mistakenly admitted as a leaf",
                description="An epic that also claims to be an active leaf.",
                labels=["area:ops"],
                issue_type="epic",
                metadata={"frontier": "active", "frontier_program_ref": "polylogue-prog"},
            ),
        ],
    )
    findings = verify_backlog_hygiene.collect_findings(
        path=path, allow_path=tmp_path / "no-allowlist.txt", receipts_path=tmp_path / "no-receipts"
    )
    assert ("F1", "polylogue-f1epic") in {(f.check, f.bead_id) for f in findings}


def test_f2_active_leaf_missing_program_ref_is_a_finding(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [
            _issue(
                "polylogue-f2missing",
                acceptance_criteria="Verify: x.",
                labels=["area:ops"],
                metadata={"frontier": "active"},
            )
        ],
    )
    findings = verify_backlog_hygiene.collect_findings(
        path=path, allow_path=tmp_path / "no-allowlist.txt", receipts_path=tmp_path / "no-receipts"
    )
    assert [(f.check, f.bead_id) for f in findings] == [("F2", "polylogue-f2missing")]
    assert "no frontier_program_ref" in findings[0].message


def test_f2_active_leaf_dangling_program_ref_is_a_finding(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [
            _issue(
                "polylogue-f2dangling",
                acceptance_criteria="Verify: x.",
                labels=["area:ops"],
                metadata={"frontier": "active", "frontier_program_ref": "polylogue-ghost-prog"},
            )
        ],
    )
    findings = verify_backlog_hygiene.collect_findings(
        path=path, allow_path=tmp_path / "no-allowlist.txt", receipts_path=tmp_path / "no-receipts"
    )
    assert [(f.check, f.bead_id) for f in findings] == [("F2", "polylogue-f2dangling")]
    assert "does not exist" in findings[0].message


def test_f2_active_leaf_program_ref_not_active_program_is_a_finding(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [
            _issue(
                "polylogue-notprog",
                title="Not marked as an active program",
                description="A bead that never got frontier_program=active.",
                labels=["area:ops"],
                issue_type="task",
            ),
            _issue(
                "polylogue-f2mismatch",
                acceptance_criteria="Verify: x.",
                labels=["area:ops"],
                metadata={"frontier": "active", "frontier_program_ref": "polylogue-notprog"},
            ),
        ],
    )
    findings = verify_backlog_hygiene.collect_findings(
        path=path, allow_path=tmp_path / "no-allowlist.txt", receipts_path=tmp_path / "no-receipts"
    )
    assert [(f.check, f.bead_id) for f in findings] == [("F2", "polylogue-f2mismatch")]
    assert "not itself frontier_program=active" in findings[0].message


def test_f3_stale_in_progress_claim_is_a_finding(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [
            _issue(
                "polylogue-f3stale",
                acceptance_criteria="Verify: x.",
                labels=["area:ops"],
                status="in_progress",
                updated_at="2026-07-01T00:00:00Z",  # 20 days before _ANCHOR
            )
        ],
    )
    findings = verify_backlog_hygiene.collect_findings(
        path=path,
        allow_path=tmp_path / "no-allowlist.txt",
        receipts_path=tmp_path / "no-receipts",
        now=_ANCHOR,
    )
    assert [(f.check, f.bead_id) for f in findings] == [("F3", "polylogue-f3stale")]
    assert "no recorded activity" in findings[0].message


def test_f3_recent_in_progress_claim_is_not_a_finding(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [
            _issue(
                "polylogue-f3fresh",
                acceptance_criteria="Verify: x.",
                labels=["area:ops"],
                status="in_progress",
                updated_at="2026-07-20T00:00:00Z",  # 1.5 days before _ANCHOR
            )
        ],
    )
    findings = verify_backlog_hygiene.collect_findings(
        path=path,
        allow_path=tmp_path / "no-allowlist.txt",
        receipts_path=tmp_path / "no-receipts",
        now=_ANCHOR,
    )
    assert findings == []


def test_f3_stale_claim_days_is_configurable(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [
            _issue(
                "polylogue-f3configurable",
                acceptance_criteria="Verify: x.",
                labels=["area:ops"],
                status="in_progress",
                updated_at="2026-07-20T00:00:00Z",  # 1.5 days before _ANCHOR
            )
        ],
    )
    findings = verify_backlog_hygiene.collect_findings(
        path=path,
        allow_path=tmp_path / "no-allowlist.txt",
        receipts_path=tmp_path / "no-receipts",
        now=_ANCHOR,
        stale_claim_days=1,
    )
    assert [(f.check, f.bead_id) for f in findings] == [("F3", "polylogue-f3configurable")]


def test_f4_program_with_no_active_leaves_is_a_finding(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [
            _issue(
                "polylogue-f4orphan",
                title="Program admitted with no members",
                description="Claims frontier_program=active but nothing references it.",
                labels=["area:ops"],
                issue_type="task",
                metadata={"frontier_program": "active"},
            )
        ],
    )
    findings = verify_backlog_hygiene.collect_findings(
        path=path, allow_path=tmp_path / "no-allowlist.txt", receipts_path=tmp_path / "no-receipts"
    )
    assert [(f.check, f.bead_id) for f in findings] == [("F4", "polylogue-f4orphan")]
    assert "no open active leaf" in findings[0].message


def test_f4_program_with_an_active_leaf_is_not_a_finding(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [
            _issue(
                "polylogue-f4prog",
                title="Program with a real member",
                description="Groups active work.",
                labels=["area:ops"],
                issue_type="task",
                metadata={"frontier_program": "active"},
            ),
            _issue(
                "polylogue-f4leaf",
                acceptance_criteria="Verify: x.",
                labels=["area:ops"],
                metadata={"frontier": "active", "frontier_program_ref": "polylogue-f4prog"},
            ),
        ],
    )
    findings = verify_backlog_hygiene.collect_findings(
        path=path, allow_path=tmp_path / "no-allowlist.txt", receipts_path=tmp_path / "no-receipts"
    )
    assert findings == []


def test_seeded_active_set_checks_all_fire_together(tmp_path: Path) -> None:
    """One deliberately-seeded violation per new F1-F4 check in a single fixture."""
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [
            _issue(
                "polylogue-okprog",
                title="Healthy program",
                description="A healthy program.",
                labels=["area:ops"],
                issue_type="task",
                metadata={"frontier_program": "active"},
            ),
            # A real (non-epic) leaf backing polylogue-okprog's frontier_program=active
            # claim, so F4 does not also fire on it -- isolating the F1 epic-leaf
            # violation below from the program-membership concern F4 checks.
            _issue(
                "polylogue-okleaf",
                title="Real member of the healthy program",
                acceptance_criteria="Verify: x.",
                labels=["area:ops"],
                metadata={"frontier": "active", "frontier_program_ref": "polylogue-okprog"},
            ),
            # F1: epic itself admitted as an active leaf.
            _issue(
                "polylogue-af1",
                title="Epic mistakenly admitted",
                description="Epic mistakenly admitted as a leaf.",
                labels=["area:ops"],
                issue_type="epic",
                metadata={"frontier": "active", "frontier_program_ref": "polylogue-okprog"},
            ),
            # F2: active leaf with a dangling program ref.
            _issue(
                "polylogue-af2",
                title="Leaf with dangling program ref",
                acceptance_criteria="Verify: x.",
                labels=["area:ops"],
                metadata={"frontier": "active", "frontier_program_ref": "polylogue-ghost"},
            ),
            # F3: stale in_progress claim.
            _issue(
                "polylogue-af3",
                title="Stale in-progress claim",
                acceptance_criteria="Verify: x.",
                labels=["area:ops"],
                status="in_progress",
                updated_at="2026-07-01T00:00:00Z",
            ),
            # F4: program admitted with no referencing leaves.
            _issue(
                "polylogue-af4",
                title="Orphaned program admission",
                description="Orphaned program admission.",
                labels=["area:ops"],
                issue_type="task",
                metadata={"frontier_program": "active"},
            ),
        ],
    )
    findings = verify_backlog_hygiene.collect_findings(
        path=path,
        allow_path=tmp_path / "no-allowlist.txt",
        receipts_path=tmp_path / "no-receipts",
        now=_ANCHOR,
    )
    fired = {(f.check, f.bead_id) for f in findings}
    assert ("F1", "polylogue-af1") in fired
    assert ("F2", "polylogue-af2") in fired
    assert ("F3", "polylogue-af3") in fired
    assert ("F4", "polylogue-af4") in fired
    # polylogue-okprog has a real (non-epic-leaf) admitted member, so it must
    # not also be flagged F4.
    assert not any(bid == "polylogue-okprog" for _check, bid in fired)


# --- compute_active_set_summary: soft-band diagnostics (never a Finding) ---


def test_active_set_summary_within_target_has_no_diagnostics(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [
            _issue(
                "polylogue-leaf1",
                acceptance_criteria="Verify: x.",
                labels=["area:ops"],
                metadata={"frontier": "active", "frontier_program_ref": "polylogue-prog"},
            ),
        ],
    )
    summary = verify_backlog_hygiene.compute_active_set_summary(path=path, target=30, warn=50)
    assert summary.active_leaf_count == 1
    assert summary.band == "within-target"
    assert summary.diagnostics == []


def test_active_set_summary_above_target_is_diagnostic_not_a_failure(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    issues = [
        _issue(
            f"polylogue-leaf{n}",
            title=f"Leaf {n}",
            acceptance_criteria="Verify: x.",
            labels=["area:ops"],
            metadata={"frontier": "active", "frontier_program_ref": "polylogue-prog"},
        )
        for n in range(5)
    ]
    # A real, validly-marked program so collect_findings() below stays clean
    # (F2 would otherwise fire on the dangling ref, which is not this test's
    # concern -- it's isolating the size-band diagnostic).
    issues.append(
        _issue(
            "polylogue-prog",
            title="Owning program",
            description="Owning program.",
            labels=["area:ops"],
            issue_type="task",
            metadata={"frontier_program": "active"},
        )
    )
    _write_jsonl(path, issues)
    summary = verify_backlog_hygiene.compute_active_set_summary(path=path, target=3, warn=10)
    assert summary.active_leaf_count == 5
    assert summary.band == "above-target"
    assert summary.diagnostics  # informational note present
    # Crucially: exceeding target never becomes a collect_findings() Finding.
    findings = verify_backlog_hygiene.collect_findings(
        path=path, allow_path=tmp_path / "no-allowlist.txt", receipts_path=tmp_path / "no-receipts"
    )
    assert findings == []


def test_active_set_summary_above_warn_band(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    issues = [
        _issue(
            f"polylogue-leaf{n}",
            acceptance_criteria="Verify: x.",
            labels=["area:ops"],
            metadata={"frontier": "active", "frontier_program_ref": "polylogue-prog"},
        )
        for n in range(5)
    ]
    _write_jsonl(path, issues)
    summary = verify_backlog_hygiene.compute_active_set_summary(path=path, target=1, warn=2)
    assert summary.active_leaf_count == 5
    assert summary.band == "above-warn"
    assert "soft warn threshold" in summary.diagnostics[0]


def test_active_set_summary_excludes_epic_leaves(tmp_path: Path) -> None:
    """An active epic is an F1 finding, not a legitimately-counted active leaf."""
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [
            _issue(
                "polylogue-epicleaf",
                description="Epic wrongly admitted as a leaf.",
                labels=["area:ops"],
                issue_type="epic",
                metadata={"frontier": "active", "frontier_program_ref": "polylogue-prog"},
            ),
        ],
    )
    summary = verify_backlog_hygiene.compute_active_set_summary(path=path)
    assert summary.active_leaf_count == 0


def test_main_json_includes_active_set_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(
        path,
        [
            _issue(
                "polylogue-clean1",
                title="Clean active leaf",
                acceptance_criteria="Verify: x.",
                labels=["area:ops"],
                metadata={"frontier": "active", "frontier_program_ref": "polylogue-prog"},
            ),
            _issue(
                "polylogue-prog",
                title="Owning program",
                description="Owning program.",
                labels=["area:ops"],
                issue_type="task",
                metadata={"frontier_program": "active"},
            ),
        ],
    )
    monkeypatch.setattr(verify_backlog_hygiene, "_get_root", lambda: tmp_path)

    exit_code = verify_backlog_hygiene.main(["--json", "--active-target", "10", str(path)])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["active_set"]["active_leaf_count"] == 1
    assert payload["active_set"]["target"] == 10
    assert payload["active_set"]["band"] == "within-target"
