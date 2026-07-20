from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import verify_backlog_hygiene


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
