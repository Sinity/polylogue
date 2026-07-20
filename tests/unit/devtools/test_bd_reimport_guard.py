from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from devtools import bd_reimport_guard as guard


@pytest.fixture(autouse=True)
def _isolated_snapshot_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect snapshot/receipt files under tmp_path instead of real paths.

    Prevents test runs from colliding with each other or with a real
    in-flight git hook using the same hook-name snapshot file, and prevents
    tests from writing receipts into the real repo's .cache/ directory.
    """
    monkeypatch.setattr(guard, "_snapshot_path", lambda hook_name: tmp_path / f"snapshot-{hook_name}.jsonl")
    monkeypatch.setattr(guard, "_receipts_dir", lambda: tmp_path / "receipts")


def test_snapshot_writes_live_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    state = {"polylogue-a": {"id": "polylogue-a", "updated_at": "2026-07-01T00:00:00Z"}}
    monkeypatch.setattr(guard, "_export_live_state", lambda: state)

    rc = guard.cmd_snapshot("post-checkout")

    assert rc == 0
    snap_path = guard._snapshot_path("post-checkout")
    assert json.loads(snap_path.read_text()) == state


def test_check_and_repair_is_noop_without_prior_snapshot() -> None:
    # No snapshot file exists for this hook name -- nothing to compare.
    rc = guard.cmd_check_and_repair("never-snapshotted")
    assert rc == 0


def test_check_and_repair_restores_clobbered_bead_and_leaves_unchanged_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = {
        "polylogue-fresh": {"id": "polylogue-fresh", "updated_at": "2026-07-19T23:00:00Z", "title": "closed live"},
        "polylogue-stable": {"id": "polylogue-stable", "updated_at": "2026-07-18T10:00:00Z", "title": "unchanged"},
    }
    after = {
        # reimport clobbered this back to an older committed state
        "polylogue-fresh": {"id": "polylogue-fresh", "updated_at": "2026-07-10T08:00:00Z", "title": "closed live"},
        "polylogue-stable": {"id": "polylogue-stable", "updated_at": "2026-07-18T10:00:00Z", "title": "unchanged"},
    }
    states = iter([after])
    monkeypatch.setattr(guard, "_export_live_state", lambda: next(states))

    snap_path = guard._snapshot_path("post-checkout")
    snap_path.write_text(json.dumps(before))

    recorded: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
        recorded.append(cmd)
        if cmd[:2] == ["bd", "import"]:
            restored_ids = [json.loads(line)["id"] for line in Path(cmd[2]).read_text().splitlines() if line.strip()]
            assert restored_ids == ["polylogue-fresh"], "only the clobbered bead should be restored"
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = guard.cmd_check_and_repair("post-checkout")

    assert rc == 0
    assert not snap_path.exists(), "snapshot must be consumed"
    assert any(cmd[:2] == ["bd", "import"] for cmd in recorded)
    assert any(cmd[:2] == ["bd", "export"] for cmd in recorded)


def test_check_and_repair_restores_vanished_bead(monkeypatch: pytest.MonkeyPatch) -> None:
    before = {"polylogue-gone": {"id": "polylogue-gone", "updated_at": "2026-07-19T23:00:00Z"}}
    after: dict[str, dict[str, Any]] = {}
    monkeypatch.setattr(guard, "_export_live_state", lambda: after)

    snap_path = guard._snapshot_path("post-merge")
    snap_path.write_text(json.dumps(before))

    recorded: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
        recorded.append(cmd)
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = guard.cmd_check_and_repair("post-merge")

    assert rc == 0
    assert any(cmd[:2] == ["bd", "import"] for cmd in recorded)


def test_check_and_repair_noop_when_nothing_clobbered(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"polylogue-stable": {"id": "polylogue-stable", "updated_at": "2026-07-18T10:00:00Z"}}
    monkeypatch.setattr(guard, "_export_live_state", lambda: state)

    snap_path = guard._snapshot_path("post-checkout")
    snap_path.write_text(json.dumps(state))

    recorded: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
        recorded.append(cmd)
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = guard.cmd_check_and_repair("post-checkout")

    assert rc == 0
    assert recorded == [], "no bd import/export calls should fire when nothing was clobbered"


def test_main_requires_command_and_hook_name() -> None:
    assert guard.main([]) == 1
    assert guard.main(["snapshot"]) == 1


def test_main_rejects_unknown_command() -> None:
    assert guard.main(["frobnicate", "post-checkout"]) == 1


def test_main_dispatches_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_cmd_snapshot(hook_name: str) -> int:
        calls.append(hook_name)
        return 0

    monkeypatch.setattr(guard, "cmd_snapshot", fake_cmd_snapshot)
    rc = guard.main(["snapshot", "post-checkout"])
    assert rc == 0
    assert calls == ["post-checkout"]


# --- merge_rows: the monotonic per-row merge engine --------------------------


def test_merge_rows_classifies_new_updated_equal_and_skips_downgrade() -> None:
    current = {
        "polylogue-a": {"id": "polylogue-a", "updated_at": "2026-07-15T10:00:00Z", "title": "a-current"},
        "polylogue-b": {"id": "polylogue-b", "updated_at": "2026-07-15T10:00:00Z", "title": "b-current"},
        "polylogue-c": {"id": "polylogue-c", "updated_at": "2026-07-15T10:00:00Z", "title": "c-current"},
    }
    candidate = {
        # newer than current -- should win
        "polylogue-a": {"id": "polylogue-a", "updated_at": "2026-07-16T10:00:00Z", "title": "a-newer"},
        # same revision -- no-op
        "polylogue-b": {"id": "polylogue-b", "updated_at": "2026-07-15T10:00:00Z", "title": "b-current"},
        # older than current -- a downgrade, must be refused by default
        "polylogue-c": {"id": "polylogue-c", "updated_at": "2026-07-14T10:00:00Z", "title": "c-stale"},
        # not present in current at all -- a new row
        "polylogue-d": {"id": "polylogue-d", "updated_at": "2026-07-16T10:00:00Z", "title": "d-new"},
    }

    merged, outcomes = guard.merge_rows(current, candidate)

    by_id = {o.issue_id: o for o in outcomes}
    assert by_id["polylogue-a"].outcome == "updated"
    assert by_id["polylogue-b"].outcome == "equal"
    assert by_id["polylogue-c"].outcome == "skipped_downgrade"
    assert by_id["polylogue-d"].outcome == "new"

    # The mutation that would make this fail: merging the stale candidate
    # for polylogue-c into `merged` (i.e. treating merge as a blind upsert,
    # as bd's own reimport does). Asserting the title stayed "c-current"
    # is exactly the anti-regression property this bead exists to enforce.
    assert merged["polylogue-c"]["title"] == "c-current"
    assert merged["polylogue-a"]["title"] == "a-newer"
    assert merged["polylogue-d"]["title"] == "d-new"


def test_merge_rows_incomparable_revision_is_conflicted_not_guessed() -> None:
    current = {"polylogue-a": {"id": "polylogue-a", "title": "no revision on current"}}
    candidate = {"polylogue-a": {"id": "polylogue-a", "updated_at": "2026-07-16T10:00:00Z", "title": "candidate"}}

    merged, outcomes = guard.merge_rows(current, candidate)

    assert outcomes[0].outcome == "conflicted"
    # Neither side is trusted as a winner -- current state is left alone.
    assert merged["polylogue-a"]["title"] == "no revision on current"


def test_merge_rows_allow_downgrade_records_explicit_recovery() -> None:
    current = {"polylogue-c": {"id": "polylogue-c", "updated_at": "2026-07-15T10:00:00Z", "title": "c-current"}}
    candidate = {"polylogue-c": {"id": "polylogue-c", "updated_at": "2026-07-14T10:00:00Z", "title": "c-stale"}}

    merged, outcomes = guard.merge_rows(current, candidate, allow_downgrade=True)

    assert outcomes[0].outcome == "recovered_downgrade"
    assert merged["polylogue-c"]["title"] == "c-stale"


# --- parse_and_validate_jsonl / atomic_write_jsonl ----------------------------


def test_parse_and_validate_jsonl_accepts_clean_payload() -> None:
    text = '{"id": "polylogue-a", "updated_at": "2026-07-15T10:00:00Z"}\n{"id": "polylogue-b"}\n'
    rows = guard.parse_and_validate_jsonl(text)
    assert set(rows) == {"polylogue-a", "polylogue-b"}


def test_parse_and_validate_jsonl_rejects_conflict_markers() -> None:
    text = '{"id": "polylogue-a"}\n<<<<<<< Updated upstream\n{"id": "polylogue-b"}\n=======\n>>>>>>> Stashed changes\n'
    with pytest.raises(guard.InvalidJsonlError, match="conflict"):
        guard.parse_and_validate_jsonl(text)


def test_parse_and_validate_jsonl_rejects_invalid_json_line() -> None:
    with pytest.raises(guard.InvalidJsonlError, match="invalid JSON"):
        guard.parse_and_validate_jsonl('{"id": "polylogue-a"}\nnot json\n')


def test_parse_and_validate_jsonl_rejects_missing_id() -> None:
    with pytest.raises(guard.InvalidJsonlError, match="missing"):
        guard.parse_and_validate_jsonl('{"title": "no id here"}\n')


def test_parse_and_validate_jsonl_rejects_duplicate_id() -> None:
    text = '{"id": "polylogue-a", "title": "first"}\n{"id": "polylogue-a", "title": "second"}\n'
    with pytest.raises(guard.InvalidJsonlError, match="duplicate"):
        guard.parse_and_validate_jsonl(text)


def test_atomic_write_jsonl_writes_valid_rows(tmp_path: Path) -> None:
    target = tmp_path / "issues.jsonl"
    rows = {"polylogue-a": {"id": "polylogue-a", "updated_at": "2026-07-15T10:00:00Z"}}

    guard.atomic_write_jsonl(target, rows)

    written = guard.parse_and_validate_jsonl(target.read_text())
    assert written == rows
    # No leaked .tmp sibling files.
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_jsonl_refuses_marker_bearing_target(tmp_path: Path) -> None:
    target = tmp_path / "issues.jsonl"
    target.write_text('{"id": "polylogue-a"}\n<<<<<<< HEAD\n{"id": "polylogue-b"}\n=======\n>>>>>>> branch\n')

    with pytest.raises(guard.InvalidJsonlError, match="conflict markers"):
        guard.atomic_write_jsonl(target, {"polylogue-c": {"id": "polylogue-c", "updated_at": "2026-07-15T10:00:00Z"}})

    # Refusal must not touch the existing (corrupt) file.
    assert "<<<<<<<" in target.read_text()


# --- cmd_reconcile: the explicit operator-invokable sync path ----------------


def test_reconcile_applies_new_and_updated_but_refuses_downgrade(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    live = {
        "polylogue-a": {"id": "polylogue-a", "updated_at": "2026-07-15T10:00:00Z", "title": "a-current"},
        "polylogue-c": {"id": "polylogue-c", "updated_at": "2026-07-15T10:00:00Z", "title": "c-current"},
    }
    monkeypatch.setattr(guard, "_export_live_state", lambda: live)

    candidate_path = tmp_path / "candidate.jsonl"
    candidate_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"id": "polylogue-a", "updated_at": "2026-07-16T10:00:00Z", "title": "a-newer"},
                {"id": "polylogue-c", "updated_at": "2026-07-01T00:00:00Z", "title": "c-stale-replacement"},
                {"id": "polylogue-d", "updated_at": "2026-07-16T10:00:00Z", "title": "d-new"},
            )
        )
        + "\n"
    )

    imported: list[str] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
        if cmd[:2] == ["bd", "import"]:
            imported.extend(json.loads(line)["id"] for line in Path(cmd[2]).read_text().splitlines() if line.strip())
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = guard.cmd_reconcile([str(candidate_path)])

    # A skipped downgrade means the sync is not "clean" -- rc reports it
    # rather than silently succeeding.
    assert rc == 1
    assert set(imported) == {"polylogue-a", "polylogue-d"}
    assert "polylogue-c" not in imported, "the stale replacement for polylogue-c must never reach bd import"

    receipts = list((tmp_path / "receipts").glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["summary"]["skipped_downgrade"] == 1
    assert receipt["summary"]["updated"] == 1
    assert receipt["summary"]["new"] == 1
    assert not receipt["is_clean"]


def test_reconcile_allow_downgrade_requires_actor_and_reason(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.jsonl"
    candidate_path.write_text('{"id": "polylogue-a", "updated_at": "2026-07-01T00:00:00Z"}\n')

    rc = guard.cmd_reconcile([str(candidate_path), "--allow-downgrade"])

    assert rc == 1


def test_reconcile_allow_downgrade_with_actor_reason_applies_and_records_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    live = {"polylogue-c": {"id": "polylogue-c", "updated_at": "2026-07-15T10:00:00Z", "title": "c-current"}}
    monkeypatch.setattr(guard, "_export_live_state", lambda: live)

    candidate_path = tmp_path / "candidate.jsonl"
    candidate_path.write_text(
        json.dumps({"id": "polylogue-c", "updated_at": "2026-07-01T00:00:00Z", "title": "c-recovered"}) + "\n"
    )

    imported: list[str] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
        if cmd[:2] == ["bd", "import"]:
            imported.extend(json.loads(line)["id"] for line in Path(cmd[2]).read_text().splitlines() if line.strip())
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = guard.cmd_reconcile(
        [
            str(candidate_path),
            "--allow-downgrade",
            "--actor",
            "sinity",
            "--reason",
            "corrupted live db, restoring known-good export",
        ]
    )

    assert rc == 0
    assert imported == ["polylogue-c"]

    receipts = list((tmp_path / "receipts").glob("*.json"))
    receipt = json.loads(receipts[0].read_text())
    assert receipt["recovery"] is True
    assert receipt["actor"] == "sinity"
    assert receipt["summary"]["recovered_downgrade"] == 1


def test_reconcile_rejects_marker_bearing_candidate_file(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.jsonl"
    candidate_path.write_text('{"id": "polylogue-a"}\n<<<<<<< HEAD\n=======\n>>>>>>> branch\n')

    rc = guard.cmd_reconcile([str(candidate_path)])

    assert rc == 2
    receipts = list((tmp_path / "receipts").glob("*.json"))
    assert len(receipts) == 1


def test_reconcile_missing_candidate_file_errors_without_writing_receipt(tmp_path: Path) -> None:
    rc = guard.cmd_reconcile([str(tmp_path / "does-not-exist.jsonl")])
    assert rc == 1
    assert not (tmp_path / "receipts").exists()


# --- cmd_export ----------------------------------------------------------------


def test_export_writes_atomically_and_refuses_marker_bearing_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    live = {"polylogue-a": {"id": "polylogue-a", "updated_at": "2026-07-15T10:00:00Z", "title": "a"}}
    monkeypatch.setattr(guard, "_export_live_state", lambda: live)

    target = tmp_path / "out.jsonl"
    assert guard.cmd_export([str(target)]) == 0
    assert guard.parse_and_validate_jsonl(target.read_text()) == live

    target.write_text('{"id": "polylogue-a"}\n<<<<<<< HEAD\n=======\n>>>>>>> branch\n')
    assert guard.cmd_export([str(target)]) == 1
    assert "<<<<<<<" in target.read_text(), "refused overwrite must not touch the corrupt target"


# --- replay of the 2026-07-15 staged-conflict recovery incident --------------


def test_replay_2026_07_15_incident_preserves_nine_repaired_beads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces the incident this bead exists to prevent.

    A staged issues.jsonl carried conflict markers; a later `bd export`
    regenerated a syntactically valid file that silently restored nine
    independently-repaired Beads to stale revisions (horizon-label
    repairs), while whole-file JSON validity and both portfolio lints
    passed. The production mutation that reproduces the bug: replacing
    `merge_rows`'s per-row comparison with an unconditional
    `merged.update(candidate)` (a blind upsert) -- that change makes this
    test fail by letting every one of the nine stale rows win.
    """
    repaired_ids = [f"polylogue-repaired-{i}" for i in range(9)]
    live = {
        issue_id: {"id": issue_id, "updated_at": "2026-07-15T18:00:00Z", "labels": ["horizon:frontier"]}
        for issue_id in repaired_ids
    }
    live["polylogue-unrelated"] = {
        "id": "polylogue-unrelated",
        "updated_at": "2026-07-15T12:00:00Z",
        "labels": ["area:ops"],
    }
    monkeypatch.setattr(guard, "_export_live_state", lambda: live)

    # A valid, but stale, whole-file replacement: the pre-repair state of
    # the nine beads (older revision, legacy horizon labels), plus one
    # genuinely newer unrelated row that should still be accepted.
    stale_rows = [
        {"id": issue_id, "updated_at": "2026-07-15T09:00:00Z", "labels": ["horizon:near"]} for issue_id in repaired_ids
    ]
    stale_rows.append(
        {"id": "polylogue-unrelated", "updated_at": "2026-07-15T19:00:00Z", "labels": ["area:ops", "area:beads"]}
    )
    candidate_path = tmp_path / "stale-whole-file.jsonl"
    candidate_path.write_text("\n".join(json.dumps(row) for row in stale_rows) + "\n")

    imported: list[dict[str, Any]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
        if cmd[:2] == ["bd", "import"]:
            imported.extend(json.loads(line) for line in Path(cmd[2]).read_text().splitlines() if line.strip())
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = guard.cmd_reconcile([str(candidate_path)])

    # The valid-but-stale whole-file replacement must fail ordinary sync.
    assert rc == 1

    imported_ids = {row["id"] for row in imported}
    assert imported_ids == {"polylogue-unrelated"}, "only the genuinely newer unrelated row may be applied"

    receipts = list((tmp_path / "receipts").glob("*.json"))
    receipt = json.loads(receipts[0].read_text())
    assert receipt["summary"]["skipped_downgrade"] == 9
    downgraded = {o["id"] for o in receipt["outcomes"] if o["outcome"] == "skipped_downgrade"}
    assert downgraded == set(repaired_ids)
