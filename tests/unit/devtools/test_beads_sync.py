"""Tests for the monotonic, receipted Beads JSONL sync engine (polylogue-gxjh.1).

Anti-vacuity note: every test drives the production entrypoints
(`merge_issue_sets`, `synchronize_files`, `atomic_write_jsonl`,
`load_jsonl_rows`, `bootstrap_union`) in `devtools/beads_sync.py`, not a
test-local reimplementation. Deleting the strict-inequality guard in
`merge_issue_sets` (the `incoming_ts > base_ts` / `<` branches), removing the
`allow_recovery`/actor+reason requirement in `RecoveryRequiredError`, or
removing the conflict-marker scan in `load_jsonl_rows`/`atomic_write_jsonl`
each make a specific assertion below fail (see per-test comments).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import beads_sync as bs


def _row(issue_id: str, updated_at: str, **extra: object) -> dict[str, object]:
    return {"id": issue_id, "updated_at": updated_at, "title": f"title-{issue_id}", **extra}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


# --------------------------------------------------------------------------
# AC #1: stale/newer/equal/incomparable outcomes, ordinary sync never downgrades
# --------------------------------------------------------------------------


def test_merge_outcomes_cover_created_updated_equal_skipped_conflicted() -> None:
    base = {
        "a": _row("a", "2026-07-10T00:00:00Z", title="a-old"),
        "b": _row("b", "2026-07-10T00:00:00Z", title="b-same"),
        "c": _row("c", "2026-07-12T00:00:00Z", title="c-newer-in-base"),
        "d": _row("d", "2026-07-10T00:00:00Z", title="d-conflict-base"),
    }
    incoming = {
        "a": _row("a", "2026-07-11T00:00:00Z", title="a-new"),  # strictly newer -> updated
        "b": _row("b", "2026-07-10T00:00:00Z", title="b-same"),  # identical -> equal
        "c": _row("c", "2026-07-09T00:00:00Z", title="c-stale"),  # older -> skipped_downgrade
        "d": _row("d", "2026-07-10T00:00:00Z", title="d-conflict-incoming"),  # same ts, diff content
        "e": _row("e", "2026-07-10T00:00:00Z", title="e-brand-new"),  # only incoming -> created
    }

    merged, receipt = bs.merge_issue_sets(base, incoming)

    outcomes = {o.id: o.outcome for o in receipt.outcomes}
    assert outcomes == {
        "a": "updated",
        "b": "equal",
        "c": "skipped_downgrade",
        "d": "conflicted",
        "e": "created",
    }
    # Mutation check: if the `incoming_ts > base_ts` branch were changed to
    # `>=`, "b" (equal timestamps) would report "updated" instead of "equal".
    assert receipt.counts() == {"updated": 1, "equal": 1, "skipped_downgrade": 1, "conflicted": 1, "created": 1}

    # Ordinary sync result: newer/created rows adopted, stale/conflicted rows
    # keep the BASE version -- this is the core "cannot downgrade" guarantee.
    assert merged["a"]["title"] == "a-new"
    assert merged["c"]["title"] == "c-newer-in-base"  # NOT c-stale
    assert merged["d"]["title"] == "d-conflict-base"  # NOT d-conflict-incoming
    assert merged["e"]["title"] == "e-brand-new"

    # Both revisions are present on the machine-readable outcome record.
    skipped = receipt.by_outcome("skipped_downgrade")[0]
    assert skipped.base_updated_at == "2026-07-12T00:00:00Z"
    assert skipped.incoming_updated_at == "2026-07-09T00:00:00Z"


def test_ordinary_sync_cannot_downgrade_even_when_requested_without_recovery() -> None:
    base = {"x": _row("x", "2026-07-15T00:00:00Z", title="live-newer")}
    incoming = {"x": _row("x", "2026-07-01T00:00:00Z", title="stale-replacement")}

    merged, receipt = bs.merge_issue_sets(base, incoming)

    assert merged["x"]["title"] == "live-newer"
    assert receipt.has_refusals() is True
    assert receipt.has_downgrades() is False


# --------------------------------------------------------------------------
# AC #2: explicit recovery override requires actor+reason, records every
# downgraded row with fingerprint/actor/reason.
# --------------------------------------------------------------------------


def test_recovery_without_actor_or_reason_is_refused() -> None:
    base = {"x": _row("x", "2026-07-15T00:00:00Z")}
    incoming = {"x": _row("x", "2026-07-01T00:00:00Z")}

    with pytest.raises(bs.RecoveryRequiredError):
        bs.merge_issue_sets(base, incoming, allow_recovery=True)

    with pytest.raises(bs.RecoveryRequiredError):
        bs.merge_issue_sets(base, incoming, allow_recovery=True, actor="sinity")

    with pytest.raises(bs.RecoveryRequiredError):
        bs.merge_issue_sets(base, incoming, allow_recovery=True, reason="restore")


def test_recovery_override_applies_downgrade_and_records_full_provenance() -> None:
    base = {"x": _row("x", "2026-07-15T00:00:00Z", title="clobbered-bad-state")}
    incoming = {"x": _row("x", "2026-07-01T00:00:00Z", title="known-good-recovery")}
    fingerprint = {"project": "polylogue", "database": "dolt", "branch": "master", "source": "recovery.jsonl"}

    merged, receipt = bs.merge_issue_sets(
        base,
        incoming,
        fingerprint=fingerprint,
        allow_recovery=True,
        actor="sinity",
        reason="restore pre-clobber state",
    )

    assert merged["x"]["title"] == "known-good-recovery"
    assert receipt.recovery is True
    assert receipt.actor == "sinity"
    assert receipt.reason == "restore pre-clobber state"
    assert dict(receipt.fingerprint) == fingerprint
    downgraded = receipt.by_outcome("downgraded")
    assert len(downgraded) == 1
    assert downgraded[0].id == "x"
    assert downgraded[0].base_updated_at == "2026-07-15T00:00:00Z"
    assert downgraded[0].incoming_updated_at == "2026-07-01T00:00:00Z"
    # Mutation check: if RecoveryRequiredError's guard were deleted entirely
    # (recovery silently always-on), this test would still pass but
    # test_recovery_without_actor_or_reason_is_refused above would fail --
    # both must hold together to prove the guard is load-bearing.


# --------------------------------------------------------------------------
# AC #3: concurrent empty bootstraps + a writer yield one complete union,
# order-independent, no lost rows.
# --------------------------------------------------------------------------


def test_bootstrap_union_is_order_independent_and_loses_nothing() -> None:
    writer_a = {"1": _row("1", "2026-07-01T00:00:00Z", title="from-a")}
    writer_b = {"2": _row("2", "2026-07-02T00:00:00Z", title="from-b")}
    writer_c_update = {"1": _row("1", "2026-07-03T00:00:00Z", title="from-c-newer")}

    orderings = [
        (writer_a, writer_b, writer_c_update),
        (writer_b, writer_c_update, writer_a),
        (writer_c_update, writer_a, writer_b),
    ]
    results = [bs.bootstrap_union(*ordering) for ordering in orderings]

    for result in results:
        assert set(result) == {"1", "2"}
        assert result["1"]["title"] == "from-c-newer"  # newest updated_at wins regardless of arrival order
        assert result["2"]["title"] == "from-b"

    assert results[0] == results[1] == results[2]


def test_bootstrap_union_from_empty_base_keeps_every_row() -> None:
    result = bs.bootstrap_union({}, {"1": _row("1", "2026-07-01T00:00:00Z")}, {"2": _row("2", "2026-07-01T00:00:00Z")})
    assert set(result) == {"1", "2"}


def test_bootstrap_union_raises_on_true_incomparable_conflict() -> None:
    same_ts_different_content_a = {"1": _row("1", "2026-07-01T00:00:00Z", title="branch-a-edit")}
    same_ts_different_content_b = {"1": _row("1", "2026-07-01T00:00:00Z", title="branch-b-edit")}

    with pytest.raises(bs.PlanningSurfaceCorruptError):
        bs.bootstrap_union(same_ts_different_content_a, same_ts_different_content_b)


# --------------------------------------------------------------------------
# AC #4: atomic, validated export; refuses marker-bearing/incomparable state.
# --------------------------------------------------------------------------


def test_atomic_write_jsonl_produces_parseable_unique_id_file(tmp_path: Path) -> None:
    out = tmp_path / "issues.jsonl"
    rows = [_row("a", "2026-07-01T00:00:00Z"), _row("b", "2026-07-02T00:00:00Z")]

    bs.atomic_write_jsonl(out, rows)

    reloaded = bs.load_jsonl_rows(out)
    assert set(reloaded) == {"a", "b"}
    assert out.exists()
    # No leftover temp file.
    assert list(tmp_path.glob(".beads-sync-*")) == []


def test_atomic_write_jsonl_refuses_duplicate_ids(tmp_path: Path) -> None:
    out = tmp_path / "issues.jsonl"
    rows = [_row("a", "2026-07-01T00:00:00Z"), _row("a", "2026-07-02T00:00:00Z")]

    with pytest.raises(bs.PlanningSurfaceCorruptError):
        bs.atomic_write_jsonl(out, rows)
    assert not out.exists()  # nothing staged on failure


def test_atomic_write_jsonl_refuses_to_overwrite_marker_bearing_target(tmp_path: Path) -> None:
    out = tmp_path / "issues.jsonl"
    out.write_text('{"id": "a", "updated_at": "x"}\n<<<<<<< HEAD\n{"id": "b"}\n=======\n{"id": "c"}\n>>>>>>> branch\n')
    original_bytes = out.read_bytes()

    with pytest.raises(bs.PlanningSurfaceCorruptError):
        bs.atomic_write_jsonl(out, [_row("a", "2026-07-01T00:00:00Z")])

    # Target left byte-identical -- the refusal did not partially clobber it.
    assert out.read_bytes() == original_bytes


def test_load_jsonl_rows_refuses_conflict_markers_even_with_valid_json_lines(tmp_path: Path) -> None:
    # Reproduces the 2026-07-15 shape: valid-looking JSON lines around a
    # literal conflict marker, git reporting a plain modification.
    path = tmp_path / "issues.jsonl"
    path.write_text(
        '{"id": "a", "updated_at": "2026-07-01T00:00:00Z"}\n'
        "<<<<<<< Updated upstream\n"
        '{"id": "b", "updated_at": "2026-07-02T00:00:00Z"}\n'
        "=======\n"
        '{"id": "b", "updated_at": "2026-07-03T00:00:00Z"}\n'
        ">>>>>>> Stashed changes\n"
    )

    with pytest.raises(bs.PlanningSurfaceCorruptError, match="conflict marker"):
        bs.load_jsonl_rows(path)


def test_load_jsonl_rows_refuses_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    _write_jsonl(path, [_row("a", "2026-07-01T00:00:00Z"), _row("a", "2026-07-02T00:00:00Z")])

    with pytest.raises(bs.PlanningSurfaceCorruptError, match="duplicate id"):
        bs.load_jsonl_rows(path)


def test_synchronize_files_refuses_marker_bearing_incoming_file(tmp_path: Path) -> None:
    base = tmp_path / "base.jsonl"
    incoming = tmp_path / "incoming.jsonl"
    output = tmp_path / "out.jsonl"
    _write_jsonl(base, [_row("a", "2026-07-01T00:00:00Z")])
    incoming.write_text("<<<<<<< HEAD\n" + json.dumps(_row("a", "2026-07-02T00:00:00Z")) + "\n=======\n>>>>>>> x\n")

    with pytest.raises(bs.PlanningSurfaceCorruptError):
        bs.synchronize_files(base, incoming, output)

    assert not output.exists()  # refusal must not stage a bad merge


# --------------------------------------------------------------------------
# AC #5: replay the 2026-07-15 staged-conflict / nine-horizon-bead
# recovery scenario.
# --------------------------------------------------------------------------


def test_replay_2026_07_15_staged_conflict_recovery_preserves_repaired_beads(tmp_path: Path) -> None:
    """A valid stale whole-file replacement must fail; targeted merge must
    preserve every newer horizon-repaired row while still merging unrelated
    (non-conflicting) rows in from the incoming file.
    """
    # "base" = current live state: nine repaired horizon beads at a later
    # updated_at, plus one unrelated bead only base knows about.
    repaired_ids = [f"polylogue-repair-{i}" for i in range(9)]
    base_rows = {
        issue_id: _row(issue_id, "2026-07-15T19:00:00Z", labels=["horizon:frontier"]) for issue_id in repaired_ids
    }
    base_rows["polylogue-unrelated-base-only"] = _row(
        "polylogue-unrelated-base-only", "2026-07-14T00:00:00Z", title="untouched"
    )

    # "incoming" = the stale whole-file snapshot from before the repair,
    # reintroducing legacy horizon labels, PLUS one genuinely new unrelated
    # row that should still be adopted.
    incoming_rows = {
        issue_id: _row(issue_id, "2026-07-10T00:00:00Z", labels=["horizon:near"]) for issue_id in repaired_ids
    }
    incoming_rows["polylogue-new-from-incoming"] = _row(
        "polylogue-new-from-incoming", "2026-07-16T00:00:00Z", title="genuinely new"
    )

    base_path = tmp_path / "base.jsonl"
    incoming_path = tmp_path / "incoming-stale-snapshot.jsonl"
    output_path = tmp_path / "merged.jsonl"
    _write_jsonl(base_path, list(base_rows.values()))
    _write_jsonl(incoming_path, list(incoming_rows.values()))

    # A naive "valid stale whole-file replacement" (plain overwrite) is
    # exactly the bug: it would silently accept the incoming file as-is.
    # The regression is that `synchronize_files` must NOT do that --
    # ordinary sync must refuse the downgrades and only adopt the safe rows.
    receipt = bs.synchronize_files(base_path, incoming_path, output_path)

    merged = bs.load_jsonl_rows(output_path)

    # All nine repaired beads keep their newer (repaired) label state.
    for issue_id in repaired_ids:
        assert merged[issue_id]["labels"] == ["horizon:frontier"], issue_id
    # Unrelated rows from both sides are present.
    assert merged["polylogue-unrelated-base-only"]["title"] == "untouched"
    assert merged["polylogue-new-from-incoming"]["title"] == "genuinely new"

    # The receipt proves every one of the nine was a reported/refused
    # downgrade, not a silent success.
    downgrade_ids = {o.id for o in receipt.by_outcome("skipped_downgrade")}
    assert downgrade_ids == set(repaired_ids)
    assert receipt.has_downgrades() is False  # nothing was actually applied without recovery
    created_ids = {o.id for o in receipt.by_outcome("created")}
    assert "polylogue-new-from-incoming" in created_ids


def test_replay_recovery_path_can_still_restore_a_deliberately_older_row(tmp_path: Path) -> None:
    """The flip side of the incident: sometimes the recovery FILE is the
    known-good state and current live state is the clobbered one. An
    operator-authorized recovery merge must apply that with full receipt
    provenance -- this is what distinguishes authorized recovery from the
    accidental whole-file overwrite bug.
    """
    base_path = tmp_path / "clobbered-live.jsonl"
    incoming_path = tmp_path / "known-good-recovery.jsonl"
    output_path = tmp_path / "restored.jsonl"

    _write_jsonl(base_path, [_row("polylogue-x", "2026-07-15T20:00:00Z", title="accidentally-reverted")])
    _write_jsonl(incoming_path, [_row("polylogue-x", "2026-07-14T00:00:00Z", title="actually-correct-older-edit")])

    receipt = bs.synchronize_files(
        base_path,
        incoming_path,
        output_path,
        allow_recovery=True,
        actor="sinity",
        reason="restore known-good pre-clobber row",
    )

    merged = bs.load_jsonl_rows(output_path)
    assert merged["polylogue-x"]["title"] == "actually-correct-older-edit"
    assert receipt.has_downgrades() is True
    assert receipt.actor == "sinity"


# --------------------------------------------------------------------------
# Receipt shape / CLI smoke
# --------------------------------------------------------------------------


def test_receipt_to_json_round_trips_every_outcome_field() -> None:
    base = {"a": _row("a", "2026-07-01T00:00:00Z")}
    incoming = {"a": _row("a", "2026-07-02T00:00:00Z")}
    _merged, receipt = bs.merge_issue_sets(base, incoming, fingerprint={"branch": "master"})

    payload = receipt.to_json()
    assert payload["fingerprint"] == {"branch": "master"}
    assert payload["recovery"] is False
    assert payload["counts"] == {"updated": 1}
    assert payload["outcomes"] == [
        {
            "id": "a",
            "outcome": "updated",
            "base_updated_at": "2026-07-01T00:00:00Z",
            "incoming_updated_at": "2026-07-02T00:00:00Z",
        }
    ]


def test_covers_detects_non_progress_receipt() -> None:
    base = {"a": _row("a", "2026-07-01T00:00:00Z")}
    incoming = {"a": _row("a", "2026-07-02T00:00:00Z")}
    _merged, receipt = bs.merge_issue_sets(base, incoming)

    assert receipt.covers(["a"]) is True
    assert receipt.covers(["a", "missing-id"]) is False


def test_cli_merge_exits_nonzero_on_unresolved_refusal_without_recover(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base_path = tmp_path / "base.jsonl"
    incoming_path = tmp_path / "incoming.jsonl"
    output_path = tmp_path / "out.jsonl"
    _write_jsonl(base_path, [_row("x", "2026-07-15T00:00:00Z")])
    _write_jsonl(incoming_path, [_row("x", "2026-07-01T00:00:00Z")])

    rc = bs.main(
        ["merge", "--base", str(base_path), "--incoming", str(incoming_path), "--output", str(output_path), "--json"]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"] == {"skipped_downgrade": 1}


def test_cli_merge_exits_zero_when_fully_resolved(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    incoming_path = tmp_path / "incoming.jsonl"
    output_path = tmp_path / "out.jsonl"
    _write_jsonl(base_path, [_row("x", "2026-07-01T00:00:00Z")])
    _write_jsonl(incoming_path, [_row("x", "2026-07-02T00:00:00Z")])

    rc = bs.main(["merge", "--base", str(base_path), "--incoming", str(incoming_path), "--output", str(output_path)])

    assert rc == 0
    assert bs.load_jsonl_rows(output_path)["x"]["updated_at"] == "2026-07-02T00:00:00Z"


def test_cli_merge_recovery_requires_actor_and_reason_flags(tmp_path: Path) -> None:
    base_path = tmp_path / "base.jsonl"
    incoming_path = tmp_path / "incoming.jsonl"
    output_path = tmp_path / "out.jsonl"
    _write_jsonl(base_path, [_row("x", "2026-07-15T00:00:00Z")])
    _write_jsonl(incoming_path, [_row("x", "2026-07-01T00:00:00Z")])

    rc = bs.main(
        ["merge", "--base", str(base_path), "--incoming", str(incoming_path), "--output", str(output_path), "--recover"]
    )

    assert rc == 1
    assert not output_path.exists()
