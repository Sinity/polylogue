from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.cli.command_inventory import iter_command_paths
from polylogue.core.enums import Provider
from polylogue.storage.blob_gc import read_gc_history
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore, PreparedBlob
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveSessionSearchHit, ArchiveSessionSummary, ArchiveStore
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session_blob_ref
from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion
from tests.infra.daemon_operations import cli_daemon_archive
from tests.infra.live_ingest import write_index_session


def _seed_raw_authority_blocker(
    archive_root: Path,
    *,
    blocker_id: str,
    plan_id: str = "raw-replay:cli-test-plan",
    observed_pass_id: str = "raw-authority-frontier-pass:cli-test",
    frontier: bool = False,
    reason: str = "immutable source/index preconditions changed after the inspection pass",
) -> None:
    """Directly seed one real, unresolved ``raw_authority_blockers`` row.

    Mirrors the production shape (blocker row plus one real ``raw_sessions``
    row so a non-frontier resolution can genuinely replan it) with hand-built
    minimal rows rather than driving a full frontier inspection -- sufficient
    to exercise ``BlockerResolveActuator.prepare``'s real read against
    ``source.db`` and, for non-frontier blockers,
    the prepared acknowledgement’s original-input replan.

    ``frontier`` seeds a ``frontier_obligation``-kind blocker: the current
    frontier plan shape the prepared inspector writes. Without
    it the row is a ``stale_plan`` -- a durable snapshot predating that shape,
    which the resolver re-derives from live evidence instead of trusting.
    """
    import asyncio

    from tests.infra.archive_templates import run_archive_fixture_write

    def seed() -> None:
        raw_id = f"raw-{blocker_id}"
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            payload = (
                b'{"type":"session_meta","payload":{"id":"' + blocker_id.encode() + b'"}}\n'
                b'{"type":"response_item","payload":{"type":"message","id":"m-1",'
                b'"role":"user","content":[{"type":"input_text","text":"hi"}]}}\n'
            )
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path=f"{blocker_id}.jsonl",
                canonical_source_path=f"{blocker_id}.jsonl",
                acquired_at_ms=1000,
                raw_id=raw_id,
            )

        witness_schema = "polylogue.raw-authority-frontier-plan.v1" if frontier else "polylogue.raw-authority-plan.v1"
        input_digest = hashlib.sha256(plan_id.encode("utf-8")).hexdigest()
        observed_json = "{}"
        with sqlite3.connect(archive_root / "source.db") as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            # The blocker is keyed on the plan's content address and carries the
            # plan snapshot itself: that snapshot, not a join into a plan ledger,
            # is what every reader resolves against.
            conn.execute(
                """
                INSERT INTO raw_authority_blockers (
                    blocker_id, plan_input_digest, observed_pass_id, reason, expected_json,
                    observed_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, 1000)
                """,
                (
                    blocker_id,
                    input_digest,
                    observed_pass_id,
                    reason,
                    json.dumps(
                        {
                            "plan_id": plan_id,
                            "input_digest": input_digest,
                            "input_raw_ids": [raw_id],
                            "logical_keys": [],
                            "authority_witness": {"schema": witness_schema},
                            "source_preconditions": {},
                            "index_preconditions": {},
                        }
                    ),
                    observed_json,
                ),
            )
            conn.commit()

    asyncio.run(run_archive_fixture_write(archive_root, seed))


def test_raw_authority_blocker_resolution_cli_requires_confirmation(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) daemon route: the confirmed half is a write the daemon owns.

    `raw-authority-blocker-resolve --yes` lowers to
    mutation.raw-authority-blocker.resolve, so the accepted invocation needs a
    resident daemon. The refused invocation is unchanged: it is rejected by the
    CLI before anything is dispatched.
    """
    root = cli_workspace["archive_root"]
    _seed_raw_authority_blocker(root, blocker_id="blocker-1")
    base = [
        "--plain",
        "ops",
        "maintenance",
        "raw-authority-blocker-resolve",
        "--blocker-id",
        "blocker-1",
        "--reason",
        "reviewed current evidence",
    ]
    with cli_daemon_archive(root, monkeypatch):
        refused = cli_runner.invoke(cli, base)
        accepted = cli_runner.invoke(cli, [*base, "--yes"], catch_exceptions=False)

    assert refused.exit_code != 0
    assert "without --yes" in refused.output
    assert accepted.exit_code == 0
    assert "Resolved blocker-1" in accepted.output
    with sqlite3.connect(root / "source.db") as conn:
        row = conn.execute(
            "SELECT resolved_at_ms, resolution FROM raw_authority_blockers WHERE blocker_id = ?", ("blocker-1",)
        ).fetchone()
    assert row is not None
    assert row[0] is not None
    assert "reviewed current evidence" in str(row[1])


def test_raw_authority_blocker_resolution_cli_refuses_unknown_blocker(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) daemon route. Anti-vacuity: an unknown blocker id must not cause a durable mutation.

    The "not found or already resolved" judgment belongs to the daemon handler
    that owns the resolution, so the daemon has to be present for this to be a
    test of that judgment rather than of the daemon's absence.

    The judgment survives the move into the daemon: BlockerResolveActuator.apply
    returns status ``already_satisfied`` with a zero affected count, and the
    envelope ``_execute_named_mutation`` returns carries that ``affected_count``
    through to the command. The JSON route therefore exits non-zero for a no-op
    as well, so a scripted caller cannot read "nothing was resolved" as success.

    Anti-vacuity: rendering the receipt without consulting ``affected_count``
    restores exit 0 here and turns this red.
    """
    with cli_daemon_archive(cli_workspace["archive_root"], monkeypatch):
        result = cli_runner.invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "raw-authority-blocker-resolve",
                "--blocker-id",
                "does-not-exist",
                "--reason",
                "reviewed current evidence",
                "--yes",
                "--output-format",
                "json",
            ],
        )
    assert result.exit_code == 1, result.output
    receipt = json.loads(result.stdout)
    # Nothing was resolved, so there is no durable receipt reference to show.
    assert not receipt.get("receipt_ref")
    assert not receipt.get("resolved_at_ms")
    # The load-bearing invariant: a typo'd id causes no durable mutation.
    with sqlite3.connect(cli_workspace["archive_root"] / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM raw_authority_blockers").fetchone() == (0,)


def test_raw_authority_blocker_resolution_plain_output_reports_unknown_blocker(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reporting half of the refusal above: a no-op never renders as success.

    The daemon receipt says ``already_satisfied`` /
    ``blocker_not_found_or_already_resolved`` with ``affected_count`` 0. The
    plain path consults that count and refuses, restoring the "not found or
    already resolved" message that survived nowhere in the product.

    Anti-vacuity: printing ``Resolved <id>`` unconditionally -- the behaviour
    this replaces -- exits 0 and turns both assertions red.
    """
    with cli_daemon_archive(cli_workspace["archive_root"], monkeypatch):
        result = cli_runner.invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "raw-authority-blocker-resolve",
                "--blocker-id",
                "does-not-exist",
                "--reason",
                "reviewed current evidence",
                "--yes",
            ],
        )
    assert result.exit_code != 0
    assert "Resolved does-not-exist" not in result.output
    assert "not found or already resolved" in result.output


@pytest.mark.parametrize("resolution", ["acknowledged", "  acknowledged  "])
def test_raw_authority_blockers_cli_lists_unresolved_and_classifies_kind(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    resolution: str,
) -> None:
    """(b) daemon route: the listing is a read, but the resolution in its tail is a write."""
    root = cli_workspace["archive_root"]
    _seed_raw_authority_blocker(root, blocker_id="blocker-stale", plan_id="raw-replay:stale-plan")
    _seed_raw_authority_blocker(
        root,
        blocker_id="blocker-frontier",
        plan_id="raw-replay:frontier-plan",
        observed_pass_id="raw-authority-frontier-pass:frontier-test",
        frontier=True,
        reason="accepted raw authority remains quarantined",
    )
    _seed_raw_authority_blocker(
        root,
        blocker_id="blocker-obligation",
        plan_id="raw-replay:obligation-plan",
        observed_pass_id="raw-authority-frontier-pass:obligation-test",
        frontier=True,
        reason="missing bytes require reacquisition",
    )

    with cli_daemon_archive(root, monkeypatch):
        result = cli_runner.invoke(
            cli,
            ["--plain", "ops", "maintenance", "raw-authority-blockers", "--output-format", "json"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        by_id = {row["blocker_id"]: row for row in payload["blockers"]}
        assert by_id["blocker-stale"]["kind"] == "stale_plan"
        assert by_id["blocker-frontier"]["kind"] == "frontier_obligation"
        assert by_id["blocker-obligation"]["kind"] == "frontier_obligation"
        assert isinstance(by_id["blocker-stale"]["expected"], dict)
        assert isinstance(by_id["blocker-stale"]["observed"], dict)
        assert payload["total_count"] == 3
        assert payload["truncated"] is False
        # Resolving one blocker removes it from the unresolved listing.
        resolve = cli_runner.invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "raw-authority-blocker-resolve",
                "--blocker-id",
                "blocker-stale",
                "--reason",
                resolution,
                "--yes",
            ],
            catch_exceptions=False,
        )
        assert resolve.exit_code == 0, resolve.output
        after = cli_runner.invoke(
            cli,
            ["--plain", "ops", "maintenance", "raw-authority-blockers", "--output-format", "json"],
            catch_exceptions=False,
        )

    after_payload = json.loads(after.stdout)
    remaining = {row["blocker_id"] for row in after_payload["blockers"]}
    assert remaining == {"blocker-frontier", "blocker-obligation"}
    assert after_payload["total_count"] == 2


def test_raw_authority_blockers_cli_paginates_past_the_limit(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    """Anti-vacuity for the Codex-flagged hard cap (PR #3258): --limit below
    the total unresolved count must still expose every blocker via
    --offset, with truncated/next_offset telling the operator to page."""
    root = cli_workspace["archive_root"]
    for index in range(3):
        _seed_raw_authority_blocker(
            root,
            blocker_id=f"blocker-page-{index}",
            plan_id=f"raw-replay:page-plan-{index}",
            observed_pass_id=f"raw-authority-frontier-pass:page-{index}",
        )

    first = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "raw-authority-blockers",
            "--limit",
            "2",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    first_payload = json.loads(first.stdout)
    assert first_payload["returned_count"] == 2
    assert first_payload["total_count"] == 3
    assert first_payload["truncated"] is True
    next_offset = first_payload["next_offset"]
    assert next_offset == 2

    second = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "raw-authority-blockers",
            "--limit",
            "2",
            "--offset",
            str(next_offset),
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )
    second_payload = json.loads(second.stdout)
    assert second_payload["returned_count"] == 1
    assert second_payload["truncated"] is False

    seen_ids = {row["blocker_id"] for row in first_payload["blockers"]} | {
        row["blocker_id"] for row in second_payload["blockers"]
    }
    assert seen_ids == {"blocker-page-0", "blocker-page-1", "blocker-page-2"}

    # Plain-mode output surfaces the truncation notice too.
    plain_first = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "raw-authority-blockers", "--limit", "2"],
        catch_exceptions=False,
    )
    assert "Truncated: pass --offset 2" in plain_first.output


def _write_gc_candidate(cli_workspace: dict[str, Path], blob_hash: str) -> Path:
    blob_root = cli_workspace["archive_root"] / "blob"
    path = blob_root / blob_hash[:2] / blob_hash[2:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"gc candidate")
    old_epoch_s = 946684800
    os.utime(path, (old_epoch_s, old_epoch_s))
    return path


def _seed_assertion_export_rows(archive_root: Path) -> None:
    with sqlite3.connect(archive_root / "user.db") as conn:
        conn.row_factory = sqlite3.Row
        upsert_assertion(
            conn,
            assertion_id="export-mark",
            target_ref="session:s-1",
            kind=AssertionKind.MARK,
            scope_ref="run:r-1",
            key="export/mark",
            value={"label": "important"},
            body_text="operator mark",
            author_ref="user:operator",
            author_kind="user",
            evidence_refs=["message:s-1:1"],
            status="active",
            visibility="private",
            now_ms=1_700_000_001_000,
        )
        upsert_assertion(
            conn,
            assertion_id="export-deleted-note",
            target_ref="session:s-2",
            kind=AssertionKind.NOTE,
            scope_ref="run:r-2",
            key="export/note",
            body_text="deleted note retained for backup",
            status="deleted",
            visibility="private",
            now_ms=1_700_000_002_000,
        )


def _seed_blob_reference_debt(archive_root: Path, source: Path) -> None:
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('{"title":"recoverable"}\n', encoding="utf-8")
    missing_raw_hash = b"b" * 32
    missing_ref_hash = b"c" * 32
    with sqlite3.connect(archive_root / "source.db") as conn:
        write_source_raw_session_blob_ref(
            conn,
            origin="chatgpt-export",
            source_path=str(source),
            canonical_source_path=str(source),
            source_index=0,
            blob_hash=missing_raw_hash,
            blob_size=source.stat().st_size,
            acquired_at_ms=1,
            native_id="recoverable-chat",
        )
        conn.execute(
            """
            INSERT INTO blob_refs (
                blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (missing_ref_hash, "raw-gone", "raw_payload", str(archive_root / "missing-browser-capture.json"), 10, 1),
        )


def test_backup_plan_cli_reports_backup_profiles_and_tier_boundaries(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "backup-plan", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["mode"] == "backup_plan"
    assert payload["mutates"] is False
    assert payload["archive_root"] == str(cli_workspace["archive_root"])

    tiers = {tier["tier"]: tier for tier in payload["tiers"]}
    assert tiers["source"]["backup_class"] == "critical"
    assert tiers["source"]["backup_required"] is True
    assert tiers["index"]["backup_class"] == "warm_cache"
    assert tiers["index"]["backup_required"] is False
    assert tiers["embeddings"]["backup_policy"] == "back_up_when_present"
    assert tiers["user"]["backup_policy"] == "always_back_up"
    assert tiers["ops"]["backup_policy"] == "diagnostics_only"
    assert all(tier["present"] is True for tier in tiers.values())

    profiles = {profile["name"] for profile in payload["profiles"]}
    assert profiles == {
        "full_evidence",
        "user_overlays",
        "rebuildable_cache_exclude",
        "diagnostics_bundle",
    }
    assert payload["blob_store"]["path"] == str(cli_workspace["archive_root"] / "blob")
    assert payload["blob_store"]["backup_policy"] == "back_up_referenced_blobs_with_source_and_user_tiers"


def test_backup_plan_cli_surfaces_missing_tiers_and_wal_checkpoint_warning(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    archive_root = cli_workspace["archive_root"]
    (archive_root / "index.db").unlink()
    (archive_root / "user.db-wal").write_text("pending", encoding="utf-8")

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "backup-plan", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    tiers = {tier["tier"]: tier for tier in payload["tiers"]}
    assert tiers["index"]["present"] is False
    assert tiers["user"]["wal_present"] is True
    assert tiers["user"]["checkpoint_recommended"] is True
    assert payload["warnings"] == ["user.db-wal is present; checkpoint before copying user.db"]


def test_backup_plan_cli_renders_plain_summary(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "backup-plan"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "Archive backup plan" in result.output
    assert "source.db: critical policy=back_up present" in result.output
    assert "full_evidence:" in result.output


def test_assertion_export_cli_emits_all_assertions_as_jsonl(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = cli_workspace["archive_root"]
    _seed_assertion_export_rows(root)

    with cli_daemon_archive(root, monkeypatch):
        result = cli_runner.invoke(
            cli,
            ["--plain", "ops", "maintenance", "assertion-export"],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert [row["assertion_id"] for row in rows] == ["export-mark", "export-deleted-note"]
    assert rows[0]["kind"] == "mark"
    assert rows[0]["value"] == {"label": "important"}
    assert rows[0]["evidence_refs"] == ["message:s-1:1"]
    assert rows[1]["status"] == "deleted"


def test_assertion_export_cli_filters_and_writes_json_file(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_assertion_export_rows(cli_workspace["archive_root"])
    out_path = cli_workspace["archive_root"] / "exports" / "assertions.json"

    with cli_daemon_archive(cli_workspace["archive_root"], monkeypatch):
        result = cli_runner.invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "assertion-export",
                "--format",
                "json",
                "--kind",
                "note",
                "--status",
                "deleted",
                "--out",
                str(out_path),
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert result.stdout == f"Exported 1 assertions to {out_path}\n"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "assertion_export"
    assert payload["count"] == 1
    assert [row["assertion_id"] for row in payload["assertions"]] == ["export-deleted-note"]


@pytest.mark.parametrize("damage", ["missing", "unreadable"])
@pytest.mark.parametrize("existing_output", [False, True])
def test_assertion_export_refusal_preserves_output(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
    existing_output: bool,
) -> None:
    root = cli_workspace["archive_root"]
    output = root / "export.json"
    if existing_output:
        output.write_bytes(b"neutral existing output")
    with cli_daemon_archive(root, monkeypatch):
        user_path = root / "user.db"
        user_path.rename(root / "user-retained.db")
        if damage == "unreadable":
            user_path.write_bytes(b"neutral invalid SQLite authority")
        result = cli_runner.invoke(
            cli,
            ["--plain", "ops", "maintenance", "assertion-export", "--format", "json", "--out", str(output)],
        )
    assert result.exit_code != 0, result.output
    if existing_output:
        assert output.read_bytes() == b"neutral existing output"
    else:
        assert not output.exists()


def test_assertion_export_present_empty_user_writes_valid_empty_output(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = cli_workspace["archive_root"]
    output = root / "empty-export.json"
    with cli_daemon_archive(root, monkeypatch):
        result = cli_runner.invoke(
            cli,
            ["--plain", "ops", "maintenance", "assertion-export", "--format", "json", "--out", str(output)],
        )
    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert payload["count"] == 0
    assert payload["assertions"] == []


def test_blob_gc_cli_dry_run_reports_without_deleting(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    blob_hash = "aa" + "1" * 62
    candidate = _write_gc_candidate(cli_workspace, blob_hash)

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "blob-gc", "--max-batch", "5", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["mode"] == "blob_gc"
    assert payload["mutates"] is False
    assert payload["dry_run"] is True
    assert payload["candidate_count"] == 1
    assert payload["inspected_count"] == 1
    assert payload["would_delete_count"] == 1
    assert payload["deleted_count"] == 0
    assert payload["generation_written"] is False
    assert candidate.exists(), "dry-run must not delete the candidate"
    assert read_gc_history(cli_workspace["archive_root"] / "index.db", limit=1) == []


def test_blob_gc_cli_plain_preview_names_skip_counts(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    _write_gc_candidate(cli_workspace, "bb" + "2" * 62)

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "blob-gc", "--max-batch", "5"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "Blob GC dry-run" in result.output
    assert "Candidates: 1" in result.output
    assert "Result:     would delete 1 blob(s)" in result.output
    assert "referenced=0 reserved=0 missing=0 unlink_error=0" in result.output


def test_blocked_blob_gc_preview_is_a_machine_and_process_failure(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.blob_gc import BlobGCResult

    monkeypatch.setattr(
        "polylogue.storage.blob_gc.run_blob_gc_report",
        lambda *_args, **_kwargs: BlobGCResult(
            db_path="source.db", blob_dir="blob", dry_run=True, max_batch=100, blocked_reason="index unavailable"
        ),
    )
    json_result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "blob-gc", "--output-format", "json"],
    )
    plain_result = cli_runner.invoke(cli, ["--plain", "ops", "maintenance", "blob-gc"])

    assert json_result.exit_code == 1
    assert json.loads(json_result.stdout)["ok"] is False
    assert "index unavailable" in json_result.stdout
    assert plain_result.exit_code == 1
    assert "Blocked:" in plain_result.stdout


def test_blob_gc_cli_has_no_mutate_flag(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    """Read-only by design (automagic-invariants, polylogue-gd6v/4jsk/cfvvt): daemon
    convergence (``periodic_blob_gc_check``) already reclaims eligible blobs
    automatically in bounded batches, so a manual apply path would be a
    redundant, doctrine-forbidden break-glass surface -- the same treatment
    PR #3286 applied to ``embedding-orphan-reconcile`` in the same change.
    """
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "blob-gc", "--yes"],
    )

    # See test_embedding_orphan_reconcile_cli_has_no_mutate_flag: Click's
    # CliRunner surfaces a rejected/unknown option as SystemExit(2), so exit
    # code 2 plus the option name in the rejection message is the stable
    # contract to assert on.
    assert result.exit_code == 2
    assert "--yes" in result.output


def _seed_unreferenced_publication_receipt(archive_root: Path) -> str:
    from polylogue.storage.sqlite.write_lease import write_lease

    with write_lease("test.blob-publication-receipt", archive_root=archive_root):
        publisher = ArchiveBlobPublisher(
            archive_root / "source.db",
            archive_root / "blob",
        )
        blob_hash, _ = publisher.write_from_bytes(b"operator-adjudicated receipt")
        receipt_id = publisher.receipt_id(blob_hash)
        publisher.flush()
    assert receipt_id is not None
    return receipt_id


def test_blob_publications_cli_inspects_and_refuses_unconfirmed_abandonment(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    """Inspection is a read; abandonment still needs `--yes` before any dispatch."""
    archive_root = cli_workspace["archive_root"]
    receipt_id = _seed_unreferenced_publication_receipt(archive_root)

    inspected = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "blob-publications", "--output-format", "json"],
        catch_exceptions=False,
    )
    assert inspected.exit_code == 0
    payload = json.loads(inspected.stdout)
    assert payload["mutates"] is False
    assert payload["receipts"][0]["publication_id"] == receipt_id

    refused = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "blob-publications", "--abandon", receipt_id],
    )
    assert refused.exit_code != 0
    assert "--yes is required" in refused.output
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone() == (1,)


def test_blob_publications_abandonment_refuses_without_a_daemon(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    """A confirmed abandonment with no daemon refuses and writes nothing.

    Anti-vacuity: restore the deleted in-process call to
    ``abandon_blob_publication_receipts`` in ``blob_publications_command`` and
    this goes red -- the command exits 0 and the reservation row is gone,
    which is exactly the durable ``source.db`` DELETE-and-commit this route
    performed with no daemon, no write lease and no audited preview.
    """
    archive_root = cli_workspace["archive_root"]
    receipt_id = _seed_unreferenced_publication_receipt(archive_root)

    refused = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "blob-publications",
            "--abandon",
            receipt_id,
            "--yes",
            "--output-format",
            "json",
        ],
    )
    assert refused.exit_code != 0
    assert "daemon is unavailable; it must execute maintenance.blob-publications.abandon" in refused.output
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM blob_publication_reservations WHERE publication_id = ?",
            (receipt_id,),
        ).fetchone() == (1,)


def test_blob_publications_cli_abandons_through_the_resident_daemon(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The daemon applies the abandonment, keeps the blob, and audits the attempt."""
    archive_root = cli_workspace["archive_root"]
    receipt_id = _seed_unreferenced_publication_receipt(archive_root)
    store = BlobStore(archive_root / "blob")
    with sqlite3.connect(archive_root / "source.db") as conn:
        blob_hash = conn.execute(
            "SELECT blob_hash FROM blob_publication_reservations WHERE publication_id = ?",
            (receipt_id,),
        ).fetchone()[0]
    assert store.exists(bytes(blob_hash).hex())
    with cli_daemon_archive(archive_root, monkeypatch):
        abandoned = cli_runner.invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "blob-publications",
                "--abandon",
                receipt_id,
                "--yes",
                "--output-format",
                "json",
            ],
            catch_exceptions=False,
        )

    assert abandoned.exit_code == 0, abandoned.output
    payload = json.loads(abandoned.stdout)
    assert payload["abandonment"]["abandoned"] == 1
    assert payload["abandonment"]["skipped_referenced"] == 0
    assert payload["abandonment"]["blob_effect"] == "none"
    assert payload["receipt_ref"] is not None
    assert payload["receipts"] == []
    # The receipt is discharged; the blob it reserved is untouched.
    assert store.exists(bytes(blob_hash).hex())
    with sqlite3.connect(archive_root / "audit.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM operation_attempts AS attempt "
            "JOIN operation_runs AS run ON run.operation_id = attempt.operation_id "
            "WHERE run.operation_name = ?",
            ("mutate-abandon-blob-publication-receipts",),
        ).fetchone() == (1,)


def test_publish_many_persists_prior_shard_when_a_later_placement_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: removing the exceptional-path shard fsync leaves zero persisted shard names."""
    store = BlobStore(tmp_path / "blob")
    staged = tmp_path / "staged"
    staged.mkdir()
    payloads = (b"first shard", b"later shard")
    prepared = []
    for index, payload in enumerate(payloads):
        path = staged / str(index)
        path.write_bytes(payload)
        prepared.append(PreparedBlob(hashlib.sha256(payload).hexdigest(), len(payload), path))
    original_place = store._place_prepared
    calls = 0

    def fail_second(item: PreparedBlob) -> tuple[tuple[str, int], Path | None, bool]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated ENOSPC after first rename")
        return original_place(item)

    synced: list[Path] = []
    monkeypatch.setattr(store, "_place_prepared", fail_second)
    monkeypatch.setattr(store, "_fsync_directory", synced.append)
    with pytest.raises(OSError, match="simulated ENOSPC"):
        store.publish_many(prepared)
    assert store.blob_path(prepared[0].hash_hex).is_file()
    assert store.blob_path(prepared[0].hash_hex).parent in synced


def test_blob_reference_debt_cli_classifies_missing_refs(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    source = cli_workspace["archive_root"] / "exports" / "recoverable.json"
    _seed_blob_reference_debt(cli_workspace["archive_root"], source)

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "blob-reference-debt",
            "--sample-limit",
            "1",
            "--group-limit",
            "2",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["mode"] == "blob_reference_debt"
    assert payload["mutates"] is False
    assert payload["ok"] is False
    assert payload["missing_distinct_blobs"] == 2
    assert payload["missing_by_table"] == {"blob_refs": 2, "raw_sessions": 1}
    assert payload["missing_by_origin"] == {"(none)": 1, "chatgpt-export": 1}
    assert payload["missing_ref_id_join"] == {
        "ref_id_has_raw_session": 1,
        "ref_id_without_raw_session": 1,
    }
    assert payload["missing_source_path_presence"] == {
        "recoverable_source_path_exists": 1,
        "source_path_missing": 1,
    }
    assert len(payload["samples"]) == 1


def test_blob_reference_debt_cli_plain_output_names_read_only_debt(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    source = cli_workspace["archive_root"] / "exports" / "recoverable.json"
    _seed_blob_reference_debt(cli_workspace["archive_root"], source)

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "blob-reference-debt", "--sample-limit", "1"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "Blob reference debt" in result.output
    assert "Status:       debt-present" in result.output
    assert "Source paths: recoverable_source_path_exists=1, source_path_missing=1" in result.output


def _seed_orphan_embedding_row(archive_root: Path) -> tuple[str, str]:
    """Seed embeddings.db with one vector row for a message that no longer
    exists under an otherwise-live session — standing in for a message
    dropped by an index rebuild (polylogue-1dk1) while the session survives.
    """

    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, MaterialOrigin, Origin
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.embedding_write import (
        ArchiveEmbeddingWrite,
        upsert_message_embeddings,
    )
    from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION
    from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

    long_text = "This live message keeps the session present in the rebuilt index."
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        session_id = write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="orphan-cli-fixture",
                title="orphan reconcile fixture",
                messages=[
                    ParsedMessage(
                        provider_message_id="live",
                        role=Role.USER,
                        text=long_text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=long_text)],
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    )
                ],
            ),
        )

    orphan_message_id = f"{session_id}:orphaned-message-no-longer-in-index"
    with sqlite3.connect(archive_root / "embeddings.db") as conn:
        loaded, error = try_load_sqlite_vec(conn)
        if not loaded:
            pytest.skip(f"sqlite-vec extension is unavailable: {error}")
        upsert_message_embeddings(
            conn,
            [
                ArchiveEmbeddingWrite(
                    message_id=orphan_message_id,
                    session_id=session_id,
                    origin=Origin.CODEX_SESSION,
                    embedding=[0.01] * EMBEDDING_DIMENSION,
                    model="voyage-4",
                    embedded_at_ms=1_700_000_000_000,
                    vector_derivation_hash=hashlib.sha256(orphan_message_id.encode("utf-8")).digest(),
                )
            ],
        )
        conn.execute(
            """
            INSERT INTO embedding_status (
                session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
            ) VALUES (?, 'codex-session', 1, 1700000000000, 0, NULL)
            """,
            (session_id,),
        )
        conn.commit()
    index_db = archive_root / "index.db"
    (archive_root / ".index-active-pointer").write_text(str(index_db.resolve()), encoding="utf-8")
    generations = archive_root / ".index-generations" / "gen-current"
    generations.mkdir(parents=True, exist_ok=True)
    (generations / "generation.json").write_text(
        json.dumps(
            {
                "generation_id": "gen-current",
                "owner_id": "test",
                "archive_root": str(archive_root),
                "index_path": str(index_db.resolve()),
                "state": "active",
                "created_at_ms": 1_700_000_000_000,
                "source_snapshot": "source-at-rebuild-start",
            }
        ),
        encoding="utf-8",
    )
    return session_id, orphan_message_id


def test_embedding_orphan_reconcile_default_quiet_window_matches_reconcile_module() -> None:
    """The CLI's hardcoded --help default (polylogue-sod7) must not drift from the real constant.

    _embeddings.py hardcodes _DEFAULT_QUIET_WINDOW_SECONDS instead of importing
    DEFAULT_QUIET_WINDOW_MS from polylogue.storage.embeddings.reconcile, so
    that constant -- and its heavy import chain -- isn't paid on the
    `--help` path. This test is the drift guard for that duplication.
    """
    from polylogue.cli.commands.maintenance._embeddings import _DEFAULT_QUIET_WINDOW_SECONDS
    from polylogue.storage.embeddings.reconcile import DEFAULT_QUIET_WINDOW_MS

    assert _DEFAULT_QUIET_WINDOW_SECONDS == DEFAULT_QUIET_WINDOW_MS // 1000


def test_embedding_orphan_reconcile_cli_dry_run_keeps_rows(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    _seed_orphan_embedding_row(cli_workspace["archive_root"])
    with sqlite3.connect(cli_workspace["archive_root"] / "embeddings.db") as conn:
        conn.execute(
            """
            INSERT INTO embedding_status (
                session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
            ) VALUES ('codex-session:absent', 'codex-session', 0, 1700000000000, 0, NULL)
            """
        )

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "embedding-orphan-reconcile", "--output-format", "json"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["mode"] == "embedding_orphan_reconcile"
    assert payload["mutates"] is False
    assert payload["dry_run"] is True
    assert payload["orphan_message_rows"] == 1
    assert payload["candidate_message_rows"] == 1
    assert payload["candidate_message_meta_rows"] == 1
    assert payload["candidate_vector_rows"] == 1
    assert payload["candidate_status_rows"] == 1
    assert payload["removed_message_rows"] == 0
    assert payload["removed_vector_rows"] == 0
    assert payload["removed_status_rows"] == 0
    with sqlite3.connect(cli_workspace["archive_root"] / "embeddings.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1


def test_embedding_orphan_reconcile_cli_plain_dry_run_reports_would_remove_counts(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    _seed_orphan_embedding_row(cli_workspace["archive_root"])
    with sqlite3.connect(cli_workspace["archive_root"] / "embeddings.db") as conn:
        conn.execute(
            """
            INSERT INTO embedding_status (
                session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
            ) VALUES ('codex-session:absent', 'codex-session', 0, 1700000000000, 0, NULL)
            """
        )

    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "embedding-orphan-reconcile"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "Would remove:  1 message ref(s), 1 status row(s)" in result.output
    assert "Removed:" not in result.output
    with sqlite3.connect(cli_workspace["archive_root"] / "embeddings.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1


def test_embedding_orphan_reconcile_cli_has_no_mutate_flag(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
) -> None:
    """Read-only by design (automagic-invariants, polylogue-gd6v/4jsk): daemon
    convergence already reconciles this backlog automatically, so a manual
    apply path would be a redundant, doctrine-forbidden break-glass surface.
    """
    result = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "embedding-orphan-reconcile", "--yes"],
    )

    # Click's CliRunner surfaces a rejected/unknown option as SystemExit(2)
    # (its own UsageError is caught and converted before invoke() returns),
    # so exit code 2 plus the option name in the rejection message is the
    # stable, public contract to assert on -- not Click's internal
    # exception wording, which isn't guaranteed across versions.
    assert result.exit_code == 2
    assert "--yes" in result.output


def test_archive_maintenance_help_omits_copy_activation_surface(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(cli, ["--plain", "ops", "maintenance", "--help"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "archive-read" in result.output
    for removed in (
        "archive-copy-raw",
        "archive-copy-archive",
        "archive-copy-insights",
        "archive-copy-user",
        "archive-copy-all",
        "archive-copy-audit",
        "archive-activate",
    ):
        assert removed not in result.output


@pytest.mark.parametrize("output_format", ["plain", "json"])
def test_raw_authority_frontier_cli_inspects_without_applying_plans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli_runner: CliRunner,
    output_format: str,
) -> None:
    """The daemon measures frontier coverage; the CLI submits and renders it.

    The empty archive gains measured coverage without changing Source bytes.
    The daemon-down matrix in ``test_cli_operation_authority.py`` proves the
    command never executes in-process; local execution would leave the
    recorded operation submissions empty.
    """
    import polylogue.cli.operation_kernel as operation_kernel
    from tests.infra.daemon_operations import cli_daemon_archive

    submitted: list[str] = []
    configured = operation_kernel.configured_mutation_operation

    def recording(config: object, operation: str, payload: dict[str, object]) -> dict[str, object]:
        submitted.append(operation)
        return configured(config, operation, payload)

    monkeypatch.setattr(operation_kernel, "configured_mutation_operation", recording)
    with cli_daemon_archive(tmp_path / "archive", monkeypatch):
        source_path = tmp_path / "archive" / "source.db"
        source_before = source_path.read_bytes()
        result = cli_runner.invoke(
            cli,
            [
                "--plain",
                "ops",
                "maintenance",
                "raw-authority-frontier",
                "--output-format",
                output_format,
            ],
            catch_exceptions=False,
        )
        assert source_path.read_bytes() == source_before

    assert result.exit_code == 0, result.output
    assert submitted == ["maintenance.raw-authority-frontier"]
    if output_format == "json":
        payload = json.loads(result.stdout)
        assert payload["mode"] == "full"
        assert payload["healthy"]
        assert payload["accepted_head_checks"] == payload["blocking_head_checks"] == 0
        assert payload["cursor_checks"] == payload["cursor_ahead_count"] == payload["cursor_gap_count"] == 0
        assert payload["pass_id"].startswith("raw-authority-frontier-pass:")
        # Coverage reports observations, not executable plans or client query handles.
        assert "executable_plan_count" not in payload
        assert "state_counts" not in payload
        assert "query_handle" not in payload
    else:
        assert "Frontier full: healthy=True heads=0 blocked=0" in result.stdout
        assert "Cursors: checked=0 ahead=0 gaps=0" in result.stdout

    help_result = cli_runner.invoke(cli, ["--plain", "ops", "maintenance", "--help"])
    assert help_result.exit_code == 0
    assert "raw-authority-frontier" in help_result.output
    for removed in (
        "missing-raw-blob-cursors",
        "quarantined-accepted-raws",
        "browser-capture-origin-mismatches",
        "legacy-browser-capture-missing-native-id",
        "browser-canonical-authority-conflicts",
        "duplicate-raw-identity",
    ):
        assert removed not in help_result.output

    frontier_help = cli_runner.invoke(
        cli,
        ["--plain", "ops", "maintenance", "raw-authority-frontier", "--help"],
        catch_exceptions=False,
    )
    assert frontier_help.exit_code == 0
    assert "Measure current frontier coverage through the daemon preparation owner." in frontier_help.output
    for removed in ("--apply-plan", "--preview-census", "--yes"):
        assert removed not in frontier_help.output


@pytest.mark.parametrize(
    ("option", "value"),
    (("--apply-plan", "raw-authority-frontier:" + "a" * 64), ("--preview-census", "census"), ("--yes", None)),
)
def test_raw_authority_frontier_cli_rejects_removed_apply_options(
    cli_runner: CliRunner,
    option: str,
    value: str | None,
) -> None:
    rejected = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "raw-authority-frontier",
            option,
            *([value] if value is not None else []),
        ],
    )
    assert rejected.exit_code == 2
    assert f"No such option {option!r}." in rejected.output


def test_archive_read_cli_lists_archive_sessions(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3

    sqlite3.connect(cli_workspace["archive_root"] / "index.db").close()

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def close(self) -> None:
            return None

        def begin_read_snapshot(self) -> None:
            return None

        def end_read_snapshot(self) -> None:
            return None

        def set_read_progress_guard(self, *_args: object, **_kwargs: object) -> None:
            return None

        def clear_read_progress_guard(self) -> None:
            return None

        def list_summaries(self, *, limit: int, origin: str | None) -> list[ArchiveSessionSummary]:
            assert limit == 2
            assert origin == "codex-session"
            return [
                ArchiveSessionSummary(
                    session_id="codex-session:native-1",
                    native_id="native-1",
                    origin="codex-session",
                    title="Copied",
                    created_at="2026-01-02T03:04:05Z",
                    updated_at="2026-01-02T03:04:06Z",
                    message_count=3,
                    word_count=9,
                    tags=("archive",),
                )
            ]

    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.open_existing",
        classmethod(lambda cls, root, **_options: FakeArchiveStore()),
    )

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "archive-read",
            "--origin",
            "codex-session",
            "--limit",
            "2",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["mode"] == "list"
    assert payload["sessions"] == [
        {
            "created_at": "2026-01-02T03:04:05Z",
            "message_count": 3,
            "native_id": "native-1",
            "origin": "codex-session",
            "session_id": "codex-session:native-1",
            "tags": ["archive"],
            "title": "Copied",
            "updated_at": "2026-01-02T03:04:06Z",
            "word_count": 9,
        }
    ]


def test_archive_read_cli_searches_archive_blocks(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3

    sqlite3.connect(cli_workspace["archive_root"] / "index.db").close()

    class FakeArchiveStore:
        def __enter__(self) -> FakeArchiveStore:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def close(self) -> None:
            return None

        def begin_read_snapshot(self) -> None:
            return None

        def end_read_snapshot(self) -> None:
            return None

        def set_read_progress_guard(self, *_args: object, **_kwargs: object) -> None:
            return None

        def clear_read_progress_guard(self) -> None:
            return None

        def search_summaries(self, query: str, *, limit: int, origin: str | None) -> list[ArchiveSessionSearchHit]:
            assert query == "needle"
            assert limit == 5
            assert origin is None
            return [
                ArchiveSessionSearchHit(
                    rank=1,
                    session_id="codex-session:native-1",
                    block_id="codex-session:native-1:m1:0",
                    message_id="codex-session:native-1:m1",
                    origin="codex-session",
                    title="Copied",
                    snippet="[needle]",
                )
            ]

    monkeypatch.setattr(
        "polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.open_existing",
        classmethod(lambda cls, root, **_options: FakeArchiveStore()),
    )

    result = cli_runner.invoke(
        cli,
        [
            "--plain",
            "ops",
            "maintenance",
            "archive-read",
            "--query",
            "needle",
            "--limit",
            "5",
            "--output-format",
            "json",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["mode"] == "search"
    assert payload["hits"][0]["block_id"] == "codex-session:native-1:m1:0"
    assert payload["hits"][0]["snippet"] == "[needle]"


#: Verbs the manual rebuild engine and the generic repair product owned. They
#: were deleted with those products, so the CLI must not resolve them.
_RETIRED_MAINTENANCE_VERBS = (
    "rebuild-index",
    "rebuild-index-status",
    "reindex-canary",
    "plan",
    "run",
    "run-preview",
    "preview",
    "status",
    "blob-reference-closure",
    # Deleted with the fresh-archive lifecycle: opening an archive creates it,
    # and no current writer can produce the state the others repaired.
    "archive-plan",
    "archive-init",
    "migrate-tier",
    "blob-disposition",
    "blob-residue-compare",
    "embedding-preservation",
    "operation-recovery",
)


def test_retired_maintenance_verbs_are_absent_from_the_command_inventory() -> None:
    """Generated docs and shell completion must not rediscover the retired verbs.

    Anti-vacuity: re-registering any retired verb in
    ``polylogue.cli.commands.maintenance`` turns this red.
    """
    paths = {item.path for item in iter_command_paths(cli, include_root=False)}

    resurrected = sorted(verb for verb in _RETIRED_MAINTENANCE_VERBS if ("ops", "maintenance", verb) in paths)
    assert not resurrected, f"retired maintenance verbs back in the inventory: {resurrected}"


@pytest.mark.parametrize("verb", _RETIRED_MAINTENANCE_VERBS)
def test_retired_maintenance_verb_fails_discovery(verb: str, cli_runner: CliRunner) -> None:
    """Invoking a retired verb is a usage error, not a silent no-op run.

    Anti-vacuity: a compatibility shim that accepts the verb and exits 0 turns
    this red.
    """
    result = cli_runner.invoke(cli, ["ops", "maintenance", verb, "--help"])

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_blob_publication_abandonment_chunks_instead_of_refusing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: sending 257 IDs in one request is refused by the 256-ID request bound."""
    from polylogue.cli import operation_kernel
    from polylogue.cli.commands.maintenance import _blob_publications
    from polylogue.config import Config
    from polylogue.operations.daemon_protocol import BlobPublicationsAbandonRequest

    batches: list[list[str]] = []

    def fake_mutation(_config: object, _operation: str, payload: dict[str, list[str]]) -> dict[str, object]:
        BlobPublicationsAbandonRequest.model_validate(payload)
        batches.append(payload["publication_ids"])
        return {"result": {"abandoned": list(payload["publication_ids"])}, "receipt_ref": f"r{len(batches)}"}

    monkeypatch.setattr(operation_kernel, "configured_mutation_operation", fake_mutation)
    ids = tuple(f"pub-{index}" for index in range(257))

    merged = _blob_publications._submit_abandonment(
        Config(archive_root=Path("/archive"), render_root=Path("/render"), sources=[]), ids
    )

    assert [len(batch) for batch in batches] == [256, 1]
    assert merged["result"] == {"abandoned": list(ids)}
    assert merged["receipt_ref"] == "r1,r2"


@pytest.mark.parametrize("output_format", ["json", "jsonl"])
def test_assertion_export_cli_walks_multiple_daemon_pages(
    cli_workspace: dict[str, Path], cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch, output_format: str
) -> None:
    root = cli_workspace["archive_root"]
    with sqlite3.connect(root / "user.db") as user:
        user.executemany(
            "INSERT INTO assertions(assertion_id,target_ref,key,kind,value_json,created_at_ms,updated_at_ms) "
            "VALUES (?,?,?,?,?,?,?)",
            [
                (f"neutral-page-{index:04d}", "session:neutral", "neutral", "tag", "{}", index + 1, index + 1)
                for index in range(513)
            ],
        )
    with cli_daemon_archive(root, monkeypatch):
        result = cli_runner.invoke(
            cli,
            ["--plain", "ops", "maintenance", "assertion-export", "--format", output_format],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    rows = (
        json.loads(result.stdout)["assertions"]
        if output_format == "json"
        else [json.loads(line) for line in result.stdout.splitlines()]
    )
    assert [row["assertion_id"] for row in rows] == [f"neutral-page-{index:04d}" for index in range(513)]
