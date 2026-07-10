"""Verification-lab direct smoke runner."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, TextIO

from devtools import repo_root as _get_root
from devtools.cli_boundary import invoke_polylogue_cli
from devtools.visual_artifacts import (
    READER_VISUAL_SMOKE_PYTEST_COMMAND,
    reader_visual_artifact_payloads,
)
from polylogue.archive.message.roles import Role
from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import BlockType, Provider
from polylogue.core.outcomes import OutcomeStatus
from polylogue.pipeline.ids import session_content_hash
from polylogue.scenarios import AssertionSpec, ExecutionSpec, polylogue_execution
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.write import read_archive_session_envelope

_SCENARIO_NAMES = ("archive-smoke", "reader-visual-smoke", "storage-correctness")
_ARCHIVE_SMOKE_TIER = 0


class _ScenarioResult(Protocol):
    report_dir: Path | None

    @property
    def scenario_name(self) -> str: ...

    def stage_statuses(self) -> dict[str, OutcomeStatus]: ...

    def failed_stages(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class ArchiveSmokeCheck:
    name: str
    execution: ExecutionSpec
    assertion: AssertionSpec
    timeout_s: float = 60.0


@dataclass(frozen=True, slots=True)
class ArchiveSmokeCheckResult:
    check: ArchiveSmokeCheck
    passed: bool
    exit_code: int
    output: str
    duration_ms: float
    error: str | None = None


_ARCHIVE_SMOKE_CHECKS: tuple[ArchiveSmokeCheck, ...] = (
    ArchiveSmokeCheck(
        name="help-main",
        execution=polylogue_execution("--help"),
        assertion=AssertionSpec(stdout_contains=("polylogue",)),
    ),
    ArchiveSmokeCheck(
        name="help-mark-candidates",
        execution=polylogue_execution("mark", "candidates", "--help"),
        assertion=AssertionSpec(stdout_contains=("candidates",)),
    ),
    ArchiveSmokeCheck(
        name="completions-bash",
        execution=polylogue_execution("config", "completions", "--shell", "bash"),
        assertion=AssertionSpec(stdout_contains=("complete",)),
    ),
)


class ArchiveSmokeResult:
    """Direct result wrapper for the archive-smoke lab smoke."""

    def __init__(
        self,
        *,
        check_results: list[ArchiveSmokeCheckResult],
        report_dir: Path | None,
        unsupported_reason: str | None = None,
    ) -> None:
        self.check_results = check_results
        self.report_dir = report_dir
        self.unsupported_reason = unsupported_reason

    @property
    def scenario_name(self) -> str:
        return "archive-smoke"

    @property
    def all_passed(self) -> bool:
        return not self.failed_stages()

    def stage_statuses(self) -> dict[str, OutcomeStatus]:
        status = (
            OutcomeStatus.ERROR
            if self.unsupported_reason is not None or any(not result.passed for result in self.check_results)
            else OutcomeStatus.OK
        )
        return {
            "cli": status,
        }

    def failed_stages(self) -> tuple[str, ...]:
        return tuple(name for name, status in self.stage_statuses().items() if status is OutcomeStatus.ERROR)


def get_archive_smoke_checks() -> tuple[ArchiveSmokeCheck, ...]:
    """Return direct CLI checks for the archive-smoke lab smoke."""
    return _ARCHIVE_SMOKE_CHECKS


@dataclass(frozen=True, slots=True)
class StorageCorrectnessCheckResult:
    name: str
    passed: bool
    duration_ms: float
    details: dict[str, object]
    error: str | None = None


class StorageCorrectnessResult:
    """Result wrapper for the archive-backed storage-correctness scenario."""

    scenario_name = "storage-correctness"

    def __init__(
        self,
        *,
        check_results: list[StorageCorrectnessCheckResult],
        report_dir: Path | None,
    ) -> None:
        self.check_results = check_results
        self.report_dir = report_dir

    @property
    def all_passed(self) -> bool:
        return not self.failed_stages()

    def stage_statuses(self) -> dict[str, OutcomeStatus]:
        return {
            result.name: OutcomeStatus.OK if result.passed else OutcomeStatus.ERROR for result in self.check_results
        }

    def failed_stages(self) -> tuple[str, ...]:
        return tuple(result.name for result in self.check_results if not result.passed)


def run_tier_0() -> dict[str, str]:
    """Run direct archive-smoke checks and return output by check name."""
    checks = get_archive_smoke_checks()
    results: dict[str, str] = {}
    failures: list[str] = []
    total = len(checks)
    for index, check in enumerate(checks, start=1):
        print(f"  [{index:03d}/{total:03d}] {check.name}", flush=True)
        result = _run_archive_smoke_check(check)
        results[check.name] = result.output or ""
        if not result.passed:
            failures.append(f"{check.name}: exit {result.exit_code}: {result.error or 'failed'}")
    if failures:
        joined = "\n  - ".join(failures)
        raise RuntimeError(f"archive-smoke checks failed:\n  - {joined}")
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run executable archive and reader smoke sets.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    run_parser = subparsers.add_parser("run", help="Run a named smoke set.")
    run_parser.add_argument("scenario", choices=_SCENARIO_NAMES, help="Smoke set to run.")
    run_parser.add_argument(
        "--live", action="store_true", help="Run against the active archive instead of a seeded workspace."
    )
    run_parser.add_argument("--tier", type=int, default=None, help="Only run smoke checks at this tier.")
    run_parser.add_argument("--report-dir", type=Path, default=None, help="Directory for scenario artifacts.")
    run_parser.add_argument("--json", action="store_true", help="Emit a machine-readable scenario payload.")
    run_parser.add_argument("--verbose", action="store_true", help="Print smoke-check outputs.")
    run_parser.add_argument("--fail-fast", action="store_true", help="Stop on first smoke-check failure.")

    list_parser = subparsers.add_parser("list", help="List available smoke sets.")
    list_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def list_scenarios(*, as_json: bool) -> int:
    """List available scenario checks."""
    archive_checks = get_archive_smoke_checks()
    scenarios: list[dict[str, object]] = [
        {
            "name": "archive-smoke",
            "kind": "cli-smoke",
            "tier_0_check_count": len(archive_checks),
        },
        {
            "name": "reader-visual-smoke",
            "kind": "reader-visual",
            "command": " ".join((sys.executable, *READER_VISUAL_SMOKE_PYTEST_COMMAND[1:])),
            "artifact_count": len(reader_visual_artifact_payloads()),
        },
        {
            "name": "storage-correctness",
            "kind": "archive-storage",
            "check_count": len(_STORAGE_CORRECTNESS_CHECKS),
        },
    ]
    payload = {"scenarios": scenarios}
    if as_json:
        print(json.dumps(payload, indent=2))
        return 0
    for entry in scenarios:
        name = str(entry["name"])
        if name == "reader-visual-smoke":
            print(f"{name:<20s}  command: {entry['command']}")
            continue
        if name == "storage-correctness":
            print(f"{name:<20s}  checks: {entry['check_count']}")
            continue
        print(f"{name:<20s}  tier-0 checks: {entry['tier_0_check_count']}")
    return 0


def run_reader_visual_smoke(*, report_dir: Path | None, as_json: bool) -> int:
    """Run the daemon reader visual/DOM smoke lane."""
    command = [sys.executable, *READER_VISUAL_SMOKE_PYTEST_COMMAND[1:]]
    result = subprocess.run(
        command,
        cwd=_get_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    artifact_report = report_dir / "reader-visual-smoke.json" if report_dir is not None else None
    payload: dict[str, object] = {
        "scenario": "reader-visual-smoke",
        "command": command,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "artifact_inventory": reader_visual_artifact_payloads(),
        "artifact_report": str(artifact_report) if artifact_report is not None else None,
    }
    if report_dir is not None and artifact_report is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        artifact_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print("Running reader visual DOM smoke...")
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
    return result.returncode


def _format_scenario_summary(result: _ScenarioResult) -> str:
    """Format the scenario runner's direct stage result without report wrapping."""
    stage_statuses = result.stage_statuses()
    failed_stages = result.failed_stages()
    lines = ["Smoke stages:"]
    for name, status in stage_statuses.items():
        lines.append(f"  {name}: {status.value}")
    if failed_stages:
        lines.append(f"Failed stages: {', '.join(failed_stages)}")
    else:
        lines.append("Failed stages: none")
    if result.report_dir is not None:
        lines.append(f"Artifacts: {result.report_dir}")
    return "\n".join(lines)


def run_archive_smoke(
    *,
    live: bool,
    tier: int | None,
    report_dir: Path | None,
    verbose: bool,
    fail_fast: bool,
    as_json: bool = False,
) -> ArchiveSmokeResult:
    """Run direct archive-smoke CLI checks without catalog wrapping."""
    if tier not in (None, _ARCHIVE_SMOKE_TIER):
        return ArchiveSmokeResult(
            check_results=[],
            report_dir=report_dir,
            unsupported_reason=f"archive-smoke only supports tier {_ARCHIVE_SMOKE_TIER} direct CLI checks",
        )
    check_results = _run_archive_smoke_checks(
        verbose=verbose,
        fail_fast=fail_fast,
        progress_stream=sys.stderr if as_json else sys.stdout,
    )
    _write_archive_smoke_report(report_dir, check_results=check_results, live=live, tier=tier)
    return ArchiveSmokeResult(
        check_results=check_results,
        report_dir=report_dir,
    )


def _scenario_payload(result: _ScenarioResult) -> dict[str, object]:
    """Return the direct lab smoke payload without report wrapping."""
    stage_statuses = result.stage_statuses()
    failed_stages = result.failed_stages()
    payload: dict[str, object] = {
        "scenario": result.scenario_name,
        "stages": {name: status.value for name, status in stage_statuses.items()},
        "failed_stages": list(failed_stages),
        "ok": not failed_stages,
        "report_dir": str(result.report_dir) if result.report_dir is not None else None,
    }
    if isinstance(result, StorageCorrectnessResult):
        payload["checks"] = [
            {
                "name": check.name,
                "passed": check.passed,
                "duration_ms": round(check.duration_ms, 1),
                "details": check.details,
                "error": check.error,
            }
            for check in result.check_results
        ]
    return payload


def _parsed_message(provider_id: str, role: Role, text: str, position: int) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=provider_id,
        role=role,
        text=text,
        position=position,
        variant_index=0,
        is_active_path=True,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _parsed_session(
    native_id: str,
    messages: tuple[ParsedMessage, ...],
    *,
    title: str,
    parent_native_id: str | None = None,
    branch_type: BranchType | None = None,
) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=native_id,
        title=title,
        updated_at="2026-01-01T00:00:00Z",
        parent_session_provider_id=parent_native_id,
        branch_type=branch_type,
        messages=list(messages),
    )


def _storage_archive_root() -> TemporaryDirectory[str]:
    return TemporaryDirectory(prefix="polylogue-storage-correctness-")


def _row_count(conn: sqlite3.Connection, table: str, where: str = "", params: tuple[object, ...] = ()) -> int:
    clause = f" WHERE {where}" if where else ""
    row = conn.execute(f"SELECT COUNT(*) FROM {table}{clause}", params).fetchone()
    return int(row[0] if row is not None else 0)


def _storage_idempotent_reingest_check() -> dict[str, object]:
    session = _parsed_session(
        "storage-idempotent",
        (
            _parsed_message("m0", Role.USER, "idempotent user storage token", 0),
            _parsed_message("m1", Role.ASSISTANT, "idempotent assistant storage token", 1),
        ),
        title="Storage idempotent",
    )
    with _storage_archive_root() as temp_root:
        root = Path(temp_root)
        with ArchiveStore(root) as archive:
            first = archive.write_raw_and_parsed_result(
                session,
                payload=b'{"session":"storage-idempotent","version":1}',
                source_path="/scenario/storage-idempotent.json",
                acquired_at_ms=1_767_000_000_000,
            )
            second = archive.write_raw_and_parsed_result(
                session,
                payload=b'{"session":"storage-idempotent","version":1}',
                source_path="/scenario/storage-idempotent-repeat.json",
                acquired_at_ms=1_767_000_000_001,
            )
        with sqlite3.connect(root / "index.db") as conn:
            conn.row_factory = sqlite3.Row
            session_row = conn.execute(
                "SELECT content_hash, raw_id FROM sessions WHERE session_id = ?",
                (first.session_id,),
            ).fetchone()
            if session_row is None:
                raise AssertionError("idempotent scenario did not persist a session row")
            derived_counts = {
                "sessions": _row_count(conn, "sessions", "session_id = ?", (first.session_id,)),
                "messages": _row_count(conn, "messages", "session_id = ?", (first.session_id,)),
                "blocks": _row_count(conn, "blocks", "session_id = ?", (first.session_id,)),
                "message_fts": _row_count(conn, "messages_fts"),
            }
        with sqlite3.connect(root / "source.db") as source_conn:
            raw_count = _row_count(source_conn, "raw_sessions")
    expected_hash = str(session_content_hash(session))
    stored_hash = session_row["content_hash"]
    stored_hash_hex = stored_hash.hex() if isinstance(stored_hash, bytes) else str(stored_hash)
    if first.content_changed is not True:
        raise AssertionError("first ingest should write the derived session")
    if second.content_changed is not False or second.counts["skipped_sessions"] != 1:
        raise AssertionError(f"repeat ingest should skip unchanged content, got {second.counts}")
    if derived_counts != {"sessions": 1, "messages": 2, "blocks": 2, "message_fts": 2}:
        raise AssertionError(f"repeat ingest changed derived row counts: {derived_counts}")
    if raw_count != 2:
        raise AssertionError(f"raw acquisition evidence should retain both captures, got {raw_count}")
    if stored_hash_hex != expected_hash:
        raise AssertionError("stored content_hash does not match session_content_hash")
    return {
        "first_counts": first.counts,
        "repeat_counts": second.counts,
        "derived_counts": derived_counts,
        "raw_sessions": raw_count,
        "content_hash": stored_hash_hex,
    }


def _storage_fts_trigger_drift_check() -> dict[str, object]:
    session = _parsed_session(
        "storage-fts",
        (_parsed_message("m0", Role.USER, "stable fts repair sentinel", 0),),
        title="Storage FTS",
    )
    with _storage_archive_root() as temp_root:
        root = Path(temp_root)
        with ArchiveStore(root) as archive:
            first = archive.write_raw_and_parsed_result(
                session,
                payload=b'{"session":"storage-fts","version":1}',
                source_path="/scenario/storage-fts.json",
                acquired_at_ms=1_767_000_000_000,
            )
        with sqlite3.connect(root / "index.db") as conn:
            before = _row_count(conn, "messages_fts")
            conn.execute("DELETE FROM messages_fts")
            conn.commit()
            drifted = _row_count(conn, "messages_fts")
        with ArchiveStore(root) as archive:
            repeat = archive.write_raw_and_parsed_result(
                session,
                payload=b'{"session":"storage-fts","version":1}',
                source_path="/scenario/storage-fts-repeat.json",
                acquired_at_ms=1_767_000_000_001,
            )
            search_hits = archive.search_blocks("sentinel")
        with sqlite3.connect(root / "index.db") as conn:
            after = _row_count(conn, "messages_fts")
    if first.content_changed is not True:
        raise AssertionError("first FTS scenario ingest should write content")
    if before != 1 or drifted != 0:
        raise AssertionError(f"FTS drift setup failed: before={before}, drifted={drifted}")
    if repeat.content_changed is not False or repeat.counts.get("_fts_repair") != 1:
        raise AssertionError(f"unchanged ingest should repair FTS drift, got {repeat.counts}")
    if after != 1 or len(search_hits) != 1:
        raise AssertionError(f"FTS repair did not restore searchable row: after={after}, hits={search_hits}")
    return {
        "before_fts_rows": before,
        "drifted_fts_rows": drifted,
        "after_fts_rows": after,
        "repeat_counts": repeat.counts,
        "search_hits": search_hits,
    }


def _storage_lineage_composition_check() -> dict[str, object]:
    parent = _parsed_session(
        "storage-parent",
        (
            _parsed_message("p0", Role.USER, "hello", 0),
            _parsed_message("p1", Role.ASSISTANT, "hi there", 1),
            _parsed_message("p2", Role.USER, "parent continues alone", 2),
        ),
        title="Storage parent",
    )
    child = _parsed_session(
        "storage-child",
        (
            _parsed_message("c0", Role.USER, "hello", 0),
            _parsed_message("c1", Role.ASSISTANT, "hi there", 1),
            _parsed_message("cx", Role.USER, "child diverges here", 2),
            _parsed_message("cy", Role.ASSISTANT, "child reply", 3),
        ),
        title="Storage child",
        parent_native_id="storage-parent",
        branch_type=BranchType.FORK,
    )
    parent_grown = _parsed_session(
        "storage-parent",
        (
            _parsed_message("p0", Role.USER, "hello", 0),
            _parsed_message("p1", Role.ASSISTANT, "hi there", 1),
            _parsed_message("p2", Role.USER, "parent continues alone", 2),
            _parsed_message("p3", Role.ASSISTANT, "parent grows later", 3),
        ),
        title="Storage parent",
    )
    with _storage_archive_root() as temp_root:
        root = Path(temp_root)
        with ArchiveStore(root) as archive:
            parent_id = archive.write_parsed(parent, content_hash=str(session_content_hash(parent)))
            child_id = archive.write_parsed(child, content_hash=str(session_content_hash(child)))
            archive.write_parsed(parent_grown, content_hash=str(session_content_hash(parent_grown)))
            archive.commit()
        with sqlite3.connect(root / "index.db") as conn:
            conn.row_factory = sqlite3.Row
            stored_positions = [
                int(row["position"])
                for row in conn.execute(
                    "SELECT position FROM messages WHERE session_id = ? ORDER BY position",
                    (child_id,),
                ).fetchall()
            ]
            link = conn.execute(
                """
                SELECT resolved_dst_session_id, inheritance, branch_point_message_id
                FROM session_links
                WHERE src_session_id = ?
                """,
                (child_id,),
            ).fetchone()
            envelope = read_archive_session_envelope(conn, child_id)
            composed_texts = ["".join(block.text or "" for block in message.blocks) for message in envelope.messages]
    if parent_id != "codex-session:storage-parent":
        raise AssertionError(f"unexpected parent id: {parent_id}")
    if stored_positions != [2, 3]:
        raise AssertionError(f"child should physically store only divergent tail, got {stored_positions}")
    if link is None:
        raise AssertionError("prefix-sharing child did not persist a session_links row")
    if link["resolved_dst_session_id"] != parent_id:
        raise AssertionError(f"lineage link did not resolve to parent: {dict(link)}")
    if link["inheritance"] != "prefix-sharing" or not link["branch_point_message_id"]:
        raise AssertionError(f"lineage link did not capture prefix-sharing branch point: {dict(link)}")
    expected_texts = ["hello", "hi there", "child diverges here", "child reply"]
    if composed_texts != expected_texts:
        raise AssertionError(f"child did not compose the logical transcript: {composed_texts}")
    return {
        "parent_id": parent_id,
        "child_id": child_id,
        "stored_child_positions": stored_positions,
        "lineage": dict(link),
        "composed_texts": composed_texts,
    }


_STORAGE_CORRECTNESS_CHECKS: tuple[tuple[str, Callable[[], dict[str, object]]], ...] = (
    ("idempotent-reingest", _storage_idempotent_reingest_check),
    ("fts-trigger-drift", _storage_fts_trigger_drift_check),
    ("lineage-composition", _storage_lineage_composition_check),
)


def run_storage_correctness(*, report_dir: Path | None) -> StorageCorrectnessResult:
    """Run archive-backed storage correctness checks."""
    results: list[StorageCorrectnessCheckResult] = []
    for name, check in _STORAGE_CORRECTNESS_CHECKS:
        started = time.monotonic()
        try:
            details = check()
            passed = True
            error = None
        except Exception as exc:
            details = {}
            passed = False
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            StorageCorrectnessCheckResult(
                name=name,
                passed=passed,
                duration_ms=(time.monotonic() - started) * 1000,
                details=details,
                error=error,
            )
        )
    result = StorageCorrectnessResult(check_results=results, report_dir=report_dir)
    _write_storage_correctness_report(result)
    return result


def _write_storage_correctness_report(result: StorageCorrectnessResult) -> None:
    if result.report_dir is None:
        return
    result.report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": result.scenario_name,
        "checks": [
            {
                "name": check.name,
                "passed": check.passed,
                "duration_ms": round(check.duration_ms, 1),
                "details": check.details,
                "error": check.error,
            }
            for check in result.check_results
        ],
    }
    (result.report_dir / "storage-correctness.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_archive_smoke_check(check: ArchiveSmokeCheck) -> ArchiveSmokeCheckResult:
    started = time.monotonic()
    try:
        cli_result = invoke_polylogue_cli(check.execution, env={"POLYLOGUE_FORCE_PLAIN": "1"}, timeout=check.timeout_s)
    except subprocess.TimeoutExpired:
        duration_ms = (time.monotonic() - started) * 1000
        return ArchiveSmokeCheckResult(
            check=check,
            passed=False,
            exit_code=-1,
            output="",
            duration_ms=duration_ms,
            error=f"timed out after {check.timeout_s:.0f}s",
        )
    except Exception as exc:
        duration_ms = (time.monotonic() - started) * 1000
        return ArchiveSmokeCheckResult(
            check=check,
            passed=False,
            exit_code=-1,
            output="",
            duration_ms=duration_ms,
            error=f"invoke crashed: {exc}",
        )
    output = cli_result.output
    error = check.assertion.validate_process(output, cli_result.exit_code)
    duration_ms = (time.monotonic() - started) * 1000
    return ArchiveSmokeCheckResult(
        check=check,
        passed=error is None,
        exit_code=cli_result.exit_code,
        output=output,
        duration_ms=duration_ms,
        error=error,
    )


def _run_archive_smoke_checks(
    *,
    verbose: bool,
    fail_fast: bool,
    progress_stream: TextIO,
) -> list[ArchiveSmokeCheckResult]:
    results: list[ArchiveSmokeCheckResult] = []
    checks = get_archive_smoke_checks()
    for index, check in enumerate(checks, start=1):
        print(f"  [{index:03d}/{len(checks):03d}] {check.name}", flush=True, file=progress_stream)
        result = _run_archive_smoke_check(check)
        results.append(result)
        if verbose and result.output:
            print(result.output, end="" if result.output.endswith("\n") else "\n", file=progress_stream)
        if fail_fast and not result.passed:
            break
    return results


def _write_archive_smoke_report(
    report_dir: Path | None,
    *,
    check_results: list[ArchiveSmokeCheckResult],
    live: bool,
    tier: int | None,
) -> None:
    if report_dir is None:
        return
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": "archive-smoke",
        "live": live,
        "tier": tier,
        "checks": [
            {
                "name": result.check.name,
                "exit_code": result.exit_code,
                "passed": result.passed,
                "duration_ms": round(result.duration_ms, 1),
                "error": result.error,
            }
            for result in check_results
        ],
    }
    (report_dir / "archive-smoke.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.action == "list":
        return list_scenarios(as_json=bool(args.json))
    if args.action != "run":
        parser.error(f"unknown action: {args.action}")
    if args.scenario == "reader-visual-smoke":
        return run_reader_visual_smoke(report_dir=args.report_dir, as_json=bool(args.json))
    result: _ScenarioResult
    if args.scenario == "storage-correctness":
        result = run_storage_correctness(report_dir=args.report_dir)
    else:
        result = run_archive_smoke(
            live=bool(args.live),
            tier=args.tier,
            report_dir=args.report_dir,
            verbose=bool(args.verbose),
            fail_fast=bool(args.fail_fast),
            as_json=bool(args.json),
        )
    if args.json:
        print(json.dumps(_scenario_payload(result), indent=2))
    else:
        print(_format_scenario_summary(result))
    return 0 if result.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
