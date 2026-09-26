"""Qualify one ordinary-daemon build through terminal convergence.

This is a scratch qualification entry point, not a unit-test benchmark. It
copies nothing and writes only the explicitly supplied empty archive root.
The daemon watches the supplied source root through ``polylogued run``; the
supervisor waits for positive intake, promotion, archive-readiness, and debt
evidence before asking the daemon to shut down cleanly.

Run through the declared ``scratch`` operation after the candidate is frozen::

    python -m devtools.daemon_finished_build --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import closing, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from devtools.query_memory_budget import _read_process_tree_rss_kb
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from tests.infra.reindex_differential import capture_streamed_finished_build_fingerprint

MIB = 1024 * 1024
ReadinessDomain = Literal[
    "archive_sessions",
    "raw_artifacts",
    "search",
    "session_profiles",
    "threads",
    "tool_usage",
    "latency_profiles",
]
REQUIRED_READINESS_DOMAINS: Final[frozenset[ReadinessDomain]] = frozenset(
    {
        "archive_sessions",
        "raw_artifacts",
        "search",
        "session_profiles",
        "threads",
        "tool_usage",
        "latency_profiles",
    }
)
# No archive-readiness domain is currently optional for this qualification.
# If the ordinary daemon gains a typed optional domain, list it here explicitly;
# unknown or newly unready domains continue to block qualification.
ALLOWED_UNAVAILABLE_OPTIONAL_DOMAINS: Final[frozenset[ReadinessDomain]] = frozenset()


@dataclass(frozen=True, slots=True)
class BuildEvidence:
    input_bytes: int
    input_sha256: str
    accepted_input_files: int
    accepted_raw_rows: int
    raw_parse_failures: int
    raw_parse_pending: int
    cursor_complete: bool
    promoted_index_path: str | None
    schema_identity: str | None
    index_session_count: int
    index_message_count: int
    index_block_count: int
    fts_source_rows: int
    fts_indexed_rows: int
    open_convergence_debt: int
    readiness_surfaces: dict[str, bool]
    input_cursor: dict[str, Any] | None = None
    input_dispositions: tuple[dict[str, Any], ...] = ()
    canonical_logical_digest: str | None = None
    schema_object_census: tuple[tuple[str, str], ...] = ()
    output_equivalence_key: str | None = None

    @property
    def converged(self) -> bool:
        return (
            self.accepted_input_files == 1
            and self.accepted_raw_rows > 0
            and self.raw_parse_failures == 0
            and self.raw_parse_pending == 0
            and self.cursor_complete
            and self.promoted_index_path is not None
            and self.schema_identity is not None
            and self.index_session_count > 0
            and self.index_message_count > 0
            and self.index_block_count > 0
            and self.fts_source_rows > 0
            and self.fts_source_rows == self.fts_indexed_rows
            and self.open_convergence_debt == 0
            and self.readiness_surfaces.keys() >= REQUIRED_READINESS_DOMAINS
            and all(
                ready or domain in ALLOWED_UNAVAILABLE_OPTIONAL_DOMAINS
                for domain, ready in self.readiness_surfaces.items()
            )
        )

    @property
    def ready(self) -> bool:
        return (
            self.converged
            and self.input_cursor is not None
            and bool(self.input_dispositions)
            and all(
                item.get("terminal_disposition") == "materialized"
                and int(item.get("materialized_session_count", 0)) > 0
                and int(item.get("membership_count", 0)) > 0
                and int(item.get("membership_pending_count", 0)) == 0
                for item in self.input_dispositions
            )
            and self.canonical_logical_digest is not None
            and bool(self.schema_object_census)
            and self.output_equivalence_key is not None
        )


def _ro(path: Path, *, tier: ArchiveTier) -> sqlite3.Connection:
    return open_readonly_connection(
        path,
        tier=tier,
        timeout_class="background-read",
        validate_schema=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_receipt_once(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _table_count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


class _ProcessTreePeak:
    """Sample the declared existing procfs process-tree helper during the run."""

    def __init__(self, root_pid: int, *, interval_s: float = 0.05) -> None:
        self.root_pid = root_pid
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._peak_kb = 0
        self._thread = threading.Thread(target=self._sample, name="finished-build-rss", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.is_set():
            value = _read_process_tree_rss_kb(self.root_pid)
            with self._lock:
                self._peak_kb = max(self._peak_kb, value)
            self._stop.wait(self.interval_s)

    def finish(self) -> int:
        was_running = self._thread.is_alive()
        self._stop.set()
        self._thread.join()
        value = _read_process_tree_rss_kb(self.root_pid) if was_running else 0
        with self._lock:
            self._peak_kb = max(self._peak_kb, value)
            return self._peak_kb * 1024


def _observe(
    archive: Path,
    source: Path,
    expected_bytes: int,
    input_digest: str,
    *,
    capture_output: bool = False,
) -> BuildEvidence:
    ops_path = archive / "ops.db"
    source_path = archive / "source.db"
    if not ops_path.exists() or not source_path.exists():
        return BuildEvidence(expected_bytes, input_digest, 0, 0, 0, 0, False, None, None, 0, 0, 0, 0, 0, 1, {})

    with closing(_ro(ops_path, tier=ArchiveTier.OPS)) as conn:
        conn.row_factory = sqlite3.Row
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"ingest_cursor", "convergence_debt"} <= tables:
            return BuildEvidence(expected_bytes, input_digest, 0, 0, 0, 0, False, None, None, 0, 0, 0, 0, 0, 1, {})
        row = conn.execute("SELECT * FROM ingest_cursor WHERE source_path = ?", (str(source),)).fetchone()
        unresolved = int(conn.execute("SELECT COUNT(*) FROM convergence_debt WHERE status != 'resolved'").fetchone()[0])
        cursor_complete = bool(
            row is not None
            and int(row["stat_size"] or -1) == expected_bytes
            and int(row["byte_offset"] or -1) >= expected_bytes
            and row["deferred_end_offset"] is None
            and int(row["failure_count"] or 0) == 0
            and int(row["excluded"] or 0) == 0
        )
        input_cursor = (
            {
                "source_path": str(row["source_path"]),
                "stat_size": int(row["stat_size"] or 0),
                "byte_offset": int(row["byte_offset"] or 0),
                "deferred_end_offset": row["deferred_end_offset"],
                "failure_count": int(row["failure_count"] or 0),
                "excluded": int(row["excluded"] or 0),
                "complete": cursor_complete,
            }
            if row is not None
            else None
        )

    with closing(_ro(source_path, tier=ArchiveTier.SOURCE)) as conn:
        conn.row_factory = sqlite3.Row
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "raw_sessions" not in tables:
            return BuildEvidence(expected_bytes, input_digest, 0, 0, 0, 0, False, None, None, 0, 0, 0, 0, 0, 1, {})
        accepted_raw = int(
            conn.execute("SELECT COUNT(*) FROM raw_sessions WHERE source_path = ?", (str(source),)).fetchone()[0]
        )
        parse_failures = int(
            conn.execute(
                "SELECT COUNT(*) FROM raw_sessions WHERE source_path = ? AND parse_error IS NOT NULL",
                (str(source),),
            ).fetchone()[0]
        )
        parse_pending = int(
            conn.execute(
                "SELECT COUNT(*) FROM raw_sessions WHERE source_path = ? AND parsed_at_ms IS NULL",
                (str(source),),
            ).fetchone()[0]
        )
        raw_rows = conn.execute(
            """
            SELECT raw_id, source_index, blob_size, parse_error, parsed_at_ms,
                   validation_status, revision_authority
            FROM raw_sessions WHERE source_path = ? ORDER BY source_index, raw_id
            """,
            (str(source),),
        ).fetchall()
        raw_ids = [str(raw["raw_id"]) for raw in raw_rows]
        membership_counts: dict[str, dict[str, Any]] = {}
        if raw_ids and "raw_session_memberships" in tables:
            for membership in conn.execute(
                """
                SELECT raw_id, COUNT(*) AS total,
                       SUM(CASE WHEN decision IS NULL OR decision IN ('ambiguous', 'deferred')
                                OR revision_authority = 'quarantined' THEN 1 ELSE 0 END) AS pending
                FROM raw_session_memberships WHERE raw_id IN ({}) GROUP BY raw_id
                """.format(",".join("?" for _ in raw_ids)),
                raw_ids,
            ):
                membership_counts[str(membership["raw_id"])] = {
                    "total": int(membership["total"]),
                    "pending": int(membership["pending"] or 0),
                    "outcomes": [],
                }
            for membership in conn.execute(
                """
                SELECT raw_id, decision, revision_authority, COUNT(*) AS count
                FROM raw_session_memberships WHERE raw_id IN ({})
                GROUP BY raw_id, decision, revision_authority
                ORDER BY raw_id, decision, revision_authority
                """.format(",".join("?" for _ in raw_ids)),
                raw_ids,
            ):
                membership_counts[str(membership["raw_id"])]["outcomes"].append(
                    {
                        "decision": membership["decision"],
                        "revision_authority": membership["revision_authority"],
                        "count": int(membership["count"]),
                    }
                )

    pointer = archive / "index.db"
    promoted_path: str | None = None
    if pointer.is_symlink():
        try:
            target = pointer.resolve(strict=True)
            target.relative_to((archive / ".index-generations").resolve(strict=True))
            if target.name == "index.db":
                promoted_path = str(target)
        except (OSError, ValueError):
            pass

    sessions = messages = blocks = fts_source = fts_indexed = 0
    schema_identity: str | None = None
    canonical_logical_digest: str | None = None
    schema_object_census: tuple[tuple[str, str], ...] = ()
    output_equivalence_key: str | None = None
    input_dispositions: tuple[dict[str, Any], ...] = ()
    if promoted_path is not None:
        with closing(_ro(Path(promoted_path), tier=ArchiveTier.INDEX)) as conn:
            conn.row_factory = sqlite3.Row
            sessions, messages, blocks = (_table_count(conn, name) for name in ("sessions", "messages", "blocks"))
            identity = conn.execute("SELECT identity FROM schema_identity WHERE tier='index'").fetchone()
            if identity is not None and identity[0]:
                schema_identity = str(identity[0])
            if conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages_fts_readiness_binding'"
            ).fetchone():
                fts = conn.execute(
                    "SELECT source_rows, indexed_rows FROM messages_fts_readiness_binding "
                    "WHERE surface = 'messages_fts'"
                ).fetchone()
                if fts is not None:
                    fts_source, fts_indexed = int(fts[0]), int(fts[1])
            if "raw_ids" in locals() and raw_ids:
                materialized = {
                    str(row["raw_id"]): int(row["count"])
                    for row in conn.execute(
                        "SELECT raw_id, COUNT(*) AS count FROM sessions WHERE raw_id IN ({}) GROUP BY raw_id".format(
                            ",".join("?" for _ in raw_ids)
                        ),
                        raw_ids,
                    )
                }
                dispositions: list[dict[str, Any]] = []
                for raw in raw_rows:
                    raw_id = str(raw["raw_id"])
                    membership = membership_counts.get(raw_id, {"total": 0, "pending": 0})
                    session_count = materialized.get(raw_id, 0)
                    if raw["parse_error"] is not None:
                        terminal = "parser_refusal"
                    elif raw["validation_status"] == "failed":
                        terminal = "validation_refusal"
                    elif raw["parsed_at_ms"] is None:
                        terminal = "parse_pending"
                    elif session_count > 0 and membership["total"] > 0 and membership["pending"] == 0:
                        terminal = "materialized"
                    else:
                        terminal = "unmaterialized_or_unclassified"
                    dispositions.append(
                        {
                            "raw_id": raw_id,
                            "source_index": int(raw["source_index"]),
                            "blob_size": int(raw["blob_size"]),
                            "validation_status": raw["validation_status"],
                            "revision_authority": raw["revision_authority"],
                            "membership_count": membership["total"],
                            "membership_dispositions": membership.get("outcomes", []),
                            "membership_pending_count": membership["pending"],
                            "materialized_session_count": session_count,
                            "terminal_disposition": terminal,
                        }
                    )
                input_dispositions = tuple(dispositions)

    from polylogue.storage.archive_readiness import archive_readiness_status

    ready_surfaces: dict[str, bool] = {}
    if (
        cursor_complete
        and accepted_raw > 0
        and parse_failures == 0
        and parse_pending == 0
        and promoted_path is not None
        and sessions > 0
        and messages > 0
        and blocks > 0
        and fts_source > 0
        and fts_source == fts_indexed
        and unresolved == 0
    ):
        readiness = archive_readiness_status(archive)
        surfaces = readiness.get("surfaces", {}) if readiness.get("checked") is True else {}
        ready_surfaces = {
            str(name): isinstance(surface, dict) and surface.get("ready") is True for name, surface in surfaces.items()
        }
        if (
            capture_output
            and ready_surfaces.keys() >= REQUIRED_READINESS_DOMAINS
            and all(ready or domain in ALLOWED_UNAVAILABLE_OPTIONAL_DOMAINS for domain, ready in ready_surfaces.items())
            and input_dispositions
            and all(item["terminal_disposition"] == "materialized" for item in input_dispositions)
        ):
            sample_ids: tuple[str, ...] = ()
            with closing(_ro(Path(promoted_path), tier=ArchiveTier.INDEX)) as conn:
                sample_ids = tuple(
                    str(row[0]) for row in conn.execute("SELECT session_id FROM sessions ORDER BY session_id LIMIT 3")
                )
            fingerprint = capture_streamed_finished_build_fingerprint(
                archive,
                Path(promoted_path),
                scratch_root=archive.parent,
                session_ids=sample_ids,
                search_queries=(),
                include_threads=False,
            )
            canonical_logical_digest = fingerprint.canonical_logical_digest
            schema_object_census = fingerprint.schema_object_census
            schema_identity = fingerprint.schema_identity
            work_identity = {
                "source_sha256": input_digest,
                "source_bytes": expected_bytes,
                "profile": "polylogued-run:cold-build-index:standalone-off:no-default-sources",
            }
            output_equivalence_key = hashlib.sha256(
                json.dumps(
                    {
                        **work_identity,
                        "canonical_logical_digest": canonical_logical_digest,
                        "schema_object_census": schema_object_census,
                        "session_count": fingerprint.output_session_count,
                        "message_count": fingerprint.output_message_count,
                        "block_count": fingerprint.output_block_count,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
    return BuildEvidence(
        input_bytes=expected_bytes,
        input_sha256=input_digest,
        accepted_input_files=int(cursor_complete),
        accepted_raw_rows=accepted_raw,
        raw_parse_failures=parse_failures,
        raw_parse_pending=parse_pending,
        cursor_complete=cursor_complete,
        promoted_index_path=promoted_path,
        schema_identity=schema_identity,
        index_session_count=sessions,
        index_message_count=messages,
        index_block_count=blocks,
        fts_source_rows=fts_source,
        fts_indexed_rows=fts_indexed,
        open_convergence_debt=unresolved,
        readiness_surfaces=ready_surfaces,
        input_cursor=input_cursor,
        input_dispositions=input_dispositions,
        canonical_logical_digest=canonical_logical_digest,
        schema_object_census=schema_object_census,
        output_equivalence_key=output_equivalence_key,
    )


def _verify_args(args: argparse.Namespace) -> tuple[Path, Path, str]:
    if args.receipt.exists():
        raise FileExistsError(f"refusing to overwrite existing qualification receipt: {args.receipt}")
    candidate = args.candidate.resolve(strict=True)
    source_root = args.source_root.resolve(strict=True)
    source = args.input.resolve(strict=True)
    source.relative_to(source_root)
    if not source.is_file() or source.stat().st_size != args.expected_bytes:
        raise ValueError("input size does not match --expected-bytes")
    digest = _sha256(source)
    if args.expected_sha256 and digest != args.expected_sha256:
        raise ValueError("input SHA-256 does not match --expected-sha256")
    revision = subprocess.run(
        ["git", "-C", str(candidate), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if revision != args.candidate_sha:
        raise ValueError(f"candidate HEAD is {revision}, expected frozen {args.candidate_sha}")
    if not (candidate / "pyproject.toml").is_file():
        raise ValueError("candidate is not a Polylogue checkout")
    if Path(__file__).resolve().parents[1] != candidate:
        raise ValueError("run this maintained probe from the same candidate named by --candidate")
    dirty = subprocess.run(
        ["git", "-C", str(candidate), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if dirty.strip():
        raise ValueError("candidate worktree is dirty; qualification requires the frozen committed candidate")
    archive = args.archive_root.absolute()
    if archive.exists() and any(archive.iterdir()):
        raise ValueError("archive root must be absent or empty")
    siblings = sorted(
        path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() in {".jsonl", ".json"}
    )
    if siblings != [source]:
        raise ValueError(f"source root must contain exactly the declared input; found {siblings!r}")
    return candidate, source, digest


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    candidate, source, digest = _verify_args(args)
    archive = args.archive_root.absolute()
    archive.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["POLYLOGUE_ARCHIVE_ROOT"] = str(archive)
    # The qualification's required domains are local archive convergence. Keep
    # externally backed Sinex publication explicitly off for this scratch run.
    env["POLYLOGUE_SINEX_MODE"] = "off"
    env.pop("POLYLOGUE_CONFIG", None)
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = str(candidate)
    command = [
        sys.executable,
        "-c",
        "from polylogue.daemon.cli import main; main()",
        "run",
        "--root",
        str(args.source_root.resolve()),
        "--no-default-sources",
        "--no-browser-capture",
        "--no-api",
        "--cold-build-index",
    ]
    log_path = args.receipt.with_suffix(".daemon.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_stream = log_path.open("wb")
    daemon_started = time.monotonic()
    daemon = subprocess.Popen(command, cwd=candidate, env=env, stdout=log_stream, stderr=log_stream)
    sampler = _ProcessTreePeak(os.getpid())
    sampler.start()
    stable_ready = 0
    observations: list[dict[str, Any]] = []
    last_observation_error: str | None = None
    last_evidence: BuildEvidence | None = None
    convergence_elapsed_seconds: float | None = None
    deadline = time.monotonic() + args.timeout_seconds
    try:
        while time.monotonic() < deadline:
            if daemon.poll() is not None:
                raise RuntimeError(
                    f"ordinary daemon exited before qualification (status={daemon.returncode}); log={log_path}"
                )
            try:
                evidence = _observe(archive, source, args.expected_bytes, digest)
                last_evidence = evidence
            except (OSError, sqlite3.Error) as exc:
                last_observation_error = f"{type(exc).__name__}: {exc}"
                stable_ready = 0
                observations.append({"converged": False, "observation_error": last_observation_error})
                time.sleep(args.poll_seconds)
                continue
            last_observation_error = None
            observations.append({"converged": evidence.converged, "open_debt": evidence.open_convergence_debt})
            if evidence.converged:
                stable_ready += 1
            else:
                stable_ready = 0
                convergence_elapsed_seconds = None
            if stable_ready == 1:
                convergence_elapsed_seconds = time.monotonic() - daemon_started
            if stable_ready >= args.stable_polls:
                break
            time.sleep(args.poll_seconds)
        else:
            raise TimeoutError(
                f"daemon did not reach finished convergence within {args.timeout_seconds}s; "
                f"last_observation_error={last_observation_error!r}; last_observations={observations[-3:]!r}"
            )
        # Final evidence is reread after the stability window; stop only if
        # the terminal state still holds, then let the daemon drain normally.
        fingerprint_started = time.monotonic()
        final = _observe(archive, source, args.expected_bytes, digest, capture_output=True)
        fingerprint_seconds = time.monotonic() - fingerprint_started
        last_evidence = final
        if not final.ready:
            raise RuntimeError(f"finished-build evidence regressed before shutdown: {final}")
        shutdown_started = time.monotonic()
        daemon.send_signal(signal.SIGINT)
        try:
            daemon.wait(timeout=args.shutdown_timeout_seconds)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait()
            raise TimeoutError("ordinary daemon did not complete graceful shutdown") from None
        if daemon.returncode != 0:
            raise RuntimeError(f"ordinary daemon shutdown failed ({daemon.returncode}); log={log_path}")
        shutdown_seconds = time.monotonic() - shutdown_started
        elapsed_seconds = time.monotonic() - daemon_started
        peak_tree_rss_bytes = sampler.finish()
        result = {
            "candidate_sha": args.candidate_sha,
            "sinex_mode": "off",
            "command": command,
            "archive_root": str(archive),
            "daemon_log": str(log_path),
            "source_path": str(source),
            "evidence": final.__dict__
            if hasattr(final, "__dict__")
            else {field: getattr(final, field) for field in final.__dataclass_fields__},
            "stable_ready_polls": stable_ready,
            "ordinary_daemon_elapsed_seconds": elapsed_seconds,
            "convergence_elapsed_seconds": convergence_elapsed_seconds,
            "final_fingerprint_seconds": fingerprint_seconds,
            "graceful_shutdown_seconds": shutdown_seconds,
            "process_tree_peak_rss_bytes": peak_tree_rss_bytes,
            "process_tree_peak_rss_mib": round(peak_tree_rss_bytes / MIB, 1),
            "rss_limit_bytes": args.max_rss_bytes,
            "rss_limit_pass": peak_tree_rss_bytes <= args.max_rss_bytes,
            "shutdown_exit_code": daemon.returncode,
        }
        result["qualification_pass"] = bool(result["rss_limit_pass"] and final.ready)
        if args.receipt:
            _write_receipt_once(args.receipt, result)
        return result
    except Exception as exc:
        if daemon.poll() is None:
            daemon.send_signal(signal.SIGINT)
            try:
                daemon.wait(timeout=args.shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait()
        peak_tree_rss_bytes = sampler.finish()
        failure_result = {
            "candidate_sha": args.candidate_sha,
            "sinex_mode": "off",
            "command": command,
            "archive_root": str(archive),
            "daemon_log": str(log_path),
            "source_path": str(source),
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "last_observation_error": last_observation_error,
            "last_evidence": (
                {field: getattr(last_evidence, field) for field in last_evidence.__dataclass_fields__}
                if last_evidence is not None
                else None
            ),
            "observations": observations,
            "stable_converged_polls": stable_ready,
            "ordinary_daemon_elapsed_seconds": time.monotonic() - daemon_started,
            "convergence_elapsed_seconds": convergence_elapsed_seconds,
            "process_tree_peak_rss_bytes": peak_tree_rss_bytes,
            "process_tree_peak_rss_mib": round(peak_tree_rss_bytes / MIB, 1),
            "rss_limit_bytes": args.max_rss_bytes,
            "rss_limit_pass": peak_tree_rss_bytes <= args.max_rss_bytes,
            "daemon_exit_code": daemon.returncode,
            "qualification_pass": False,
        }
        with suppress(FileExistsError):
            _write_receipt_once(args.receipt, failure_result)
        raise
    finally:
        if daemon.poll() is None:
            daemon.send_signal(signal.SIGINT)
            try:
                daemon.wait(timeout=args.shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait()
        sampler.finish()
        log_stream.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True, help="frozen Polylogue worktree")
    parser.add_argument("--candidate-sha", required=True, help="expected immutable HEAD commit")
    parser.add_argument("--archive-root", type=Path, required=True, help="new private scratch archive root")
    parser.add_argument("--source-root", type=Path, required=True, help="directory containing only --input")
    parser.add_argument("--input", type=Path, required=True, help="one ordinary provider export under source root")
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--max-rss-bytes", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=28_000)
    parser.add_argument("--shutdown-timeout-seconds", type=int, default=120)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--stable-polls", type=int, default=3)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.expected_bytes <= 0 or args.max_rss_bytes <= 0 or args.stable_polls < 2:
        raise SystemExit("expected bytes, RSS limit, and stable poll count must be positive (stable polls >= 2)")
    try:
        result = qualify(args)
    except Exception as exc:
        failure = {
            "candidate_sha": args.candidate_sha,
            "source_path": str(args.input),
            "archive_root": str(args.archive_root),
            "qualification_pass": False,
            "failure": {"type": type(exc).__name__, "message": str(exc)},
        }
        if not args.receipt.exists():
            _write_receipt_once(args.receipt, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["qualification_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
