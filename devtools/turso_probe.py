"""Probe Turso Database compatibility for Polylogue storage research."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

ProbeStatus = Literal["pass", "fail", "skip"]


@dataclass(frozen=True, slots=True)
class ProbeResult:
    name: str
    status: ProbeStatus
    expected_status: ProbeStatus
    summary: str
    command: list[str]
    stdout: str = ""
    stderr: str = ""

    @property
    def expected(self) -> bool:
        return self.status == self.expected_status

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["expected"] = self.expected
        return payload


SQL_PROBES: tuple[tuple[str, tuple[str, ...], str, ProbeStatus, str], ...] = (
    (
        "strict_tables",
        (),
        "CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT) STRICT; "
        "INSERT INTO t(name) VALUES ('polylogue'); SELECT count(*) FROM t;",
        "pass",
        "STRICT tables are required by every archive tier.",
    ),
    (
        "stored_generated_columns",
        ("--experimental-generated-columns",),
        "CREATE TABLE t(a INTEGER, b INTEGER GENERATED ALWAYS AS (a + 1) STORED) STRICT;",
        "fail",
        "Polylogue index.db uses stored generated IDs; this is a direct-swap blocker when unsupported.",
    ),
    (
        "virtual_generated_columns",
        ("--experimental-generated-columns",),
        "CREATE TABLE t(a INTEGER, b INTEGER GENERATED ALWAYS AS (a + 1) VIRTUAL) STRICT; "
        "INSERT INTO t(a) VALUES (1); SELECT b FROM t;",
        "pass",
        "JSON-derived virtual columns are useful for tool/search projection fields.",
    ),
    (
        "fts5_virtual_table",
        (),
        "CREATE VIRTUAL TABLE messages_fts USING fts5(text);",
        "fail",
        "Polylogue's current FTS provider uses SQLite FTS5 virtual tables.",
    ),
    (
        "wal_journal_size_limit",
        (),
        "PRAGMA journal_mode=wal; PRAGMA journal_size_limit=1000;",
        "fail",
        "Polylogue's SQLite connection profile uses journal_size_limit to bound WAL sidecars.",
    ),
    (
        "mvcc_begin_concurrent",
        (),
        "PRAGMA journal_mode=mvcc; CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT) STRICT; "
        "BEGIN CONCURRENT; INSERT INTO t(name) VALUES ('a'); COMMIT; SELECT count(*) FROM t;",
        "pass",
        "MVCC and BEGIN CONCURRENT are the concurrency feature worth measuring.",
    ),
    (
        "cdc_id_mode",
        (),
        "PRAGMA capture_data_changes_conn('id'); CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT) STRICT; "
        "INSERT INTO t(name) VALUES ('a'); SELECT count(*) FROM turso_cdc;",
        "pass",
        "CDC is a plausible ops.db/archive-debt signal source if it remains stable.",
    ),
    (
        "vector_distance",
        (),
        "SELECT vector_distance_cos(vector32('[1,0]'), vector32('[0,1]')) AS distance;",
        "pass",
        "Built-in exact vector functions could support a separate vector-provider experiment.",
    ),
)


def _find_python_turso() -> object | None:
    return importlib.util.find_spec("turso")


def _import_python_turso() -> Any:
    return importlib.import_module("turso")


def _find_tursodb() -> str | None:
    return shutil.which("tursodb")


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False, timeout=20)


def _python_unavailable_result(name: str, summary: str) -> ProbeResult:
    return ProbeResult(
        name=name,
        status="skip",
        expected_status="skip",
        summary=summary,
        command=[sys.executable, "-c", "import turso"],
    )


def _python_binding_probe() -> ProbeResult:
    spec = _find_python_turso()
    if spec is None:
        return _python_unavailable_result(
            "python_binding",
            "Python package `turso` is not importable in this environment.",
        )
    try:
        turso = _import_python_turso()

        conn = turso.connect(":memory:")
    except Exception as exc:  # pragma: no cover - depends on optional external package
        return ProbeResult(
            name="python_binding",
            status="fail",
            expected_status="pass",
            summary=f"Python package `turso` imported but connect(':memory:') failed: {type(exc).__name__}: {exc}",
            command=[sys.executable, "-c", "import turso; turso.connect(':memory:')"],
        )
    with suppress(Exception):
        conn.close()
    return ProbeResult(
        name="python_binding",
        status="pass",
        expected_status="pass",
        summary="Python package `turso` imported and opened an in-memory connection.",
        command=[sys.executable, "-c", "import turso; turso.connect(':memory:')"],
    )


def _python_runtime_api_probe() -> ProbeResult:
    if _find_python_turso() is None:
        return _python_unavailable_result(
            "python_runtime_api",
            "Skipped because Python package `turso` is not importable.",
        )
    command = [
        sys.executable,
        "-c",
        "import turso; c=turso.connect(':memory:'); c.execute('select 1')",
    ]
    conn: Any | None = None
    try:
        turso = _import_python_turso()
        conn = turso.connect(":memory:")
        conn.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT) STRICT")
        cursor = conn.execute("INSERT INTO t(name) VALUES (?)", ("polylogue",))
        rowcount = cursor.rowcount
        lastrowid = cursor.lastrowid
        conn.commit()
        conn.row_factory = turso.Row
        row = conn.execute("SELECT id, name FROM t WHERE name = ?", ("polylogue",)).fetchone()
        if row is None:
            raise RuntimeError("row_factory query returned no row")
        if row["name"] != "polylogue":
            raise RuntimeError("row_factory did not support name lookup")
        if rowcount != 1:
            raise RuntimeError(f"unexpected rowcount {rowcount!r}")
        if lastrowid is None:
            raise RuntimeError("lastrowid was not populated")
        conn.executescript("CREATE TABLE aux(id INTEGER PRIMARY KEY) STRICT; INSERT INTO aux VALUES (1);")
        aux_row = conn.execute("SELECT count(*) FROM aux").fetchone()
        if aux_row is None or aux_row[0] != 1:
            raise RuntimeError(f"executescript did not create aux row: {aux_row!r}")
        conn.execute("INSERT INTO t(name) VALUES ('rollback-target')")
        conn.rollback()
        names = [record[0] for record in conn.execute("SELECT name FROM t ORDER BY id")]
        if names != ["polylogue"]:
            raise RuntimeError(f"rollback did not restore expected rows: {names!r}")
        conn.execute("PRAGMA user_version=42")
        version_row = conn.execute("PRAGMA user_version").fetchone()
        if version_row is None or version_row[0] != 42:
            raise RuntimeError(f"PRAGMA user_version roundtrip failed: {version_row!r}")
    except Exception as exc:  # pragma: no cover - depends on optional external package
        return ProbeResult(
            name="python_runtime_api",
            status="fail",
            expected_status="pass",
            summary=f"Python runtime API smoke failed: {type(exc).__name__}: {exc}",
            command=command,
        )
    finally:
        if conn is not None:
            with suppress(Exception):
                conn.close()
    return ProbeResult(
        name="python_runtime_api",
        status="pass",
        expected_status="pass",
        summary="Parameter binding, row_factory, executescript, rollback, cursor metadata, and PRAGMA user_version work.",
        command=command,
    )


def _python_readonly_uri_probe(*, scratch_dir: Path) -> ProbeResult:
    if _find_python_turso() is None:
        return _python_unavailable_result(
            "python_readonly_uri",
            "Skipped because Python package `turso` is not importable.",
        )
    db_path = scratch_dir / "python-readonly-uri.db"
    command = [
        sys.executable,
        "-c",
        "import turso; turso.connect('file:archive.db?mode=ro')",
    ]
    writer: Any | None = None
    reader: Any | None = None
    try:
        turso = _import_python_turso()
        db_path.unlink(missing_ok=True)
        writer = turso.connect(str(db_path))
        writer.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, name TEXT) STRICT")
        writer.execute("INSERT INTO t(name) VALUES ('seed')")
        writer.commit()
        writer.close()
        reader = turso.connect(f"file:{db_path}?mode=ro")
        count_row = reader.execute("SELECT count(*) FROM t").fetchone()
        if count_row is None or count_row[0] != 1:
            raise RuntimeError(f"read-only URI did not read seeded row: {count_row!r}")
        try:
            reader.execute("INSERT INTO t(name) VALUES ('should-fail')")
            reader.commit()
        except Exception:
            pass
        else:
            raise RuntimeError("read-only URI allowed a write")
    except Exception as exc:  # pragma: no cover - depends on optional external package
        return ProbeResult(
            name="python_readonly_uri",
            status="fail",
            expected_status="fail",
            summary=f"sqlite3-style file:...?mode=ro URI is not usable as Polylogue's readonly connection pattern: {type(exc).__name__}: {exc}",
            command=command,
        )
    finally:
        if writer is not None:
            with suppress(Exception):
                writer.close()
        if reader is not None:
            with suppress(Exception):
                reader.close()
    return ProbeResult(
        name="python_readonly_uri",
        status="pass",
        expected_status="fail",
        summary="sqlite3-style file:...?mode=ro URI opened and rejected writes.",
        command=command,
    )


def _python_multiprocess_probe(*, scratch_dir: Path) -> ProbeResult:
    if _find_python_turso() is None:
        return _python_unavailable_result(
            "python_multiprocess_wal",
            "Skipped because Python package `turso` is not importable.",
        )
    command = [
        sys.executable,
        "-c",
        "import turso; turso.connect('archive.db', experimental_features='multiprocess_wal')",
    ]
    conn: Any | None = None
    try:
        turso = _import_python_turso()
        db_path = scratch_dir / "python-multiprocess-wal.db"
        db_path.unlink(missing_ok=True)
        conn = turso.connect(str(db_path), experimental_features="multiprocess_wal")
        conn.execute("CREATE TABLE t(id INTEGER PRIMARY KEY) STRICT")
    except Exception as exc:  # pragma: no cover - depends on optional external package
        return ProbeResult(
            name="python_multiprocess_wal",
            status="fail",
            expected_status="pass",
            summary=f"Python multiprocess_wal connection failed: {type(exc).__name__}: {exc}",
            command=command,
        )
    finally:
        if conn is not None:
            with suppress(Exception):
                conn.close()
    return ProbeResult(
        name="python_multiprocess_wal",
        status="pass",
        expected_status="pass",
        summary="Python binding accepts experimental_features='multiprocess_wal'.",
        command=command,
    )


def _run_tursodb_sql(
    *,
    tursodb: str,
    scratch_dir: Path,
    name: str,
    flags: tuple[str, ...],
    sql: str,
    expected_status: ProbeStatus,
    summary: str,
) -> ProbeResult:
    db_path = scratch_dir / f"{name}.db"
    command = [tursodb, "--quiet", *flags, str(db_path), sql]
    result = _run_command(command)
    status: ProbeStatus = "pass" if result.returncode == 0 else "fail"
    return ProbeResult(
        name=name,
        status=status,
        expected_status=expected_status,
        summary=summary,
        command=command,
        stdout=result.stdout.strip(),
        stderr=result.stderr.strip(),
    )


def _attach_probe(*, tursodb: str, scratch_dir: Path) -> ProbeResult:
    sibling = scratch_dir / "attach_sibling.db"
    command_seed = [
        tursodb,
        "--quiet",
        str(sibling),
        "CREATE TABLE b(id INTEGER PRIMARY KEY) STRICT; INSERT INTO b VALUES (1);",
    ]
    seed = _run_command(command_seed)
    if seed.returncode != 0:
        return ProbeResult(
            name="attach_experimental",
            status="fail",
            expected_status="pass",
            summary="Failed to create the sibling database used for ATTACH probing.",
            command=command_seed,
            stdout=seed.stdout.strip(),
            stderr=seed.stderr.strip(),
        )
    db_path = scratch_dir / "attach_parent.db"
    sql = f"ATTACH DATABASE '{sibling}' AS sibling; SELECT count(*) FROM sibling.b;"
    command = [tursodb, "--quiet", "--experimental-attach", str(db_path), sql]
    result = _run_command(command)
    return ProbeResult(
        name="attach_experimental",
        status="pass" if result.returncode == 0 else "fail",
        expected_status="pass",
        summary="Polylogue cross-tier reads currently depend on ATTACH-style access.",
        command=command,
        stdout=result.stdout.strip(),
        stderr=result.stderr.strip(),
    )


def run_probe(*, tursodb: str | None = None, scratch_dir: Path | None = None) -> dict[str, object]:
    resolved_tursodb = tursodb or _find_tursodb()
    if scratch_dir is None:
        tmp_context = tempfile.TemporaryDirectory()
        scratch_root = Path(tmp_context.name)
    else:
        tmp_context = None
        scratch_root = scratch_dir
    try:
        results = [
            _python_binding_probe(),
            _python_runtime_api_probe(),
            _python_readonly_uri_probe(scratch_dir=scratch_root),
            _python_multiprocess_probe(scratch_dir=scratch_root),
        ]
        if resolved_tursodb is None:
            results.append(
                ProbeResult(
                    name="tursodb_binary",
                    status="skip",
                    expected_status="skip",
                    summary="`tursodb` is not on PATH; CLI feature probes were not run.",
                    command=["tursodb", "--version"],
                )
            )
        else:
            with tempfile.TemporaryDirectory(dir=scratch_root) as tmp:
                root = Path(tmp)
                for name, flags, sql, expected_status, summary in SQL_PROBES:
                    results.append(
                        _run_tursodb_sql(
                            tursodb=resolved_tursodb,
                            scratch_dir=root,
                            name=name,
                            flags=flags,
                            sql=sql,
                            expected_status=expected_status,
                            summary=summary,
                        )
                    )
                results.append(_attach_probe(tursodb=resolved_tursodb, scratch_dir=root))
    finally:
        if tmp_context is not None:
            tmp_context.cleanup()
    blockers = [
        result.name
        for result in results
        if (
            (result.name == "python_binding" and result.status != "pass")
            or (result.name == "python_readonly_uri" and result.status == "fail")
            or (result.name in {"stored_generated_columns", "fts5_virtual_table"} and result.status != "pass")
        )
    ]
    unexpected = [result.name for result in results if not result.expected]
    return {
        "ok": not unexpected,
        "tursodb": resolved_tursodb,
        "results": [result.to_dict() for result in results],
        "compatibility_blockers": blockers,
        "unexpected": unexpected,
        "recommendation": _recommendation(blockers=blockers, unexpected=unexpected),
    }


def _recommendation(*, blockers: list[str], unexpected: list[str]) -> str:
    if unexpected:
        return "Probe behavior changed; inspect unexpected results before drawing storage conclusions."
    if blockers:
        return (
            "Do not attempt a drop-in backend swap. Use this evidence to scope a tier-specific experiment, "
            "with ops.db and a separate vector provider as the lowest-risk candidates."
        )
    return "No immediate blocker found by the small probe; proceed to representative workload benchmarks."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit the probe payload as JSON.")
    parser.add_argument("--tursodb", help="Path to the tursodb binary. Defaults to PATH lookup.")
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path(".cache/turso-probe"),
        help="Directory for temporary probe databases.",
    )
    parser.add_argument(
        "--check", action="store_true", help="Return non-zero when probe outcomes differ from expectations."
    )
    args = parser.parse_args(argv)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    payload = run_probe(tursodb=args.tursodb, scratch_dir=args.scratch_dir)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Turso probe: {'ok' if payload['ok'] else 'unexpected'}")
        print(f"  tursodb: {payload['tursodb'] or 'missing'}")
        compatibility_blockers = payload["compatibility_blockers"]
        assert isinstance(compatibility_blockers, list)
        print(f"  blockers: {', '.join(str(blocker) for blocker in compatibility_blockers) or 'none'}")
        results = payload["results"]
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, dict)
            marker = "expected" if result["expected"] else "UNEXPECTED"
            print(f"  - {result['name']}: {result['status']} ({marker})")
        print(f"  recommendation: {payload['recommendation']}")
    return 1 if args.check and not payload["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
