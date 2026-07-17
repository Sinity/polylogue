"""Executable anti-vacuity proof for the pytest-testmon affected gate.

The proof deliberately operates on a disposable copy of an actual Polylogue
production module and its existing production-route test.  It does not mock
pytest, testmon, changed paths, selection, or the verdict: pytest-testmon
records the dependency graph and the real test must fail after a semantic
source mutation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from devtools import repo_root

_SOURCE = Path("polylogue/core/web_urls.py")
_UNRELATED_SOURCE = Path("polylogue/core/stats.py")
_TEST = Path("tests/unit/core/test_web_urls.py")
_UNRELATED_TEST = Path("tests/unit/core/test_percentile.py")
_TARGET_NODEID = "tests/unit/core/test_web_urls.py::test_chatgpt_url_bare"
_MUTATION_FROM = 'return f"https://chatgpt.com/c/{native_id}"'
_MUTATION_TO = 'return f"https://chatgpt.invalid/c/{native_id}"'
_TIMEOUT_S = 60.0


@dataclass(frozen=True, slots=True)
class ProofResult:
    ok: bool
    target_nodeid: str
    selected_nodeids: tuple[str, ...]
    selected_count: int
    total_seeded_nodes: int
    mutation_exit_code: int
    restored_exit_code: int
    severed_edge_rejected: bool
    unrelated_selected_count: int
    cleanup_complete: bool
    failure: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _copy_required_tree(source_root: Path, scratch: Path) -> None:
    shutil.copytree(source_root / "polylogue" / "core", scratch / "polylogue" / "core")
    for relative in (
        Path("polylogue/__init__.py"),
        _TEST,
        _UNRELATED_TEST,
        Path("devtools/__init__.py"),
        Path("devtools/pytest_progress_plugin.py"),
    ):
        destination = scratch / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, destination)


def _pytest_env(scratch: Path, name: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(scratch)
    # The project devshell normally points testmon at .cache/testmon/testmondata.
    # Keep this proof's graph inside its disposable copy instead.
    env["TESTMON_DATAFILE"] = str(scratch / ".testmondata")
    env["POLYLOGUE_PYTEST_SELECTION_PATH"] = str(scratch / ".artifacts" / f"{name}-selection.json")
    env["POLYLOGUE_PYTEST_EVENTS_PATH"] = str(scratch / ".artifacts" / f"{name}-events.jsonl")
    env["POLYLOGUE_PYTEST_SUMMARY_PATH"] = str(scratch / ".artifacts" / f"{name}-summary.json")
    env["POLYLOGUE_PYTEST_SELECTION_NODEID_LIMIT"] = "100"
    return env


def _run_pytest(scratch: Path, *, name: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "devtools.pytest_progress_plugin", *args],
        cwd=scratch,
        env=_pytest_env(scratch, name),
        text=True,
        capture_output=True,
        timeout=_TIMEOUT_S,
        check=False,
    )


def _selection(scratch: Path, name: str) -> tuple[int, tuple[str, ...]]:
    path = scratch / ".artifacts" / f"{name}-selection.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0, ()
    selected = payload.get("selected_nodeids") if isinstance(payload, dict) else None
    count = payload.get("selected_count") if isinstance(payload, dict) else None
    if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
        selected = []
    return (count if isinstance(count, int) else len(selected), tuple(selected))


def _mutate(path: Path, before: str, after: str) -> None:
    source = path.read_text(encoding="utf-8")
    if source.count(before) != 1:
        raise RuntimeError(f"expected one mutation anchor in {path}")
    path.write_text(source.replace(before, after), encoding="utf-8")


def _sever_target_edge(testmon_db: Path) -> None:
    with sqlite3.connect(testmon_db) as connection:
        cursor = connection.execute(
            """
            DELETE FROM test_execution_file_fp
            WHERE test_execution_id IN (
                SELECT id FROM test_execution WHERE test_name = ?
            )
            AND fingerprint_id IN (
                SELECT id FROM file_fp WHERE filename LIKE ?
            )
            """,
            (_TARGET_NODEID, "%polylogue/core/web_urls.py"),
        )
        if cursor.rowcount < 1:
            raise RuntimeError("could not sever the named testmon dependency edge")


def _target_edge_present(testmon_db: Path) -> bool:
    with sqlite3.connect(testmon_db) as connection:
        row = connection.execute(
            """
            SELECT 1
            FROM test_execution_file_fp AS edge
            JOIN test_execution AS execution ON execution.id = edge.test_execution_id
            JOIN file_fp AS fingerprint ON fingerprint.id = edge.fingerprint_id
            WHERE execution.test_name = ? AND fingerprint.filename LIKE ?
            LIMIT 1
            """,
            (_TARGET_NODEID, "%polylogue/core/web_urls.py"),
        ).fetchone()
    return row is not None


@contextmanager
def _cleanup_on_interrupt() -> Iterator[None]:
    # typeshed models the previous handler returned by ``signal.signal`` more
    # narrowly than the handler value it accepts on restore.
    previous: dict[int, Any] = {}

    def interrupt(signum: int, _frame: object) -> None:
        raise InterruptedError(f"received signal {signum}")

    for signum in (int(signal.SIGINT), int(signal.SIGTERM)):
        previous[signum] = signal.signal(signum, interrupt)
    try:
        yield
    finally:
        for restored_signal, handler in previous.items():
            signal.signal(restored_signal, handler)


def run_proof(*, source_root: Path | None = None) -> ProofResult:
    """Run the real, bounded testmon dependency proof in a disposable copy."""
    source_root = (source_root or repo_root()).resolve()
    scratch: Path | None = None
    try:
        with _cleanup_on_interrupt(), tempfile.TemporaryDirectory(prefix="polylogue-testmon-proof-") as temporary:
            scratch = Path(temporary)
            _copy_required_tree(source_root, scratch)
            seed = _run_pytest(scratch, name="seed", args=["--testmon", "--testmon-noselect"])
            if seed.returncode != 0:
                raise RuntimeError(f"real-route seed failed: {seed.stderr or seed.stdout}")
            total_seeded_nodes, _ = _selection(scratch, "seed")
            if total_seeded_nodes < 2:
                raise RuntimeError("seed did not record a meaningful real-route test set")
            if not _target_edge_present(scratch / ".testmondata"):
                raise RuntimeError("seed did not record the named real-route dependency edge")
            seed_graph = scratch / ".testmondata.seed"
            shutil.copy2(scratch / ".testmondata", seed_graph)

            _mutate(scratch / _SOURCE, _MUTATION_FROM, _MUTATION_TO)
            affected = _run_pytest(scratch, name="mutation", args=["--testmon", "--testmon-forceselect", "-n", "0"])
            selected_count, selected_nodeids = _selection(scratch, "mutation")
            if (
                affected.returncode == 0
                or _TARGET_NODEID not in selected_nodeids
                or _TARGET_NODEID not in (affected.stdout + affected.stderr)
            ):
                raise RuntimeError("semantic mutation was not selected and failed by the named real-route test")

            _mutate(scratch / _SOURCE, _MUTATION_TO, _MUTATION_FROM)
            restored = _run_pytest(scratch, name="restored", args=[])
            if restored.returncode != 0:
                raise RuntimeError("removing the semantic mutation did not return the real route to green")

            shutil.copy2(seed_graph, scratch / ".testmondata")
            _sever_target_edge(scratch / ".testmondata")
            if _target_edge_present(scratch / ".testmondata"):
                raise RuntimeError("severed testmon dependency edge remained readable")
            severed_edge_rejected = True

            shutil.copy2(seed_graph, scratch / ".testmondata")
            # A byte-preserving rewrite is not a source change, so append a comment
            # outside executable behavior solely to make testmon inspect an unrelated file.
            unrelated = scratch / _UNRELATED_SOURCE
            unrelated.write_text(
                unrelated.read_text(encoding="utf-8") + "\n# testmon-proof unrelated change\n", encoding="utf-8"
            )
            unrelated_run = _run_pytest(
                scratch, name="unrelated", args=["--testmon", "--testmon-forceselect", "-n", "0"]
            )
            del unrelated_run
            unrelated_selected_count, _ = _selection(scratch, "unrelated")
            if unrelated_selected_count >= total_seeded_nodes:
                raise RuntimeError("unrelated production change fell back to the complete seeded route")

            proof = ProofResult(
                ok=True,
                target_nodeid=_TARGET_NODEID,
                selected_nodeids=selected_nodeids,
                selected_count=selected_count,
                total_seeded_nodes=total_seeded_nodes,
                mutation_exit_code=affected.returncode,
                restored_exit_code=restored.returncode,
                severed_edge_rejected=severed_edge_rejected,
                unrelated_selected_count=unrelated_selected_count,
                cleanup_complete=False,
            )
        return replace(proof, cleanup_complete=not scratch.exists())
    except (InterruptedError, subprocess.TimeoutExpired, OSError, RuntimeError, sqlite3.Error) as exc:
        return ProofResult(
            ok=False,
            target_nodeid=_TARGET_NODEID,
            selected_nodeids=(),
            selected_count=0,
            total_seeded_nodes=0,
            mutation_exit_code=-1,
            restored_exit_code=-1,
            severed_edge_rejected=False,
            unrelated_selected_count=0,
            cleanup_complete=scratch is not None and not scratch.exists(),
            failure=str(exc),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prove real pytest-testmon affected selection with a semantic mutation."
    )
    parser.add_argument("--json", action="store_true", help="Emit the complete proof receipt as JSON.")
    args = parser.parse_args(argv)
    result = run_proof()
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    elif result.ok:
        print(f"testmon mutation proof passed: {result.target_nodeid}")
    else:
        print(f"testmon mutation proof failed: {result.failure}", file=sys.stderr)
    return 0 if result.ok else 1
