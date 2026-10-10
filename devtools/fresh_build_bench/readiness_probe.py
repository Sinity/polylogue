"""Read-only readiness census executed with the measured arm's imports.

The driver owns this helper. It starts no daemon, compute owner or profiler;
its process cost is observer overhead, outside daemon resource samples.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class ReadinessProbeError(OSError):
    """The observer cannot bind a valid candidate readiness result."""


def _transient_census(readiness: Any) -> bool:
    # The product census reports SQLite read failures as unchecked data.
    if not isinstance(readiness, dict):
        return False
    reason = str(readiness.get("reason", "")).lower()
    return readiness.get("checked") is False and reason.startswith(
        (
            "database is locked",
            "database table is locked",
            "database schema is locked",
            "database is busy",
            "unable to open database file",
            "disk i/o error",
            "locking protocol",
        )
    )


def binding_valid(document: dict[str, Any]) -> bool:
    binding = document.get("binding") or {}
    candidate = document.get("candidate")
    if not isinstance(binding, dict) or not isinstance(candidate, str) or not candidate:
        return False
    expected = str(Path(candidate) / "polylogue/storage/archive_readiness.py")
    readiness = document.get("readiness")
    window = document.get("checked_monotonic_s")
    return (
        document.get("outcome") in ("bound", "retryable-observation-error")
        and document.get("helper_exit_code") == 0
        and isinstance(readiness, dict)
        and (
            (document["outcome"] == "bound" and readiness.get("checked") is True)
            or (document["outcome"] == "retryable-observation-error" and _transient_census(readiness))
        )
        and isinstance(readiness.get("surfaces"), dict)
        and all(isinstance(surface, dict) for surface in readiness["surfaces"].values())
        and isinstance(window, list)
        and len(window) == 2
        and all(isinstance(value, (float, int)) for value in window)
        and window[0] <= window[1]
        and binding.get("path") == expected
        and binding.get("module_path") == expected
        and bool(binding.get("sha256"))
        and binding.get("sha256") == document.get("expected_source_sha256")
    )


class CandidateReadinessProbe:
    def __init__(self, *, candidate: Path, python: str, environment: dict[str, str], work: Path) -> None:
        self.candidate = candidate.resolve()
        self.python = python
        self.work = work
        self.helper_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        # No profiler bootstrap or compute-event tracing in the observer.
        self.environment = {
            key: value
            for key, value in environment.items()
            if not key.startswith("POLYLOGUE_BENCH_") and key != "POLYLOGUE_TRACE_COMPUTE"
        }
        self.censuses: list[dict[str, Any]] = []

    @property
    def cache_key(self) -> str:
        source_hash = self._source_hash()
        return json.dumps(
            [str(self.candidate), source_hash, self.python, self.helper_hash, sorted(self.environment.items())],
            sort_keys=True,
        )

    def __call__(self, archive: Path) -> dict[str, Any]:
        source_hash = self._source_hash()
        began = time.monotonic()
        bootstrap = (
            "import runpy,sys;sys.path.insert(0,sys.argv[1]);"
            "runpy.run_path(sys.argv[2])['main'](sys.argv[1],sys.argv[3])"
        )
        try:
            result = subprocess.run(
                [self.python, "-c", bootstrap, str(self.candidate), str(Path(__file__).resolve()), str(archive)],
                cwd=self.candidate,
                env=self.environment,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise ReadinessProbeError(f"candidate readiness helper could not start: {exc}") from exc
        try:
            document = json.loads(result.stdout) if result.returncode == 0 else {}
        except ValueError:
            document = {}
        if not isinstance(document, dict):
            document = {}
        document.update(
            candidate=str(self.candidate),
            expected_source_sha256=source_hash,
            helper_sha256=self.helper_hash,
            helper_wall_s=round(time.monotonic() - began, 3),
            helper_exit_code=result.returncode,
            helper_stderr=result.stderr,
        )
        if document.get("outcome") == "bound" and _transient_census(document.get("readiness") or {}):
            document["outcome"] = "retryable-observation-error"
        if not binding_valid(document):
            document["outcome"] = "refused"
        self.censuses.append(document)
        (self.work / "readiness.json").write_text(json.dumps({"censuses": self.censuses}, sort_keys=True) + "\n")
        if document["outcome"] == "retryable-observation-error":
            raise sqlite3.OperationalError(document["readiness"]["reason"])
        if document["outcome"] != "bound":
            raise ReadinessProbeError(
                f"candidate readiness probe refused ({document.get('error', 'unbound imports or census')})"
            )
        return document

    def _source_hash(self) -> str:
        try:
            return hashlib.sha256((self.candidate / "polylogue/storage/archive_readiness.py").read_bytes()).hexdigest()
        except OSError as exc:
            raise ReadinessProbeError(f"candidate readiness source unavailable: {exc}") from exc


def main(candidate: str, archive: str) -> None:
    document: dict[str, Any] = {"candidate": str(Path(candidate).resolve())}
    try:
        import polylogue.storage.archive_readiness as module

        function = module.archive_readiness_status
        path = Path(function.__code__.co_filename).resolve()
        expected = Path(candidate).resolve() / "polylogue/storage/archive_readiness.py"
        document["binding"] = {
            "path": str(path),
            "module_path": str(Path(module.__file__).resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if path != expected or Path(module.__file__).resolve() != expected:
            document["outcome"] = "mismatch"
        else:
            began = time.monotonic()
            document["readiness"] = function(Path(archive))
            document["checked_monotonic_s"] = [began, time.monotonic()]
            document["outcome"] = "bound"
    except Exception as exc:
        document.update(outcome="error", error=type(exc).__name__, error_detail=str(exc))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    document["helper_cpu_s"] = {"user": usage.ru_utime, "system": usage.ru_stime}
    checker = getattr(sys, "_is_gil_enabled", None)
    document["gil_enabled_after_imports"] = bool(checker()) if checker else True
    document["python_executable"] = os.path.realpath(sys.executable)
    print(json.dumps(document, sort_keys=True))
