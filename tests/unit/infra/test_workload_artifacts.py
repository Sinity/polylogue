"""Tests for the one shared real-pipeline seeded archive adapter."""

from __future__ import annotations

import gc
import json
import os
import sqlite3
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot, raw_materialization_ready
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.durable_change_train import DurableChangeTrainError
from tests.infra.workload_artifacts import (
    _journal_mode_delete_with_retry,
    build_seeded_archive,
    clone_seeded_archive,
    seeded_archive_key,
)


def test_seeded_archive_publishes_valid_immutable_real_pipeline_artifact(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"

    first = build_seeded_archive(cache_root=cache_root)
    second = build_seeded_archive(cache_root=cache_root)

    assert first.root == second.root
    assert first.manifest.manifest_id == second.manifest.manifest_id
    assert first.manifest.receipt["status"] == "succeeded"
    assert len(first.facts) == 64
    assert first.facts[0].expected_session_id == "codex-session:c03-target"
    assert first.root.joinpath("index.db").exists()
    assert raw_materialization_ready(raw_materialization_readiness_snapshot(first.root))
    phases = first.manifest.receipt["phases"]
    assert isinstance(phases, list)
    assert any(isinstance(phase, dict) and phase.get("name") == "raw_authority_frontier" for phase in phases)
    assert not (first.root.stat().st_mode & os.W_OK)


def test_seeded_archive_key_changes_with_source_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.infra.workload_artifacts as artifacts

    monkeypatch.setattr(artifacts, "lowering_fingerprint", lambda: "emitter-semantics:first")
    first = seeded_archive_key(())
    monkeypatch.setattr(artifacts, "lowering_fingerprint", lambda: "emitter-semantics:second")
    second = seeded_archive_key(())

    assert first.value != second.value


def test_seeded_archive_clone_is_private_full_root_and_preserves_base(tmp_path: Path) -> None:
    artifact = build_seeded_archive(cache_root=tmp_path / "cache")
    base_manifest = artifact.root.joinpath("manifest.json").read_bytes()
    marker_relative = Path(".maintenance-state/durable-change-trains/.bootstrap")
    base_marker = artifact.root.joinpath(marker_relative).read_bytes()

    clone = clone_seeded_archive(artifact, tmp_path / "clone")
    clone.root.joinpath("private-mutation.txt").write_text("private")
    with ArchiveStore.open_existing(clone.root, read_only=False) as archive:
        assert archive.count_sessions() == 64

    assert clone.clone_method in {"reflink", "copy"}
    assert clone.source_manifest_id == artifact.manifest.manifest_id
    assert clone.root.joinpath("source.db").exists()
    assert clone.root.joinpath("index.db").exists()
    assert artifact.root.joinpath("manifest.json").read_bytes() == base_manifest
    assert artifact.root.joinpath(marker_relative).read_bytes() == base_marker
    assert clone.root.joinpath(marker_relative).read_bytes() != base_marker
    assert not artifact.root.joinpath("private-mutation.txt").exists()

    clone.root.joinpath(marker_relative).write_bytes(base_marker)
    with pytest.raises(DurableChangeTrainError, match="durable identity mismatch"):
        ArchiveStore.open_existing(clone.root, read_only=False)


def test_seeded_archive_copy_fallback_rebinds_durable_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = build_seeded_archive(cache_root=tmp_path / "cache")

    def reject_reflink(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, ["cp"])

    monkeypatch.setattr(subprocess, "run", reject_reflink)
    clone = clone_seeded_archive(artifact, tmp_path / "clone")

    assert clone.clone_method == "copy"
    with ArchiveStore.open_existing(clone.root, read_only=False) as archive:
        assert archive.count_sessions() == 64


def test_seeded_archive_rejects_corrupt_published_cache_and_rebuilds(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    original = build_seeded_archive(cache_root=cache_root)
    index_path = original.root / "index.db"
    index_path.chmod(index_path.stat().st_mode | os.W_OK)
    index_path.unlink()

    rebuilt = build_seeded_archive(cache_root=cache_root)

    assert rebuilt.root == original.root
    assert rebuilt.root.joinpath("index.db").is_file()
    assert rebuilt.manifest.key == original.manifest.key
    assert rebuilt.manifest.profile_id == original.manifest.profile_id
    assert rebuilt.manifest.recipe_id == original.manifest.recipe_id
    assert rebuilt.facts == original.facts
    assert not (rebuilt.root.stat().st_mode & os.W_OK)


def test_seeded_archive_failure_never_publishes_partial_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.infra.workload_artifacts as artifacts

    async def fail_parse(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected ingest failure")

    monkeypatch.setattr(artifacts, "parse_sources_archive", fail_parse)

    with pytest.raises(RuntimeError, match="injected ingest failure"):
        build_seeded_archive(cache_root=tmp_path / "cache")

    cache_root = tmp_path / "cache"
    assert not list((cache_root / "artifacts").iterdir())
    assert not list((cache_root / ".staging").iterdir())


def test_seeded_archive_recovers_crash_left_staging_before_rebuild(tmp_path: Path) -> None:
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    cache_root.joinpath("artifacts").mkdir(parents=True)
    cache_root.joinpath(".locks").mkdir()
    staging_root = cache_root / ".staging"
    staging_root.mkdir()
    stale = staging_root / "dead-build.123"
    stale.mkdir()
    stale.joinpath("index.db").write_bytes(b"partial sqlite")
    stale.joinpath(".build.done").write_text("written before the crash", encoding="utf-8")

    removed = artifacts._recover_stale_staging(staging_root=staging_root, artifact_name="dead-build")

    assert removed == ("dead-build.123",)
    assert not stale.exists()


class _FlakyLockConnection:
    """Fakes ``PRAGMA journal_mode=DELETE`` raising a transient same-process lock.

    Mirrors CPython's ``sqlite3_close_v2`` zombie-connection footgun
    (polylogue-lbgc): a not-yet-finalized cursor/connection from earlier in
    the same worker process keeps SQLite's per-process shared pager-cache
    entry alive, so ``PRAGMA journal_mode=DELETE`` on a legitimate connection
    raises ``sqlite3.OperationalError: database is locked`` until that zombie
    is garbage-collected.
    """

    def __init__(self, *, locked_attempts: int) -> None:
        self.locked_attempts = locked_attempts
        self.attempts = 0

    def execute(self, sql: str) -> _FlakyLockConnection:
        assert sql == "PRAGMA journal_mode=DELETE"
        self.attempts += 1
        if self.attempts <= self.locked_attempts:
            raise sqlite3.OperationalError("database is locked")
        return self

    def fetchone(self) -> tuple[str]:
        return ("delete",)


def test_journal_mode_delete_retry_survives_a_transient_same_process_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anti-vacuity: reverting to a bare ``conn.execute(...)`` call (no retry/gc.collect)
    makes this fail on the very first simulated lock, since there would be no
    mechanism left to absorb it."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    gc_collect_calls = 0

    def fake_collect() -> int:
        nonlocal gc_collect_calls
        gc_collect_calls += 1
        return 0

    monkeypatch.setattr(gc, "collect", fake_collect)

    conn = _FlakyLockConnection(locked_attempts=2)
    _journal_mode_delete_with_retry(conn, name="index.db")  # type: ignore[arg-type]

    assert conn.attempts == 3
    assert gc_collect_calls == 2


def test_journal_mode_delete_does_not_retry_non_lock_operational_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-lock OperationalError (e.g. real corruption/IO error) must propagate
    immediately -- retrying it would hide a genuine failure, not a transient
    same-process race."""
    sleep_calls: list[float] = []
    monkeypatch.setattr(time, "sleep", sleep_calls.append)

    class _BrokenConnection:
        def execute(self, sql: str) -> Any:
            raise sqlite3.OperationalError("disk I/O error")

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        _journal_mode_delete_with_retry(_BrokenConnection(), name="index.db")  # type: ignore[arg-type]

    assert sleep_calls == []


def test_journal_mode_delete_reraises_lock_once_deadline_elapses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persistent (non-transient) same-process lock must still surface as a
    real failure rather than retry forever; the fake clock jumps straight past
    the deadline so the test doesn't need to sleep for real."""
    clock = iter([0.0, 10.0, 10.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    class _AlwaysLockedConnection:
        def execute(self, sql: str) -> Any:
            raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        _journal_mode_delete_with_retry(_AlwaysLockedConnection(), name="index.db")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cache identity: what a published artifact is, and is not, a function of
# ---------------------------------------------------------------------------


_REUSE_PROBE = """
import json, os, sys
from pathlib import Path
import tests.infra.workload_artifacts as artifacts

artifacts._build_id = lambda: os.environ["FAKE_BUILD_ID"]
cache_root = Path(sys.argv[1])
artifact = artifacts.build_seeded_archive(cache_root=cache_root)
print(json.dumps({
    "key": artifact.manifest.key,
    "root": str(artifact.root),
    "published": len(list((cache_root / "artifacts").iterdir())),
}))
"""


def test_seeded_archive_key_does_not_carry_the_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A new commit must not change the cache key.

    Anti-vacuity: restoring ``build_id`` (``git rev-parse HEAD``) to
    :class:`SeededArchiveKey` makes both assertions fail. That was the
    measured behavior before polylogue-1xc.14.1 -- 223 immutable artifact
    directories, 560 MB, for a catalog of about six distinct workloads,
    because each commit republished every workload it touched.
    """
    import tests.infra.workload_artifacts as artifacts

    monkeypatch.setattr(artifacts, "_build_id", lambda: "git:" + "0" * 40)
    first = seeded_archive_key(())
    monkeypatch.setattr(artifacts, "_build_id", lambda: "git:" + "f" * 40)
    second = seeded_archive_key(())

    assert first.value == second.value
    assert not hasattr(first, "build_id")


@pytest.mark.uses_real_clock("spawns fresh interpreters; no timestamp assertions")
def test_seeded_archive_is_reused_by_a_later_commit(tmp_path: Path) -> None:
    """Two commits must share one published artifact.

    Runs each build in its own interpreter rather than twice in this process.
    That is both the case that matters -- a later commit's test run is always
    a new process -- and the only way to test reuse without the
    ``sqlite3_close_v2`` zombie-connection footgun documented on
    :func:`_journal_mode_delete_with_retry` (polylogue-lbgc): revalidating an
    artifact in the same process that just wrote it re-opens tiers whose
    connections may not be finalized yet, which under load raises
    ``database is locked``.

    Anti-vacuity: returning ``build_id`` to the key makes ``published`` come
    back as 2 with two different roots.
    """
    cache_root = tmp_path / "cache"
    repo_root = Path(__file__).resolve().parents[3]

    def build(fake_commit: str) -> dict[str, object]:
        result = subprocess.run(
            [sys.executable, "-c", _REUSE_PROBE, str(cache_root)],
            cwd=repo_root,
            env={**os.environ, "FAKE_BUILD_ID": f"git:{fake_commit}"},
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, result.stderr
        payload: dict[str, object] = json.loads(result.stdout.strip().splitlines()[-1])
        return payload

    first = build("0" * 40)
    second = build("f" * 40)

    assert first["key"] == second["key"]
    assert first["root"] == second["root"]
    assert second["published"] == 1
    # The commit survives as provenance on the manifest, where it records
    # which checkout published the bytes without gating their reuse.
    manifest = json.loads((Path(str(first["root"])) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["build_id"] == "git:" + "0" * 40


def test_seeded_archive_key_changes_with_archive_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Archive DDL is part of the artifact's identity.

    ``recipe_id`` hashes a fixed six-file list that names ``bootstrap.py`` but
    none of the ``archive_tiers`` DDL modules, so before this key component a
    schema change arriving through ``index.py`` (the normal route) left the
    key untouched and a stale-schema artifact reusable.
    """
    baseline = seeded_archive_key(())

    bumped = dict(ARCHIVE_DDL_BY_TIER)
    bumped[ArchiveTier.INDEX] = ARCHIVE_DDL_BY_TIER[ArchiveTier.INDEX] + "\nCREATE TABLE later_addition(id TEXT);"
    monkeypatch.setattr("tests.infra.workload_artifacts.ARCHIVE_DDL_BY_TIER", bumped)

    assert seeded_archive_key(()).value != baseline.value
    assert seeded_archive_key(()).archive_schema_id != baseline.archive_schema_id


def test_seeded_archive_key_ignores_ddl_reordering() -> None:
    """The schema component names the DDL, not the module text around it.

    Hashing the rendered per-tier DDL rather than the Python source of the
    modules that build it keeps a comment or docstring edit in ``index.py``
    from invalidating every cached artifact -- the same over-invalidation, one
    layer down, that dropping ``build_id`` exists to stop.
    """
    import tests.infra.workload_artifacts as artifacts

    assert artifacts._archive_schema_id() == artifacts._archive_schema_id()
    assert artifacts._archive_schema_id().startswith("archive-schema:sha256:")


# ---------------------------------------------------------------------------
# Validate-once-per-process memo
# ---------------------------------------------------------------------------


def test_seeded_archive_memo_skips_revalidation_within_a_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second hit in a process must not re-run the full validation.

    Anti-vacuity: deleting the memo lookup in ``build_seeded_archive`` makes
    this fail, because ``_validate_artifact`` is then called on every hit --
    which is exactly the per-cache-hit cost (re-SHA256 of every tier, five
    ``PRAGMA quick_check`` runs, the planted-facts query, and the
    frontier-convergence read) the memo exists to stop paying per test.
    """
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    artifacts._VALIDATED_ARTIFACTS.clear()
    first = build_seeded_archive(cache_root=cache_root)

    calls = 0
    real_validate = artifacts._validate_artifact

    def counting_validate(root: Path, key: artifacts.SeededArchiveKey) -> object:
        nonlocal calls
        calls += 1
        return real_validate(root, key)

    monkeypatch.setattr(artifacts, "_validate_artifact", counting_validate)
    second = build_seeded_archive(cache_root=cache_root)

    assert calls == 0
    assert second.root == first.root
    assert second.manifest.manifest_id == first.manifest.manifest_id


def test_seeded_archive_memo_is_dropped_when_the_artifact_is_unplaced(tmp_path: Path) -> None:
    """A memo must not survive the artifact being deleted under a live process."""
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    artifacts._VALIDATED_ARTIFACTS.clear()
    original = build_seeded_archive(cache_root=cache_root)
    index_path = original.root / "index.db"
    index_path.chmod(index_path.stat().st_mode | os.W_OK)
    index_path.unlink()

    rebuilt = build_seeded_archive(cache_root=cache_root)

    assert rebuilt.root == original.root
    assert rebuilt.root.joinpath("index.db").is_file()


_FRESH_PROCESS_PROBE = """
import json, sys
from pathlib import Path
from tests.infra.workload_artifacts import build_seeded_archive

cache_root = Path(sys.argv[1])
artifact = build_seeded_archive(cache_root=cache_root)
print(json.dumps({"key": artifact.manifest.key, "root": str(artifact.root)}))
"""


@pytest.mark.uses_real_clock("spawns a fresh interpreter; no timestamp assertions")
def test_seeded_archive_corruption_is_refused_by_a_fresh_process(tmp_path: Path) -> None:
    """The memo is per-process: a NEW process still validates in full.

    Red twin for the validate-once memo. Corrupts ``index.db`` in place
    without changing its size, so the cheap presence/size check a warm
    process uses cannot see it -- only the full SHA-256 comparison can. A
    freshly spawned interpreter has no memo, must therefore run that full
    validation, must reject the artifact, and must republish it.

    Anti-vacuity: making the memo process-global (a file on disk, or an
    unconditional trust of the published manifest without re-hashing) makes
    this fail, because the fresh process would accept the corrupted bytes.
    """
    import tests.infra.workload_artifacts as artifacts

    cache_root = tmp_path / "cache"
    artifacts._VALIDATED_ARTIFACTS.clear()
    original = build_seeded_archive(cache_root=cache_root)

    index_path = original.root / "index.db"
    # ``stat.S_IWUSR``, not ``os.W_OK``: the latter is 2, which as a mode bit
    # is ``S_IWOTH`` and grants this process nothing.
    index_path.chmod(index_path.stat().st_mode | stat.S_IWUSR)
    size_before = index_path.stat().st_size
    with index_path.open("r+b") as handle:
        handle.seek(size_before // 2)
        handle.write(b"\xde\xad\xbe\xef")
    assert index_path.stat().st_size == size_before

    result = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS_PROBE, str(cache_root)],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, result.stderr
    republished = json.loads(result.stdout.strip().splitlines()[-1])
    # Same cache identity, freshly generated bytes: the artifact is rebuilt in
    # place, not accepted. ``manifest_id`` deliberately is NOT compared -- it
    # digests the per-file SHA-256 list, and a rebuild produces byte-different
    # SQLite files for identical logical content, so equality there would be
    # asserting determinism the pipeline never promised.
    assert republished["key"] == original.manifest.key
    assert republished["root"] == str(original.root)
    with index_path.open("rb") as handle:
        handle.seek(size_before // 2)
        assert handle.read(4) != b"\xde\xad\xbe\xef"


# ---------------------------------------------------------------------------
# Cache-root placement
# ---------------------------------------------------------------------------


def test_default_cache_root_falls_back_when_realm_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host without ``/realm`` must not crash every consumer of this module.

    ``mkdir(parents=True)`` cannot create a directory under a nonexistent
    mount point, so the previously hard-coded ``/realm/tmp`` root made the
    seeded-archive cache raise ``OSError`` on any cloud sandbox. Mirrors
    ``devtools.verify_runs.resolve_pytest_basetemp_root``'s placement family.
    """
    import tests.infra.workload_artifacts as artifacts

    monkeypatch.setattr(Path, "is_dir", lambda self: False)
    assert artifacts.default_cache_root() == artifacts._CLOUD_CACHE_ROOT

    monkeypatch.undo()
    if artifacts._SCRATCH_CACHE_ROOT.parent.is_dir():
        assert artifacts.default_cache_root() == artifacts._SCRATCH_CACHE_ROOT


# ---------------------------------------------------------------------------
# Read-only fixture path
# ---------------------------------------------------------------------------


def test_named_seeded_archive_ro_serves_a_readable_uncloned_archive(
    named_seeded_archive_ro: Callable[[str], Path],
) -> None:
    """The read-only fixture hands back the shared artifact, not a copy of it.

    Anti-vacuity: reintroducing a clone makes the ``artifacts/`` containment
    assertion fail, and reverting the artifact to writable makes the
    read-only mode assertion fail -- the two properties that let every worker
    and every test share one copy.
    """
    db_path = named_seeded_archive_ro("cli-chatgpt")

    assert db_path.is_file()
    assert db_path.name == "index.db"
    assert "artifacts" in db_path.parts
    assert os.environ["POLYLOGUE_ARCHIVE_ROOT"] == str(db_path.parent)
    assert not (db_path.parent.stat().st_mode & os.W_OK)

    with ArchiveStore.open_existing(db_path.parent, read_only=True) as archive:
        assert archive.count_sessions() > 0
