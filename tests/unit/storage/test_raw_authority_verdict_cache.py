"""Persisted RawAuthorityVerdict cache (polylogue-w6hql Phase 2 / polylogue-tw4ar).

Anti-vacuity: ``test_second_read_does_not_recompute`` proves the cache is
actually consulted (not merely written and ignored) by monkeypatching the
underlying projection to raise if called a second time. ``test_new_revision_
invalidates_the_cache`` proves the fingerprint-based invalidation is real by
adding a new raw to a cached cohort and checking the stale cache is rejected
and the recomputed result reflects the new member, not the cached one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.revision_authority import RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider, RawAuthorityVerdict
from polylogue.storage import raw_authority_verdict_cache as cache_module
from polylogue.storage.raw_authority_verdict_cache import (
    get_or_compute_raw_authority_verdicts,
    read_cached_raw_authority_verdicts,
    write_raw_authority_verdict_cache,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _bind_full(archive: ArchiveStore, *, raw_id: str, payload: bytes, logical_source_key: str) -> str:
    written_id = archive.write_raw_payload(
        provider=Provider.CODEX,
        payload=payload,
        source_path="session.jsonl",
        acquired_at_ms=1,
        raw_id=raw_id,
    )
    archive.bind_raw_revision(
        written_id,
        RawRevisionEnvelope(logical_source_key, RawRevisionKind.FULL, f"revision-{raw_id}", 0),
    )
    return written_id


def test_cold_cache_computes_and_persists(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        _bind_full(archive, raw_id="oldest", payload=b"one\n", logical_source_key="codex:s1")
        _bind_full(archive, raw_id="newest", payload=b"one\ntwo\n", logical_source_key="codex:s1")

        assert read_cached_raw_authority_verdicts(archive, "codex:s1") is None

        verdicts = get_or_compute_raw_authority_verdicts(archive, "codex:s1", now_ms=1000)

        cached = read_cached_raw_authority_verdicts(archive, "codex:s1")

    assert verdicts == {
        "oldest": RawAuthorityVerdict.SUPERSEDED,
        "newest": RawAuthorityVerdict.VERIFIED,
    }
    assert cached == verdicts


def test_second_read_does_not_recompute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        _bind_full(archive, raw_id="only", payload=b"payload", logical_source_key="codex:s1")

        first = get_or_compute_raw_authority_verdicts(archive, "codex:s1", now_ms=1000)

        def _fail(*args: object, **kwargs: object) -> dict[str, RawAuthorityVerdict]:
            raise AssertionError("project_raw_authority_verdicts must not run against a warm cache")

        monkeypatch.setattr(cache_module, "project_raw_authority_verdicts", _fail)

        second = get_or_compute_raw_authority_verdicts(archive, "codex:s1", now_ms=2000)

    assert first == {"only": RawAuthorityVerdict.SOLE_COPY}
    assert second == first


def test_new_revision_invalidates_the_cache(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        _bind_full(archive, raw_id="only", payload=b"payload", logical_source_key="codex:s1")
        first = get_or_compute_raw_authority_verdicts(archive, "codex:s1", now_ms=1000)
        assert first == {"only": RawAuthorityVerdict.SOLE_COPY}

        # A second revision joins the cohort -- the previously-cached SOLE_COPY
        # verdict for "only" is now wrong (it has a successor).
        _bind_full(archive, raw_id="newer", payload=b"payload\nmore\n", logical_source_key="codex:s1")

        assert read_cached_raw_authority_verdicts(archive, "codex:s1") is None

        second = get_or_compute_raw_authority_verdicts(archive, "codex:s1", now_ms=2000)

    assert second == {
        "only": RawAuthorityVerdict.SUPERSEDED,
        "newer": RawAuthorityVerdict.VERIFIED,
    }


def test_write_requires_full_cohort_and_replaces_prior_rows(tmp_path: Path) -> None:
    """A rewrite must clear the cohort's whole prior row set, not accumulate."""
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        _bind_full(archive, raw_id="oldest", payload=b"one\n", logical_source_key="codex:s1")
        _bind_full(archive, raw_id="newest", payload=b"one\ntwo\n", logical_source_key="codex:s1")

        write_raw_authority_verdict_cache(
            archive,
            "codex:s1",
            {"oldest": RawAuthorityVerdict.SUPERSEDED, "newest": RawAuthorityVerdict.VERIFIED},
            now_ms=1000,
        )
        write_raw_authority_verdict_cache(
            archive,
            "codex:s1",
            {"oldest": RawAuthorityVerdict.SUPERSEDED, "newest": RawAuthorityVerdict.VERIFIED},
            now_ms=2000,
        )

        rows = (
            archive._ensure_source_conn()
            .execute("SELECT raw_id, computed_at_ms FROM raw_authority_verdicts WHERE logical_source_key = 'codex:s1'")
            .fetchall()
        )

    assert {str(r[0]) for r in rows} == {"oldest", "newest"}
    assert all(int(r[1]) == 2000 for r in rows)


def test_missing_cohort_returns_no_verdicts_and_no_cache_write(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        verdicts = get_or_compute_raw_authority_verdicts(archive, "codex:does-not-exist", now_ms=1000)

        rows = archive._ensure_source_conn().execute("SELECT COUNT(*) FROM raw_authority_verdicts").fetchall()

    assert verdicts == {}
    assert rows[0][0] == 0
