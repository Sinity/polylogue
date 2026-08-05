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

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider, RawAuthorityVerdict
from polylogue.storage import raw_authority_verdict_cache as cache_module
from polylogue.storage.raw_authority_verdict_cache import (
    RawAuthorityVerdictCacheWarmup,
    RawAuthorityVerdictCacheWork,
    find_raw_authority_verdict_cache_work,
    get_or_compute_raw_authority_verdicts,
    read_cached_raw_authority_verdicts,
    warm_raw_authority_verdict_cache,
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


def test_changed_content_fingerprint_invalidates_the_cache(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        _bind_full(archive, raw_id="oldest", payload=b"one\n", logical_source_key="codex:s1")
        _bind_full(archive, raw_id="newest", payload=b"one\ntwo\n", logical_source_key="codex:s1")
        _bind_full(archive, raw_id="replacement", payload=b"diverged", logical_source_key="codex:replacement")

        first = get_or_compute_raw_authority_verdicts(archive, "codex:s1", now_ms=1000)
        assert first == {
            "oldest": RawAuthorityVerdict.SUPERSEDED,
            "newest": RawAuthorityVerdict.VERIFIED,
        }

        conn = archive._ensure_source_conn()
        conn.execute(
            """
            UPDATE raw_sessions
            SET blob_hash = (SELECT blob_hash FROM raw_sessions WHERE raw_id = 'replacement'),
                blob_size = (SELECT blob_size FROM raw_sessions WHERE raw_id = 'replacement')
            WHERE raw_id = 'newest'
            """
        )
        conn.commit()

        assert read_cached_raw_authority_verdicts(archive, "codex:s1") is None
        second = get_or_compute_raw_authority_verdicts(archive, "codex:s1", now_ms=2000)

    assert second == {
        "oldest": RawAuthorityVerdict.DIVERGED,
        "newest": RawAuthorityVerdict.DIVERGED,
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


def test_warmup_is_bounded_and_skips_append_cohorts(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        _bind_full(archive, raw_id="full-oldest", payload=b"one\n", logical_source_key="codex:full")
        _bind_full(archive, raw_id="full-newest", payload=b"one\ntwo\n", logical_source_key="codex:full")
        _bind_full(archive, raw_id="full-single", payload=b"single", logical_source_key="codex:full-single")
        append_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"append-payload",
            source_path="session.jsonl",
            acquired_at_ms=1,
            raw_id="append-only",
        )
        archive.bind_raw_revision(
            append_id,
            RawRevisionEnvelope(
                "codex:append",
                RawRevisionKind.APPEND,
                "revision-append",
                0,
                predecessor_source_revision="revision-base",
                predecessor_raw_id="base",
                baseline_raw_id="base",
                append_start_offset=0,
                append_end_offset=1,
                authority=RawRevisionAuthority.ASSERTED,
            ),
        )

        work = find_raw_authority_verdict_cache_work(archive._ensure_source_conn(), max_cohorts=1)
        assert isinstance(work, RawAuthorityVerdictCacheWork)
        assert work.pending_logical_source_keys == ("codex:full",)
        assert work.skipped_append_cohorts == 1

        first = warm_raw_authority_verdict_cache(archive, max_cohorts=1, now_ms=1000)
        assert isinstance(first, RawAuthorityVerdictCacheWarmup)
        assert first.warmed_cohorts == 1
        assert first.pending_cohorts is True
        assert first.skipped_append_cohorts == 1

        second = warm_raw_authority_verdict_cache(archive, max_cohorts=1, now_ms=2000)
        assert second.warmed_cohorts == 1
        assert second.pending_cohorts is False
        assert second.skipped_append_cohorts == 1

        cached_keys = {
            str(row[0])
            for row in archive._ensure_source_conn()
            .execute("SELECT logical_source_key FROM raw_authority_verdicts")
            .fetchall()
        }
        cached_verdicts = {
            key: read_cached_raw_authority_verdicts(archive, key) for key in ("codex:full", "codex:full-single")
        }
        assert read_cached_raw_authority_verdicts(archive, "codex:append") is None

    assert cached_keys == {"codex:full", "codex:full-single"}
    assert cached_verdicts == {
        "codex:full": {
            "full-oldest": RawAuthorityVerdict.SUPERSEDED,
            "full-newest": RawAuthorityVerdict.VERIFIED,
        },
        "codex:full-single": {"full-single": RawAuthorityVerdict.SOLE_COPY},
    }


def test_warmup_does_not_recompute_fresh_cohorts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        _bind_full(archive, raw_id="only", payload=b"payload", logical_source_key="codex:s1")
        warm_raw_authority_verdict_cache(archive, max_cohorts=1, now_ms=1000)

        def _fail(*args: object, **kwargs: object) -> dict[str, RawAuthorityVerdict]:
            raise AssertionError("a fresh cohort must not invoke the projection")

        monkeypatch.setattr(cache_module, "project_raw_authority_verdicts", _fail)

        outcome = warm_raw_authority_verdict_cache(archive, max_cohorts=1, now_ms=2000)

    assert outcome.warmed_cohorts == 0
    assert outcome.pending_cohorts is False
