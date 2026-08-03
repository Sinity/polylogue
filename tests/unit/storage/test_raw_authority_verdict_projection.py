"""Read-only RawAuthorityVerdict projection over a live ArchiveStore (polylogue-w6hql, Phase 2).

Anti-vacuity: every test here binds real ``raw_sessions`` rows through the
production writer (``ArchiveStore.write_raw_payload`` /
``ArchiveStore.bind_raw_revision``) and reads the real blob store through
``project_raw_authority_verdicts`` -- no mock or in-memory stand-in for the
archive is used. Deleting the module's blob-open call or replacing
``classify_historical_full_revision_streams`` with a stub would make these
tests fail (see ``test_verdicts_match_direct_classification_of_the_same_bytes``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.revision_authority import (
    RawRevisionAuthority,
    RawRevisionEnvelope,
    RawRevisionKind,
)
from polylogue.core.enums import Provider, RawAuthorityVerdict
from polylogue.storage.raw_authority_verdict_projection import project_raw_authority_verdicts
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


def test_sole_raw_projects_sole_copy(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        _bind_full(archive, raw_id="only", payload=b"payload", logical_source_key="codex:s1")

        verdicts = project_raw_authority_verdicts(archive, "codex:s1")

    assert verdicts == {"only": RawAuthorityVerdict.SOLE_COPY}


def test_proven_chain_projects_head_verified_ancestor_superseded(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        _bind_full(archive, raw_id="oldest", payload=b"one\n", logical_source_key="codex:s1")
        _bind_full(archive, raw_id="newest", payload=b"one\ntwo\n", logical_source_key="codex:s1")

        verdicts = project_raw_authority_verdicts(archive, "codex:s1")

    assert verdicts == {
        "oldest": RawAuthorityVerdict.SUPERSEDED,
        "newest": RawAuthorityVerdict.VERIFIED,
    }


def test_unprovable_fork_projects_diverged(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        _bind_full(archive, raw_id="left", payload=b"left", logical_source_key="codex:s1")
        _bind_full(archive, raw_id="right", payload=b"right", logical_source_key="codex:s1")

        verdicts = project_raw_authority_verdicts(archive, "codex:s1")

    assert verdicts == {
        "left": RawAuthorityVerdict.DIVERGED,
        "right": RawAuthorityVerdict.DIVERGED,
    }


def test_unresolved_kind_projects_unchecked_alongside_a_proven_sibling(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        _bind_full(archive, raw_id="proven", payload=b"payload", logical_source_key="codex:s1")
        unresolved_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"not yet classified",
            source_path="session.jsonl",
            acquired_at_ms=1,
            raw_id="pending",
        )
        archive.bind_raw_revision(
            unresolved_id,
            RawRevisionEnvelope("codex:s1", RawRevisionKind.UNKNOWN, "revision-pending", 0),
        )

        verdicts = project_raw_authority_verdicts(archive, "codex:s1")

    assert verdicts == {
        "proven": RawAuthorityVerdict.SOLE_COPY,
        "pending": RawAuthorityVerdict.UNCHECKED,
    }


def test_append_kind_in_the_cohort_is_out_of_scope_this_phase(tmp_path: Path) -> None:
    """Deferred scope (see module docstring): raising loudly beats a silently
    wrong verdict for a proof shape this phase does not attempt to collapse."""
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        _bind_full(archive, raw_id="baseline", payload=b"one\n", logical_source_key="codex:s1")
        append_raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b"suffix",
            source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=1,
        )
        archive.bind_raw_revision(
            append_raw_id,
            RawRevisionEnvelope(
                "codex:s1",
                RawRevisionKind.APPEND,
                "revision-append",
                0,
                predecessor_source_revision="revision-baseline",
                append_start_offset=4,
                append_end_offset=10,
                authority=RawRevisionAuthority.QUARANTINED,
            ),
        )

        with pytest.raises(NotImplementedError, match="revision_kind='full'"):
            project_raw_authority_verdicts(archive, "codex:s1")


def test_missing_cohort_projects_no_verdicts(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        verdicts = project_raw_authority_verdicts(archive, "codex:does-not-exist")

    assert verdicts == {}


def test_verdicts_match_direct_classification_of_the_same_bytes(tmp_path: Path) -> None:
    """Prove the projection reuses real governance evidence, not a shortcut.

    Binds the exact cohort the real classifier
    (``ArchiveStore.classify_raw_revision_cohort``) already has coverage for
    elsewhere, and checks the projection's verdict for the proven head agrees
    with what governance itself would persist (``revision_authority=
    'byte_proven'`` with no successor -- ``VERIFIED``), rather than merely
    re-asserting the pure function's own logic against itself.
    """
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        _bind_full(archive, raw_id="oldest", payload=b"one\n", logical_source_key="codex:s1")
        _bind_full(archive, raw_id="newest", payload=b"one\ntwo\n", logical_source_key="codex:s1")

        plan = archive.classify_raw_revision_cohort_for_live_watch("codex:s1")
        verdicts = project_raw_authority_verdicts(archive, "codex:s1")

        row = (
            archive._ensure_source_conn()
            .execute("SELECT raw_id, revision_authority FROM raw_sessions WHERE logical_source_key = 'codex:s1'")
            .fetchall()
        )

    persisted_authority = {str(r[0]): str(r[1]) for r in row}
    assert persisted_authority == {"oldest": "byte_proven", "newest": "byte_proven"}
    assert plan is not None
    assert verdicts == {
        "oldest": RawAuthorityVerdict.SUPERSEDED,
        "newest": RawAuthorityVerdict.VERIFIED,
    }
