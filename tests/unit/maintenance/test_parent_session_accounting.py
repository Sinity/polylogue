"""Red twins for parent-session source/index accounting.

Every row this census reads is produced by a live production write route --
``ArchiveStore.write_raw_and_parsed_result`` (acquire + parse + index),
``write_raw_payload`` (acquire only), ``mark_raw_parse_failed``, and
``delete_sessions``.  The parent references themselves come from the writer's
own topology derivation inside ``write_parsed_session_to_archive``: a fixture
declares ``ParsedSession.parent_session_provider_id`` and the writer decides
whether a ``session_links`` row exists, what identity it carries, and whether
it resolves.  Nothing here hand-inserts a ``sessions``, ``session_links`` or
``raw_sessions`` row.

ANTI-VACUITY -- the mutation that must make these red is a PRODUCER change.
Stop ``write_parsed_session_to_archive`` from emitting parent edges (make the
``parser-parent`` ``_upsert_session_link`` call in ``write.py`` a no-op) and
every test in this module fails, because ``reference_total`` collapses to 0
and every disposition, blocking count and ``verify_archive`` status derived
from those references disappears.  The previous fixture hand-inserted
``session_links``/``sessions``/``raw_sessions`` rows through raw ``sqlite3``,
so that exact producer mutation left it green: it proved the counting SQL and
nothing about the route that has to feed it.

Two states are injected *past* the producer, and only because no writer
produces them -- they are the durable damage the census exists to report:

* ``source_unavailable`` -- a retained ``raw_sessions`` row whose bytes are
  gone.  No production route reaches it: every raw-deleting route
  (``cleanup_superseded_raw_snapshots``, ``apply_session_excision``) removes
  the ``raw_sessions`` row with the ref, and
  ``prune_orphan_blob_reference_debt`` explicitly skips any ref whose
  ``ref_id`` still has a ``raw_sessions`` row.  The fixture therefore removes
  the export file and the ``blob_refs`` row after acquisition, which is the
  loss event itself, not a fabricated census row.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.core.outcomes import OutcomeStatus
from polylogue.maintenance.archive_verification import ArchiveVerificationCheck, verify_archive
from polylogue.maintenance.parent_session_accounting import (
    ParentSessionAccountingReport,
    audit_parent_session_accounting,
)
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

#: Fixed acquisition instant. The census only ever reads ``parsed_at_ms`` as
#: NULL / not-NULL, so no assertion here depends on a clock; the constant keeps
#: the writer's ordering deterministic anyway.
_ACQUIRED_AT_MS = 1_700_000_000_000


def _parsed(
    native_id: str,
    *,
    provider: Provider = Provider.CODEX,
    parent: str | None = None,
) -> ParsedSession:
    """One minimal session; ``parent`` is the only parent assertion made."""
    return ParsedSession(
        source_name=provider,
        provider_session_id=native_id,
        parent_session_provider_id=parent,
        messages=[ParsedMessage(provider_message_id=f"{native_id}-m0", role=Role.USER, text=f"transcript {native_id}")],
    )


def _export_file(tmp_path: Path, native_id: str, payload: bytes) -> Path:
    exports = tmp_path / "exports"
    exports.mkdir(exist_ok=True)
    path = exports / f"{native_id}.jsonl"
    path.write_bytes(payload)
    return path


def _acquire_and_index(archive: ArchiveStore, tmp_path: Path, session: ParsedSession, *, order: int) -> str:
    """Acquire real bytes and index them through the production write route."""
    native_id = session.provider_session_id
    payload = f"transcript bytes for {native_id}".encode()
    path = _export_file(tmp_path, native_id, payload)
    return archive.write_raw_and_parsed_result(
        session,
        payload=payload,
        source_path=str(path),
        acquired_at_ms=_ACQUIRED_AT_MS + order,
    ).session_id


def _acquire_only(
    archive: ArchiveStore,
    tmp_path: Path,
    native_id: str,
    *,
    provider: Provider = Provider.CODEX,
    order: int,
) -> tuple[str, Path]:
    """Acquire raw bytes without indexing, as the acquire stage alone does."""
    payload = f"transcript bytes for {native_id}".encode()
    path = _export_file(tmp_path, native_id, payload)
    raw_id = archive.write_raw_payload(
        provider=provider,
        payload=payload,
        source_path=str(path),
        acquired_at_ms=_ACQUIRED_AT_MS + order,
        native_id=native_id,
    )
    return raw_id, path


def _lose_source_bytes(root: Path, raw_id: str, path: Path) -> None:
    """Simulate durable byte loss for one retained raw.

    Not a producer step: see the module docstring. The ``raw_sessions`` row
    stays exactly as the acquisition route wrote it; only the evidence it
    points at disappears.
    """
    path.unlink()
    conn = sqlite3.connect(root / "source.db")
    try:
        conn.execute("DELETE FROM blob_refs WHERE ref_id = ?", (raw_id,))
        conn.commit()
    finally:
        conn.close()


def _census_handles(root: Path) -> tuple[sqlite3.Connection, sqlite3.Connection]:
    """Open both tiers read-only, exactly as the archive check does."""
    source = sqlite3.connect(f"file:{root / 'source.db'}?mode=ro", uri=True)
    index = sqlite3.connect(f"file:{root / 'index.db'}?mode=ro", uri=True)
    return source, index


def _audit(root: Path) -> ParentSessionAccountingReport:
    source, index = _census_handles(root)
    try:
        return audit_parent_session_accounting(source, index, archive_root=root)
    finally:
        source.close()
        index.close()


def _accounting_check(root: Path) -> ArchiveVerificationCheck:
    check = verify_archive(
        root,
        checks=("parent-session-accounting",),
        index_path_override=root / "index.db",
    ).checks[0]
    assert isinstance(check, ArchiveVerificationCheck)
    return check


def test_parent_accounting_conserves_identity_and_explicit_unavailable_states(tmp_path: Path) -> None:
    """Three produced references; three distinct source-grounded dispositions."""
    root = tmp_path / "archive"
    with ArchiveStore(root) as archive:
        # materialized: the parent is acquired, parsed and indexed first, and
        # the writer resolves the child's edge onto it.
        _acquire_and_index(archive, tmp_path, _parsed("parent", provider=Provider.CLAUDE_CODE), order=0)
        _acquire_and_index(
            archive,
            tmp_path,
            _parsed("claude-child", provider=Provider.CLAUDE_CODE, parent="parent"),
            order=1,
        )
        # source_unavailable: acquired, never indexed, then the bytes are lost.
        gone_raw_id, gone_path = _acquire_only(archive, tmp_path, "gone", order=2)
        _acquire_and_index(archive, tmp_path, _parsed("gone-child", parent="gone"), order=3)
        # not_acquired: the parser asserts a parent that was never acquired.
        _acquire_and_index(archive, tmp_path, _parsed("orphan-child", parent="never-acquired"), order=4)
        archive.commit()
    _lose_source_bytes(root, gone_raw_id, gone_path)

    report = _audit(root)

    assert report.available
    assert report.reference_total == 3
    assert report.unique_parent_total == 3
    assert report.materialized_parent_total == 1
    assert report.source_unavailable_total == 1
    assert report.not_acquired_total == 1
    assert report.available_unmaterialized_total == 0
    assert {entry.disposition for entry in report.references} == {
        "materialized",
        "source_unavailable",
        "not_acquired",
    }
    # The producer, not the fixture, decided which identity each edge carries.
    assert {(entry.origin, entry.native_id) for entry in report.references} == {
        ("claude-code-session", "parent"),
        ("codex-session", "gone"),
        ("codex-session", "never-acquired"),
    }


def test_parent_identity_never_matches_native_id_from_another_origin(tmp_path: Path) -> None:
    """A Codex reference must not bind a Claude raw that shares its native id."""
    root = tmp_path / "archive"
    with ArchiveStore(root) as archive:
        # Same native id, different origin -- acquired and indexed for real.
        _acquire_and_index(archive, tmp_path, _parsed("same-id", provider=Provider.CLAUDE_CODE), order=0)
        _acquire_and_index(archive, tmp_path, _parsed("codex-child", parent="same-id"), order=1)
        archive.commit()

    report = _audit(root)

    assert report.reference_total == 1
    assert report.references[0].origin == "codex-session"
    assert report.references[0].native_id == "same-id"
    assert report.not_acquired_total == 1
    assert report.materialized_parent_total == 0
    assert report.references[0].source_raw_ids == ()
    assert report.references[0].indexed_session_ids == ()


def test_parsed_available_parent_without_candidate_session_is_blocking(tmp_path: Path) -> None:
    """A retained parsed parent raw with no candidate session is untyped debt."""
    root = tmp_path / "archive"
    with ArchiveStore(root) as archive:
        _acquire_and_index(archive, tmp_path, _parsed("missing"), order=0)
        _acquire_and_index(archive, tmp_path, _parsed("missing-child", parent="missing"), order=1)
        archive.commit()
    # Production index deletion: the rebuildable session row goes, the durable
    # parsed raw stays. That is the "parsed but never indexed" shape.
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        assert archive.delete_sessions(("codex-session:missing",)) == 1

    report = _audit(root)

    assert report.available_unmaterialized_total == 1
    assert report.blocking_count > 0
    assert report.raw_unexplained_total == 1
    assert report.raw_disposition_counts == {"untyped_unmaterialized": 1}
    assert [entry.native_id for entry in report.raw_dispositions] == ["missing"]

    check = _accounting_check(root)
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["untyped_total"] == 1
    # The denominator is the complete retained parent-origin raw population:
    # both the parent raw and the child raw the producer acquired.
    assert check.evidence["untyped_denominator"] == 2


def test_materialized_parent_with_unresolved_reference_is_blocking(tmp_path: Path) -> None:
    """A producer-quarantined edge onto an indexed session stays unresolved."""
    root = tmp_path / "archive"
    with ArchiveStore(root) as archive:
        # The writer's cycle guard quarantines a self-parent edge, so the
        # candidate session exists while its own reference never resolves.
        _acquire_and_index(archive, tmp_path, _parsed("self-parent", parent="self-parent"), order=0)
        archive.commit()

    report = _audit(root)

    assert report.materialized_parent_total == 0
    assert report.unresolved_reference_total == 1
    assert report.references[0].disposition == "materialized_unresolved"
    assert report.references[0].indexed_session_ids == ("codex-session:self-parent",)
    assert report.references[0].resolved_reference_count == 0
    assert report.blocking_count == 1

    check = _accounting_check(root)
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["unresolved_reference_total"] == 1


def test_unmaterialized_parent_raw_with_parse_refusal_is_conserved(tmp_path: Path) -> None:
    """A durable parser refusal types the parent raw instead of leaving debt."""
    root = tmp_path / "archive"
    with ArchiveStore(root) as archive:
        refused_raw_id, _ = _acquire_only(archive, tmp_path, "refused", order=0)
        archive.mark_raw_parse_failed(
            refused_raw_id,
            provider=Provider.CODEX,
            error=ValueError("malformed JSON"),
        )
        _acquire_and_index(archive, tmp_path, _parsed("refused-child", parent="refused"), order=1)
        archive.commit()

    report = _audit(root)

    assert report.available_unmaterialized_total == 1
    assert report.raw_unexplained_total == 0
    assert report.raw_disposition_counts == {"parse_failure": 1}
    assert report.raw_dispositions[0].raw_id == refused_raw_id
    assert report.blocking_count == 1
