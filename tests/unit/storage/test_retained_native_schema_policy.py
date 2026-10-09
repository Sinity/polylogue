"""Native retained input carries its captured schema policy through Source."""

from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from polylogue.archive.revision_authority import raw_authority_parser_fingerprint
from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.enums import Provider
from polylogue.operations.raw_observation_derivation import make_raw_observation_derivation, raw_observation_frame
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.retained_parser_payloads import _antigravity_trajectory_db_bytes
from tests.unit.storage.test_raw_observation_derivation import (
    _fixture_archive,
    _publish_to_valid,
    _run,
    _run_raw_law,
    _snapshot,
)


def _native_schema_policy_law(tmp_path: Path, *, current_non_session: bool, has_messages: bool) -> None:

    def exercise(compute_adapter: BoundedComputeAdapter) -> None:
        bootstrap_archive_root(tmp_path)
        payload = _antigravity_trajectory_db_bytes(tmp_path)
        source = tmp_path / "trajectory.db"
        if not has_messages:
            from polylogue.sources.sqlite_export import logical_export_bytes

            with sqlite3.connect(source) as conn:
                conn.executescript(
                    "DELETE FROM trajectory_meta; DELETE FROM steps; DELETE FROM conversation_summaries; DELETE FROM parent_references;"
                )
            payload = logical_export_bytes(source)
        with _fixture_archive(tmp_path) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.ANTIGRAVITY,
                payload=payload,
                source_path=str(source),
                canonical_source_path=str(source),
                acquired_at_ms=1,
            )
        source.unlink()
        if current_non_session:
            fingerprint = raw_authority_parser_fingerprint()
            with sqlite3.connect(tmp_path / "source.db") as conn:
                conn.execute(
                    "INSERT INTO raw_authority_parser_census(raw_id,parser_fingerprint,status,logical_keys_json) "
                    "VALUES (?,?,'complete','[]')",
                    (raw_id, fingerprint),
                )
                conn.execute(
                    "INSERT INTO raw_membership_census(raw_id,parser_fingerprint,status,member_count,censused_at_ms) "
                    "VALUES (?,?,'non_session',0,0)",
                    (raw_id, fingerprint),
                )
                conn.execute("UPDATE raw_sessions SET parsed_at_ms=0 WHERE raw_id=?", (raw_id,))
        adapter = make_raw_observation_derivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        replacement = adapter.compute(frame, raw_id, replay_current=True)
        published_index = _publish_to_valid(adapter, frame, replacement)
        assert published_index is has_messages
        assert adapter.inspect(frame, (raw_id,)) == {raw_id: "valid"}
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert conn.execute(
                "SELECT parse_as_session,schema_eligible FROM raw_artifacts WHERE raw_id=?", (raw_id,)
            ).fetchall() == [(1, 0)]
            assert conn.execute(
                "SELECT validation_status,validation_mode FROM raw_sessions WHERE raw_id=?", (raw_id,)
            ).fetchone() == (None, None)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone() == (int(has_messages),)
            assert conn.execute("SELECT text FROM blocks").fetchall() == ([("retained text",)] if has_messages else [])
        settled_source = _snapshot(tmp_path)
        assert _run(tmp_path, compute_adapter=compute_adapter).made_no_publication_attempts
        assert _snapshot(tmp_path) == settled_source

    _run_raw_law(tmp_path, exercise)


@pytest.mark.parametrize("current_non_session", [False, True])
def test_native_sqlite_captured_schema_policy_reaches_current_index(tmp_path: Path, current_non_session: bool) -> None:
    """Native session replay publishes its actual captured schema policy."""
    _native_schema_policy_law(tmp_path, current_non_session=current_non_session, has_messages=True)


def test_empty_native_sqlite_current_census_carries_schema_policy(tmp_path: Path) -> None:
    """A native parser with no sessions still owes its captured schema policy."""
    _native_schema_policy_law(tmp_path, current_non_session=True, has_messages=False)


def test_eligible_json_prepared_carrier_missing_verdict_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A coarse false taxonomy cannot give JSON a native grammar exemption."""
    from polylogue.sources import revision_backfill
    from polylogue.sources.prepared_jsonl import PreparedJsonl
    from tests.unit.storage.test_raw_observation_derivation import _admit

    def exercise(compute_adapter: BoundedComputeAdapter) -> None:
        bootstrap_archive_root(tmp_path)
        raw_id = _admit(tmp_path, ("json-policy-required",))
        assert _run(tmp_path, compute_adapter=compute_adapter).failed == 0
        with sqlite3.connect(tmp_path / "source.db") as conn:
            conn.execute(
                "UPDATE raw_sessions SET validated_at_ms=NULL,validation_status=NULL,validation_error=NULL,"
                "validation_mode=NULL,validation_drift_count=0 WHERE raw_id=?",
                (raw_id,),
            )
        prepare = revision_backfill.prepare_retained_jsonl_artifact

        def lose_verdict(*args: Any, **kwargs: Any) -> PreparedJsonl:
            artifact = prepare(*args, **kwargs)
            stream = artifact.stream_classification()
            assert stream is not None and stream.classification.parse_as_session
            assert not stream.classification.schema_eligible
            assert artifact.validation_verdict is not None
            return dataclasses.replace(artifact, validation_verdict=None)

        monkeypatch.setattr(revision_backfill, "prepare_retained_jsonl_artifact", lose_verdict)
        adapter = make_raw_observation_derivation(tmp_path, compute_adapter=compute_adapter)
        frame = raw_observation_frame(tmp_path)
        with pytest.raises(revision_backfill.RetainedPreparationRetryableError):
            unexpected = adapter.compute(frame, raw_id, replay_current=True)
            unexpected.close()
        with sqlite3.connect(tmp_path / "source.db") as conn:
            assert conn.execute(
                "SELECT validation_status,validation_mode FROM raw_sessions WHERE raw_id=?", (raw_id,)
            ).fetchone() == (None, None)

    _run_raw_law(tmp_path, exercise)
