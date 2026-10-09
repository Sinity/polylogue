"""Schema drift telemetry follows committed retained-writer outcomes."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider, ValidationMode
from polylogue.daemon.derivation import DerivationReport
from polylogue.schemas.drift_sentinel import SchemaDriftObservation
from polylogue.schemas.packages import SchemaResolution, SchemaResolutionReason
from polylogue.schemas.retained_validation import RetainedValidationVerdict
from polylogue.schemas.validator_resolution import canonical_provider
from polylogue.storage.sqlite.archive_tiers.ops_write import iter_schema_drift_signature
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner


def _payload() -> bytes:
    return (
        b'{"type":"session_meta","payload":{"id":"drift-route",'
        b'"metadata":{"operatorNote":"private-value"}}}\n'
        b'{"type":"response_item","payload":{"type":"message","id":"m1",'
        b'"role":"user","content":[{"type":"input_text","text":"known prompt"}]}}\n'
    )


@pytest.mark.parametrize("mode", [ValidationMode.ADVISORY, ValidationMode.STRICT, ValidationMode.OFF])
@pytest.mark.parametrize("telemetry_fails", [False, True])
def test_retained_schema_drift_telemetry_follows_replay_and_never_changes_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: ValidationMode,
    telemetry_fails: bool,
) -> None:
    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)

    def acquire() -> str:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_payload(),
                source_path="drift-route.jsonl",
                canonical_source_path="drift-route.jsonl",
                acquired_at_ms=1,
            )

    observations: list[SchemaDriftObservation] = []
    verdicts: list[RetainedValidationVerdict] = []
    resolution_reason: SchemaResolutionReason = "exact_structure"
    # Use the real spill-backed validator and verdict emitter against a
    # controlled exact schema. Its required `kind` is missing from both
    # records, while the raw also carries a value under an unknown field.
    # This makes the exact same ordinary verdict ADVISORY-valid and
    # STRICT-refusing, with field paths but no private field value in drift.
    from polylogue.schemas import retained_validation

    schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string"},
            "payload": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "type": {"type": "string"}},
                "required": ["kind"],
                "additionalProperties": True,
            },
        },
        "required": ["type", "payload"],
        "additionalProperties": True,
    }
    resolution = SchemaResolution(
        provider="codex",
        package_version="test-v1",
        element_kind="session_record_stream",
        exact_structure_id="codex-record-v1",
        bundle_scope=None,
        reason=resolution_reason,
    )
    real_validate = retained_validation.validate_retained_document

    def resolve_test_schema(
        provider: str | Provider,
        _payload: object,
        **_kwargs: object,
    ) -> tuple[Provider, dict[str, object], tuple[str, str, str], SchemaResolution]:
        return canonical_provider(provider), schema, ("codex", "test-v1", "session_record_stream"), resolution

    monkeypatch.setattr(retained_validation, "resolve_retained_schema", resolve_test_schema)

    def validate_retained_document(
        provider: str | Provider,
        path: Path,
        *,
        mode: ValidationMode,
        raw_id: str,
        revision_sha256: str,
        evidence_id: str,
        **kwargs: object,
    ) -> RetainedValidationVerdict:
        verdict = real_validate(
            provider,
            path,
            mode=mode,
            raw_id=raw_id,
            revision_sha256=revision_sha256,
            evidence_id=evidence_id,
            source_path=str(kwargs.get("source_path") or ""),
            jsonl=bool(kwargs.get("jsonl", False)),
            schema_resolution=resolution,
            schema_resolution_is_explicit=True,
        )
        verdicts.append(verdict)
        if verdict.drift_observation is not None:
            observations.append(verdict.drift_observation)
        return verdict

    monkeypatch.setattr("polylogue.schemas.validate_retained_document", validate_retained_document)
    recorder_calls: list[tuple[Path, list[SchemaDriftObservation], Path | None]] = []
    if telemetry_fails:

        def record_observations(
            index_db_path: Path,
            observations: list[SchemaDriftObservation],
            *,
            archive_root: Path | None = None,
        ) -> int:
            recorder_calls.append((index_db_path, observations, archive_root))
            raise sqlite3.OperationalError("synthetic ops.db write failure")

        monkeypatch.setattr(
            "polylogue.schemas.drift_sentinel_sampling.record_schema_drift_observations_to_ops_sync",
            record_observations,
        )

    async def settle() -> DerivationReport:
        raw_id = await run_archive_fixture_write(archive_root, acquire)
        async with prepared_live_convergence_owner(archive_root) as owner:
            owner._validation_mode = mode
            owner._archive._validation_mode = mode
            result = await owner.converge_raw_id(raw_id)
            if mode is ValidationMode.STRICT:
                await owner.converge_raw_id(raw_id)
            return result

    result = asyncio.run(settle())
    assert len(observations) == int(mode is not ValidationMode.OFF)
    if mode is not ValidationMode.OFF:
        assert observations[0].classification == "field_changed"
        assert observations[0].unseen_key_signature
    with sqlite3.connect(archive_root / "index.db") as conn:
        session_row = conn.execute(
            "SELECT raw_id, message_count FROM sessions WHERE native_id = 'drift-route'"
        ).fetchone()
        message_count = conn.execute(
            """SELECT COUNT(*) FROM messages AS message
               JOIN sessions AS session USING (session_id)
               WHERE session.native_id = 'drift-route'"""
        ).fetchone()[0]
    if mode is ValidationMode.OFF:
        assert result.failed == 0
        assert recorder_calls == []
        assert session_row is not None and message_count > 0
        return

    if telemetry_fails:
        assert len(recorder_calls) == 1
        _, persisted, routed_archive_root = recorder_calls[0]
        assert persisted == observations
        assert routed_archive_root == archive_root
    else:
        with sqlite3.connect(archive_root / "ops.db") as conn:
            count = conn.execute("SELECT COUNT(*) FROM schema_drift_samples").fetchone()[0]
            row = conn.execute(
                """SELECT sample_id, origin, element_kind, classification, signature_byte_count,
                          native_id_example, raw_id
                   FROM schema_drift_samples"""
            ).fetchone()
        assert count == 1
        assert row == (
            row[0],
            "codex-session",
            observations[0].element_kind,
            observations[0].classification,
            observations[0].unseen_key_signature.byte_count,
            observations[0].native_id_example,
            observations[0].raw_id,
        )
        signature = b"".join(iter_schema_drift_signature(conn, row[0]))
        assert signature == b"".join(observations[0].unseen_key_signature.iter_utf8_chunks())
        assert "private-value" not in repr(row)
        persisted = observations
    # Only the classified signature and stable IDs leave the verdict; the
    # input's unknown field value is not carried by SchemaDriftObservation.
    assert persisted[0].unseen_key_signature == observations[0].unseen_key_signature
    assert persisted[0].raw_id == observations[0].raw_id
    assert "private-value" not in repr(persisted)
    if mode is ValidationMode.STRICT:
        assert verdicts[0].strict_refusal
        assert session_row is None
        assert message_count == 0
    else:
        assert result.failed == 0
        assert session_row is not None
        assert session_row[0] == observations[0].raw_id
        assert message_count > 0


def test_strict_retained_historical_schema_verdict_reaches_session_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy Claude record accepted by v1 remains publishable on the daemon route."""
    import json
    from functools import partial

    from polylogue.core.enums import ValidationStatus
    from polylogue.schemas.registry import SchemaRegistry

    archive_root = tmp_path / "archive"
    bootstrap_archive_root(archive_root)
    schema_root = tmp_path / "schemas"
    registry = SchemaRegistry(storage_root=schema_root)

    def schema(content_type: str) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "type": {"const": "assistant"},
                "message": {
                    "type": "object",
                    "properties": {
                        "role": {"const": "assistant"},
                        "content": {"type": content_type},
                    },
                    "required": ["role", "content"],
                    "additionalProperties": True,
                },
            },
            "required": ["type", "message"],
            "additionalProperties": True,
        }

    registry.write_schema_version("claude-code", "v1", schema("array"), element_kind="session_record_stream")
    registry.write_schema_version("claude-code", "v2", schema("string"), element_kind="session_record_stream")
    monkeypatch.setattr(
        "polylogue.schemas.retained_validation.SchemaRegistry", partial(SchemaRegistry, storage_root=schema_root)
    )
    from polylogue.schemas.validator import validate_retained_document as real_validate_retained_document

    verdicts: list[RetainedValidationVerdict] = []

    def capture_verdict(*args: object, **kwargs: object) -> RetainedValidationVerdict:
        verdict = real_validate_retained_document(*args, **kwargs)  # type: ignore[arg-type]
        verdicts.append(verdict)
        return verdict

    monkeypatch.setattr("polylogue.schemas.validate_retained_document", capture_verdict)

    legacy_record = {
        "type": "assistant",
        "uuid": "legacy-assistant",
        "sessionId": "legacy-schema-route",
        "timestamp": "2026-07-01T10:00:01Z",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "legacy record accepted by v1"}],
        },
    }
    payload = (json.dumps(legacy_record) + "\n").encode()

    def acquire() -> str:
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CLAUDE_CODE,
                payload=payload,
                source_path="projects/legacy-schema-route/session.jsonl",
                canonical_source_path="projects/legacy-schema-route/session.jsonl",
                acquired_at_ms=1,
            )

    raw_id = asyncio.run(run_archive_fixture_write(archive_root, acquire))

    async def converge() -> DerivationReport:
        async with prepared_live_convergence_owner(archive_root, validation_mode=ValidationMode.STRICT) as owner:
            return await owner.converge_raw_id(raw_id)

    report = asyncio.run(converge())
    with sqlite3.connect(archive_root / "source.db") as source:
        validation = source.execute(
            "SELECT validation_status, validation_mode FROM raw_sessions WHERE raw_id = ?", (raw_id,)
        ).fetchone()
    with sqlite3.connect(archive_root / "index.db") as index:
        session = index.execute("SELECT native_id FROM sessions WHERE native_id = 'legacy-schema-route'").fetchone()
        message_count = index.execute(
            "SELECT COUNT(*) FROM messages AS message JOIN sessions AS session USING (session_id) "
            "WHERE session.native_id = 'legacy-schema-route'"
        ).fetchone()[0]

    assert report.failed == 0
    assert len(verdicts) == 1
    assert verdicts[0].schema_resolution is not None
    assert verdicts[0].schema_resolution.package_version == "v1"
    assert not verdicts[0].strict_refusal
    assert validation == (ValidationStatus.PASSED.value, ValidationMode.STRICT.value)
    assert session == ("legacy-schema-route",)
    assert message_count == 1
