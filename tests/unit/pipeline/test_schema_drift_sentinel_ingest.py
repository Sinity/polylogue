"""Record-level drift reduction follows the retained publication route."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider, ValidationMode, ValidationStatus
from polylogue.schemas.packages import SchemaResolution
from polylogue.schemas.retained_validation import RetainedValidationVerdict
from polylogue.schemas.validator_resolution import canonical_provider
from tests.infra.archive_templates import bootstrap_archive_root, run_archive_fixture_write
from tests.infra.live_ingest import prepared_live_convergence_owner


@pytest.mark.parametrize("benign_first", [True, False])
@pytest.mark.parametrize("mode", [ValidationMode.ADVISORY, ValidationMode.STRICT])
def test_retained_validation_keeps_later_type_failure_over_new_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    benign_first: bool,
    mode: ValidationMode,
) -> None:
    """A benign new-field sample cannot hide a later incompatible value."""
    from polylogue.schemas import retained_validation

    schema = {
        "type": "object",
        "x-polylogue-sample-granularity": "record",
        "properties": {
            "type": {"type": "string"},
            "payload": {
                "type": "object",
                "properties": {"known": {"type": "string"}},
                "required": ["known"],
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
        reason="exact_structure",
    )

    def resolve(
        provider: str | Provider,
        _payload: object,
        **_kwargs: object,
    ) -> tuple[Provider, dict[str, object], tuple[str, str, str], SchemaResolution]:
        return canonical_provider(provider), schema, ("codex", "test-v1", "session_record_stream"), resolution

    monkeypatch.setattr(retained_validation, "resolve_retained_schema", resolve)
    real_validate = retained_validation.validate_retained_document
    verdicts: list[RetainedValidationVerdict] = []

    def validate(
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
            signature_directory=path.parent,
        )
        verdicts.append(verdict)
        return verdict

    monkeypatch.setattr("polylogue.schemas.validate_retained_document", validate)
    benign = {"type": "session_meta", "payload": {"id": "drift-order", "known": "ok", "new_provider_field": "x"}}
    changed = {
        "type": "response_item",
        "payload": {
            "type": "message",
            "id": "m1",
            "role": "user",
            "content": [{"type": "input_text", "text": "kept content"}],
            "known": 7,
        },
    }
    records = [benign, changed] if benign_first else [changed, benign]
    payload = b"".join(json.dumps(record).encode() + b"\n" for record in records)
    archive = tmp_path / "archive"

    def acquire() -> str:
        bootstrap_archive_root(archive)
        from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

        with ArchiveStore.open_existing(archive, read_only=False) as store:
            return store.write_raw_payload(
                provider=Provider.CODEX,
                payload=payload,
                source_path="drift-order.jsonl",
                canonical_source_path="drift-order.jsonl",
                acquired_at_ms=1,
            )

    raw_id = asyncio.run(run_archive_fixture_write(archive, acquire))

    async def replay() -> None:
        async with prepared_live_convergence_owner(archive, validation_mode=mode) as owner:
            await owner.converge_raw_id(raw_id)

    asyncio.run(replay())
    assert len(verdicts) == 1
    assert verdicts[0].strict_refusal is (mode is ValidationMode.STRICT)
    assert verdicts[0].status is (ValidationStatus.FAILED if mode is ValidationMode.STRICT else ValidationStatus.PASSED)
    assert verdicts[0].drift_observation is not None
    assert verdicts[0].drift_observation.classification == "field_changed"
    with sqlite3.connect(archive / "ops.db") as conn:
        classification = conn.execute(
            "SELECT classification FROM schema_drift_samples WHERE raw_id=?", (raw_id,)
        ).fetchone()
    assert classification == ("field_changed",)
    with sqlite3.connect(archive / "index.db") as conn:
        if mode is ValidationMode.STRICT:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        else:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
