"""Direct tests for polylogue.schemas.verification module.

Covers ProviderSchemaVerification serialization, SchemaVerificationReport,
and verify_raw_corpus with nonexistent DB and provider filtering.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from polylogue.schemas.packages import SchemaElementManifest, SchemaResolution, SchemaVersionPackage
from polylogue.schemas.verification import (
    ArtifactProofReport,
    ProviderArtifactProof,
    ProviderSchemaVerification,
    SchemaVerificationReport,
    list_artifact_cohort_rows,
    list_artifact_observation_rows,
    prove_raw_artifact_coverage,
    verify_raw_corpus,
)
from polylogue.storage.backends.connection import open_connection


def _insert_raw_record(
    *,
    db_path: Path,
    raw_id: str,
    provider_name: str,
    payload_provider: str | None = None,
    source_name: str,
    source_path: str,
    raw_content: bytes,
) -> None:
    with open_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO raw_conversations (
                raw_id, provider_name, payload_provider, source_name, source_path, source_index,
                raw_content, acquired_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_id,
                provider_name,
                payload_provider,
                source_name,
                source_path,
                0,
                raw_content,
                datetime.now(tz=timezone.utc).isoformat(),
            ),
        )
        conn.commit()


class TestProviderSchemaVerification:
    def test_to_dict(self) -> None:
        v = ProviderSchemaVerification(
            provider="chatgpt",
            total_records=100,
            valid_records=90,
            invalid_records=5,
            drift_records=3,
            skipped_no_schema=0,
            decode_errors=2,
        )
        d = v.to_dict()
        assert d["provider"] == "chatgpt"
        assert d["total_records"] == 100
        assert d["valid_records"] == 90
        assert d["invalid_records"] == 5
        assert d["drift_records"] == 3
        assert d["decode_errors"] == 2
        assert d["skipped_no_schema"] == 0

    def test_to_dict_quarantined(self) -> None:
        v = ProviderSchemaVerification(
            provider="test",
            total_records=10,
            quarantined_records=2,
        )
        d = v.to_dict()
        assert d["quarantined_records"] == 2

    def test_default_zeros(self) -> None:
        v = ProviderSchemaVerification(provider="test")
        d = v.to_dict()
        assert d["total_records"] == 0
        assert d["valid_records"] == 0
        assert d["invalid_records"] == 0
        assert d["decode_errors"] == 0
        assert d["quarantined_records"] == 0

    def test_partial_specification(self) -> None:
        v = ProviderSchemaVerification(
            provider="partial",
            total_records=50,
            valid_records=40,
        )
        d = v.to_dict()
        assert d["total_records"] == 50
        assert d["valid_records"] == 40
        assert d["invalid_records"] == 0  # default


class TestSchemaVerificationReport:
    def test_to_dict_structure(self) -> None:
        report = SchemaVerificationReport(
            providers={
                "chatgpt": ProviderSchemaVerification(provider="chatgpt", total_records=10),
            },
            max_samples=None,
            total_records=10,
        )
        d = report.to_dict()
        assert d["max_samples"] == "all"
        assert "chatgpt" in d["providers"]
        assert d["total_records"] == 10
        assert d["record_limit"] == "all"
        assert d["record_offset"] == 0

    def test_to_dict_with_limits(self) -> None:
        report = SchemaVerificationReport(
            providers={},
            max_samples=5,
            total_records=0,
            record_limit=100,
            record_offset=10,
        )
        d = report.to_dict()
        assert d["max_samples"] == 5
        assert d["record_limit"] == 100
        assert d["record_offset"] == 10

    def test_to_dict_multiple_providers(self) -> None:
        report = SchemaVerificationReport(
            providers={
                "chatgpt": ProviderSchemaVerification(provider="chatgpt", total_records=10),
                "claude-ai": ProviderSchemaVerification(provider="claude-ai", total_records=20),
            },
            max_samples=None,
            total_records=30,
        )
        d = report.to_dict()
        assert len(d["providers"]) == 2
        assert "chatgpt" in d["providers"]
        assert "claude-ai" in d["providers"]

    def test_to_dict_sorts_providers(self) -> None:
        report = SchemaVerificationReport(
            providers={
                "zebra": ProviderSchemaVerification(provider="zebra", total_records=1),
                "alpha": ProviderSchemaVerification(provider="alpha", total_records=1),
                "beta": ProviderSchemaVerification(provider="beta", total_records=1),
            },
            max_samples=None,
            total_records=3,
        )
        d = report.to_dict()
        provider_names = list(d["providers"].keys())
        assert provider_names == ["alpha", "beta", "zebra"]

    def test_max_samples_none_displays_as_all(self) -> None:
        report = SchemaVerificationReport(
            providers={},
            max_samples=None,
            total_records=0,
        )
        d = report.to_dict()
        assert d["max_samples"] == "all"

    def test_record_limit_none_displays_as_all(self) -> None:
        report = SchemaVerificationReport(
            providers={},
            max_samples=None,
            total_records=0,
            record_limit=None,
        )
        d = report.to_dict()
        assert d["record_limit"] == "all"


class TestProviderArtifactProof:
    def test_to_dict(self) -> None:
        report = ProviderArtifactProof(
            provider="claude-code",
            total_records=4,
            contract_backed_records=1,
            unsupported_parseable_records=1,
            recognized_non_parseable_records=1,
            unknown_records=1,
            decode_errors=1,
            artifact_counts={"agent_sidecar_meta": 1, "subagent_conversation_stream": 1},
            package_versions={"v1": 1},
            element_kinds={"conversation_record_stream": 1},
            resolution_reasons={"bundle_scope": 1},
            linked_sidecars=1,
            orphan_sidecars=0,
            subagent_streams=1,
            streams_with_sidecars=1,
            sidecar_agent_types={"general-purpose": 1},
        )

        data = report.to_dict()

        assert data["provider"] == "claude-code"
        assert data["contract_backed_records"] == 1
        assert data["linked_sidecars"] == 1
        assert data["sidecar_agent_types"] == {"general-purpose": 1}


class TestArtifactProofReport:
    def test_to_dict_structure(self) -> None:
        report = ArtifactProofReport(
            providers={
                "chatgpt": ProviderArtifactProof(
                    provider="chatgpt",
                    total_records=1,
                    contract_backed_records=1,
                    package_versions={"v1": 1},
                    element_kinds={"conversation_document": 1},
                    resolution_reasons={"exact_structure": 1},
                ),
            },
            total_records=1,
        )

        data = report.to_dict()

        assert data["total_records"] == 1
        assert data["record_limit"] == "all"
        assert data["summary"]["contract_backed_records"] == 1
        assert data["summary"]["package_versions"] == {"v1": 1}
        assert data["summary"]["element_kinds"] == {"conversation_document": 1}
        assert data["summary"]["resolution_reasons"] == {"exact_structure": 1}
        assert data["summary"]["clean"] is True


class TestVerifyRawCorpus:
    def test_nonexistent_db_returns_empty(self, tmp_path: Path) -> None:
        report = verify_raw_corpus(db_path=tmp_path / "nope.db")
        assert report.total_records == 0
        assert report.providers == {}

    def test_empty_db(self, tmp_path: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        db = tmp_path / "empty.db"
        with open_connection(db):
            pass
        report = verify_raw_corpus(db_path=db)
        assert report.total_records == 0
        assert report.providers == {}

    def test_provider_filter_empty(self, tmp_path: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        db = tmp_path / "empty.db"
        with open_connection(db):
            pass
        report = verify_raw_corpus(db_path=db, providers=["chatgpt"])
        assert report.total_records == 0

    def test_max_samples_preserved_in_report(self, tmp_path: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        db = tmp_path / "empty.db"
        with open_connection(db):
            pass
        report = verify_raw_corpus(db_path=db, max_samples=100)
        assert report.max_samples == 100

    def test_record_limit_preserved_in_report(self, tmp_path: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        db = tmp_path / "empty.db"
        with open_connection(db):
            pass
        report = verify_raw_corpus(db_path=db, record_limit=50)
        assert report.record_limit == 50

    def test_record_offset_preserved_in_report(self, tmp_path: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        db = tmp_path / "empty.db"
        with open_connection(db):
            pass
        report = verify_raw_corpus(db_path=db, record_offset=25)
        assert report.record_offset == 25

    def test_offset_negative_becomes_zero(self, tmp_path: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        db = tmp_path / "empty.db"
        with open_connection(db):
            pass
        report = verify_raw_corpus(db_path=db, record_offset=-10)
        assert report.record_offset == 0

    def test_defaults_no_db_path(self) -> None:
        # When db_path=None and default DB doesn't exist, should return empty report
        from pathlib import Path
        from unittest.mock import patch

        fake_path = Path("/nonexistent/fake/db.db")
        with patch("polylogue.schemas.verification.default_db_path", return_value=fake_path):
            report = verify_raw_corpus()
            assert report.total_records == 0

    def test_quarantine_malformed_flag_preserved(self, tmp_path: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        db = tmp_path / "empty.db"
        with open_connection(db):
            pass
        # Should not raise even with quarantine_malformed=True on empty DB
        report = verify_raw_corpus(db_path=db, quarantine_malformed=True)
        assert report.total_records == 0

    def test_report_structure_matches_schema(self, tmp_path: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        db = tmp_path / "empty.db"
        with open_connection(db):
            pass
        report = verify_raw_corpus(db_path=db)
        assert hasattr(report, "providers")
        assert hasattr(report, "max_samples")
        assert hasattr(report, "total_records")
        assert hasattr(report, "record_limit")
        assert hasattr(report, "record_offset")

    def test_provider_stats_have_required_fields(self, tmp_path: Path) -> None:
        from polylogue.storage.backends.connection import open_connection

        db = tmp_path / "empty.db"
        with open_connection(db):
            pass
        report = verify_raw_corpus(db_path=db)

        # Even with empty DB, any stats should have these fields
        for stat in report.providers.values():
            assert hasattr(stat, "provider")
            assert hasattr(stat, "total_records")
            assert hasattr(stat, "valid_records")
            assert hasattr(stat, "invalid_records")
            assert hasattr(stat, "drift_records")
            assert hasattr(stat, "decode_errors")


class TestProveRawArtifactCoverage:
    def test_nonexistent_db_returns_empty(self, tmp_path: Path) -> None:
        report = prove_raw_artifact_coverage(db_path=tmp_path / "missing.db")
        assert report.total_records == 0
        assert report.providers == {}

    def test_reports_contracts_sidecars_unknowns_and_decode_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from polylogue.storage.backends.connection import open_connection

        db_path = tmp_path / "proof.db"
        with open_connection(db_path):
            pass

        _insert_raw_record(
            db_path=db_path,
            raw_id="raw-chatgpt-1",
            provider_name="chatgpt",
            source_name="chatgpt",
            source_path="/tmp/chatgpt.json",
            raw_content=b'{"id":"one","mapping":{}}',
        )
        _insert_raw_record(
            db_path=db_path,
            raw_id="raw-sidecar-1",
            provider_name="claude-code",
            source_name="claude-code",
            source_path="/tmp/subagents/agent-a123.meta.json",
            raw_content=b'{"agentType":"general-purpose"}',
        )
        _insert_raw_record(
            db_path=db_path,
            raw_id="raw-subagent-1",
            provider_name="claude-code",
            payload_provider="claude-code",
            source_name="claude-code",
            source_path="/tmp/subagents/agent-a123.jsonl",
            raw_content=(
                b'{"type":"session_meta"}\n'
                b'{"type":"response_item","payload":{"type":"message"}}\n'
            ),
        )
        _insert_raw_record(
            db_path=db_path,
            raw_id="raw-unknown-1",
            provider_name="inbox",
            source_name="inbox",
            source_path="/tmp/unknown.json",
            raw_content=b"42",
        )
        _insert_raw_record(
            db_path=db_path,
            raw_id="raw-codex-1",
            provider_name="codex",
            source_name="codex",
            source_path="/tmp/session.jsonl",
            raw_content=(
                b'{"type":"session_meta"}\n'
                b'not json at all\n'
                b'{"type":"response_item","payload":{"type":"message"}}\n'
            ),
        )

        package = SchemaVersionPackage(
            provider="chatgpt",
            version="v1",
            anchor_kind="conversation_document",
            default_element_kind="conversation_document",
            first_seen="2026-03-01T00:00:00+00:00",
            last_seen="2026-03-01T00:00:00+00:00",
            bundle_scope_count=1,
            sample_count=1,
            elements=[
                SchemaElementManifest(
                    element_kind="conversation_document",
                    schema_file="chatgpt-v1.json",
                    sample_count=1,
                    artifact_count=1,
                )
            ],
        )

        def _resolve_payload(self, provider, payload, *, source_path=None):
            if str(provider) == "chatgpt":
                return SchemaResolution(
                    provider="chatgpt",
                    package_version="v1",
                    element_kind="conversation_document",
                    exact_structure_id=None,
                    bundle_scope=None,
                    reason="exact_structure",
                )
            return None

        def _get_package(self, provider, version="default"):
            if str(provider) == "chatgpt" and version == "v1":
                return package
            return None

        monkeypatch.setattr("polylogue.storage.artifact_observations.SchemaRegistry.resolve_payload", _resolve_payload)
        monkeypatch.setattr("polylogue.storage.artifact_observations.SchemaRegistry.get_package", _get_package)

        report = prove_raw_artifact_coverage(db_path=db_path)

        assert report.total_records == 5
        assert report.contract_backed_records == 1
        assert report.unsupported_parseable_records == 1
        assert report.recognized_non_parseable_records == 1
        assert report.unknown_records == 1
        assert report.decode_errors == 1

        claude_stats = report.providers["claude-code"]
        assert claude_stats.linked_sidecars == 1
        assert claude_stats.orphan_sidecars == 0
        assert claude_stats.subagent_streams == 1
        assert claude_stats.streams_with_sidecars == 1
        assert claude_stats.sidecar_agent_types == {"general-purpose": 1}

        chatgpt_stats = report.providers["chatgpt"]
        assert chatgpt_stats.package_versions == {"v1": 1}
        assert chatgpt_stats.element_kinds == {"conversation_document": 1}
        assert chatgpt_stats.resolution_reasons == {"exact_structure": 1}

        with open_connection(db_path) as conn:
            observation_count = conn.execute(
                "SELECT COUNT(*) FROM artifact_observations"
            ).fetchone()[0]
        assert observation_count == 5

    def test_lists_artifact_rows_and_cohorts_from_durable_control_plane(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = tmp_path / "artifacts.db"
        with open_connection(db_path):
            pass

        _insert_raw_record(
            db_path=db_path,
            raw_id="raw-chatgpt-1",
            provider_name="chatgpt",
            source_name="chatgpt",
            source_path="/tmp/chatgpt.json",
            raw_content=b'{"id":"one","mapping":{}}',
        )
        _insert_raw_record(
            db_path=db_path,
            raw_id="raw-sidecar-1",
            provider_name="claude-code",
            source_name="claude-code",
            source_path="/tmp/subagents/agent-a123.meta.json",
            raw_content=b'{"agentType":"general-purpose"}',
        )
        _insert_raw_record(
            db_path=db_path,
            raw_id="raw-subagent-1",
            provider_name="claude-code",
            source_name="claude-code",
            source_path="/tmp/subagents/agent-a123.jsonl",
            raw_content=(
                b'{"type":"session_meta"}\n'
                b'{"type":"response_item","payload":{"type":"message"}}\n'
            ),
        )

        package = SchemaVersionPackage(
            provider="chatgpt",
            version="v1",
            anchor_kind="conversation_document",
            default_element_kind="conversation_document",
            first_seen="2026-03-01T00:00:00+00:00",
            last_seen="2026-03-01T00:00:00+00:00",
            bundle_scope_count=1,
            sample_count=1,
            elements=[
                SchemaElementManifest(
                    element_kind="conversation_document",
                    schema_file="chatgpt-v1.json",
                    sample_count=1,
                    artifact_count=1,
                )
            ],
        )

        def _resolve_payload(self, provider, payload, *, source_path=None):
            if str(provider) == "chatgpt":
                return SchemaResolution(
                    provider="chatgpt",
                    package_version="v1",
                    element_kind="conversation_document",
                    exact_structure_id=None,
                    bundle_scope=None,
                    reason="exact_structure",
                )
            return None

        def _get_package(self, provider, version="default"):
            if str(provider) == "chatgpt" and version == "v1":
                return package
            return None

        monkeypatch.setattr("polylogue.storage.artifact_observations.SchemaRegistry.resolve_payload", _resolve_payload)
        monkeypatch.setattr("polylogue.storage.artifact_observations.SchemaRegistry.get_package", _get_package)

        rows = list_artifact_observation_rows(db_path=db_path)
        assert len(rows) == 3
        assert {row.artifact_kind for row in rows} == {
            "conversation_document",
            "agent_sidecar_meta",
            "subagent_conversation_stream",
        }

        supported_rows = list_artifact_observation_rows(
            db_path=db_path,
            support_statuses=["supported_parseable"],
        )
        assert len(supported_rows) == 1
        assert supported_rows[0].resolved_package_version == "v1"

        cohorts = list_artifact_cohort_rows(db_path=db_path)
        assert len(cohorts) == 3
        chatgpt_cohort = next(row for row in cohorts if row.provider_name == "chatgpt")
        assert chatgpt_cohort.support_status.value == "supported_parseable"
        assert chatgpt_cohort.resolved_package_version == "v1"
        sidecar_cohort = next(row for row in cohorts if row.artifact_kind == "agent_sidecar_meta")
        assert sidecar_cohort.linked_sidecar_count == 1
