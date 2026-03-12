"""Direct tests for polylogue.schemas.verification module.

Covers ProviderSchemaVerification serialization, SchemaVerificationReport,
and verify_raw_corpus with nonexistent DB and provider filtering.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.schemas.verification import (
    ProviderSchemaVerification,
    SchemaVerificationReport,
    verify_raw_corpus,
)


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
