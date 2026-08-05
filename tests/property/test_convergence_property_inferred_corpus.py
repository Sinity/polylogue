"""Persisted inferred-origin receipts feed the real multi-origin corpus."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from polylogue.core.sources import origin_from_provider
from tests.infra.convergence_harness import (
    append_convergence_members,
    append_convergence_unsupported_receipts,
    assert_archives_equivalent,
    build_converged_archive,
    rich_convergence_pathology,
)
from tests.infra.inferred_corpus import (
    assert_inferred_corpus_convergence_handoff_complete,
    build_inferred_corpus_convergence_handoff,
)


def test_convergence_property_persisted_origins_and_unsupported_receipts(tmp_path: Path) -> None:
    corpus = rich_convergence_pathology()
    assert corpus.manifest is not None
    handoff = build_inferred_corpus_convergence_handoff(corpus.manifest)
    assert_inferred_corpus_convergence_handoff_complete(corpus.manifest, handoff)
    assert len(corpus.members) == len(handoff.selections)
    assert {(member.provider, member.spec.package_version, member.spec.element_kind) for member in corpus.members} == {
        (spec.provider, spec.package_version, spec.element_kind) for spec in handoff.specs
    }
    assert all(member.receipt is not None for member in corpus.members)
    unsupported = {entry.unsupported.reason for entry in corpus.manifest.entries if entry.unsupported is not None}
    assert "unsupported_wire_route" in unsupported
    archive = build_converged_archive(tmp_path / "archive", corpus)
    with sqlite3.connect(archive.root / "index.db") as conn:
        origins = {str(row[0]) for row in conn.execute("SELECT DISTINCT origin FROM sessions")}
        assert origins == {origin_from_provider(member.provider).value for member in corpus.members}
        assert conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0] > 0

    replay = build_converged_archive(tmp_path / "replay", corpus)
    assert_archives_equivalent(archive, replay)


def test_inferred_corpus_handoff_is_anti_vacuous() -> None:
    corpus = rich_convergence_pathology()
    assert corpus.manifest is not None
    handoff = build_inferred_corpus_convergence_handoff(corpus.manifest)
    with pytest.raises(AssertionError, match="omitted or substituted"):
        assert_inferred_corpus_convergence_handoff_complete(
            corpus.manifest,
            replace(handoff, selections=handoff.selections[:-1]),
        )


def test_append_partition_preserves_every_jsonl_selection() -> None:
    corpus = rich_convergence_pathology()
    jsonl_keys = {
        (member.provider, member.spec.package_version, member.spec.element_kind)
        for member in corpus.members
        if member.selection.wire_format.encoding == "jsonl"
    }
    append_keys = {
        (member.provider, member.spec.package_version, member.spec.element_kind)
        for member in append_convergence_members()
    }
    unsupported_keys = {
        (receipt["provider"], receipt["package_version"], receipt["element_kind"])
        for receipt in append_convergence_unsupported_receipts()
    }
    assert append_keys.isdisjoint(unsupported_keys)
    assert append_keys | unsupported_keys == jsonl_keys
    assert all(receipt["status"] == "unsupported" for receipt in append_convergence_unsupported_receipts())


def test_inferred_corpus_route_cannot_bypass_parser_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import polylogue.sources.dispatch as dispatch
    import polylogue.sources.emitter as emitter

    def fail_parser(*args: object, **kwargs: object) -> object:
        raise AssertionError("parser dispatch was bypassed")

    monkeypatch.setattr(dispatch, "parse_payload", fail_parser)
    monkeypatch.setattr(dispatch, "parse_stream_payload", fail_parser)
    monkeypatch.setattr(emitter, "parse_payload", fail_parser)
    corpus = rich_convergence_pathology()
    with pytest.raises(AssertionError, match="production parser dispatch or ingest dropped"):
        build_converged_archive(tmp_path / "bypassed", type(corpus)((corpus.members[0],)))


def test_inferred_corpus_route_cannot_skip_fts_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.infra.convergence_harness as harness

    monkeypatch.setattr(harness, "make_fts_stage", harness.make_insights_stage)
    corpus = rich_convergence_pathology()
    with pytest.raises(AssertionError, match="omitted a required stage"):
        build_converged_archive(tmp_path / "skipped", type(corpus)((corpus.members[0],)))
