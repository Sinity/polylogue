from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import pytest

from devtools.raw_authority_scale_proof import (
    ProcessSample,
    RawAuthorityScaleScenario,
    run_raw_authority_scale_proof,
)
from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.storage import repair
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def test_raw_authority_scale_proof_reaches_two_matching_quiescent_censuses(tmp_path: Path) -> None:
    payload = run_raw_authority_scale_proof(
        tmp_path,
        components=3,
        raws=5,
        pass_limit=2,
        max_io_full_avg10=None,
        max_memory_full_avg10=None,
    )

    assert payload["requested_shape"] == {
        "components": 3,
        "direct_candidates": 5,
        "expanded_candidates": 5,
        "total_payload_bytes": 5120,
        "pass_limit": 2,
    }
    digests = cast(list[str], payload["fixed_point_digests"])
    passes = cast(list[dict[str, object]], payload["passes"])
    receipt = cast(dict[str, object], payload["receipt"])
    phases = cast(list[dict[str, object]], receipt["phases"])
    assert len(digests) == 2
    assert digests[0] == digests[1]
    assert passes[-1]["candidate_count"] == 0
    assert passes[-2]["mode"] == "dry_run"
    assert passes[-1]["mode"] == "dry_run"
    assert passes[-1]["fixed_point"] is True
    assert all(isinstance(item["peak_rss_bytes"], int) and item["peak_rss_bytes"] > 0 for item in passes)
    assert "admission_sample" in payload
    assert receipt["status"] == "succeeded"
    assert isinstance(phases[3]["wall_ms"], int | float) and phases[3]["wall_ms"] > 0
    assert "wall_ms" not in phases[2]


def test_raw_authority_scale_proof_refuses_a_contended_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "devtools.raw_authority_scale_proof._process_sample",
        lambda: ProcessSample(
            rss_bytes=1,
            pss_bytes=1,
            swap_bytes=0,
            cpu_ms=0,
            read_io_bytes=0,
            write_io_bytes=0,
            io_full_avg10=2.1,
            memory_full_avg10=0.0,
        ),
    )

    with pytest.raises(RuntimeError, match="I/O pressure gate"):
        run_raw_authority_scale_proof(tmp_path, max_io_full_avg10=2.0)


def test_raw_authority_scale_proof_consumes_reservations_and_has_stable_corpus_identity(tmp_path: Path) -> None:
    first = run_raw_authority_scale_proof(
        tmp_path / "first",
        components=3,
        raws=5,
        pass_limit=2,
        keep=True,
        max_io_full_avg10=None,
        max_memory_full_avg10=None,
    )
    second = run_raw_authority_scale_proof(
        tmp_path / "second",
        components=3,
        raws=5,
        pass_limit=2,
        keep=True,
        max_io_full_avg10=None,
        max_memory_full_avg10=None,
    )

    first_root = Path(cast(str, first["archive_root"]))
    with sqlite3.connect(first_root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone() == (0,)
    first_receipt = cast(dict[str, object], first["receipt"])
    second_receipt = cast(dict[str, object], second["receipt"])
    assert first_receipt["archive_id"] == second_receipt["archive_id"]


def test_raw_authority_scale_profile_is_aggregate_only(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"private-native-id"}}\n',
            source_path="/private/source/session.jsonl",
            acquired_at_ms=1,
        )

    profile = repair.raw_materialization_scale_profile(
        Config(archive_root=tmp_path, render_root=tmp_path, sources=[], db_path=tmp_path / "index.db")
    )

    assert profile["available"] is True
    assert profile["candidate_count"] == 1
    assert profile["expanded_candidate_count"] == 1
    assert profile["authority_component_count"] == 1
    assert profile["component_raw_count_histogram"] == [{"upper_bound_raw_count": 1, "count": 1}]
    serialized = str(profile)
    assert "private-native-id" not in serialized
    assert "/private/source/session.jsonl" not in serialized


def test_raw_authority_scale_proof_generates_exact_scenario_bytes_and_expansion(tmp_path: Path) -> None:
    scenario = RawAuthorityScaleScenario(
        components=2,
        direct_candidates=3,
        expanded_candidates=4,
        total_payload_bytes=8192,
    )
    payload = run_raw_authority_scale_proof(
        tmp_path,
        scenario=scenario,
        pass_limit=2,
        keep=True,
        prepare_only=True,
        max_io_full_avg10=None,
        max_memory_full_avg10=None,
    )

    root = Path(cast(str, payload["archive_root"]))
    with sqlite3.connect(root / "source.db") as conn:
        assert conn.execute("SELECT COUNT(*), SUM(blob_size) FROM raw_sessions").fetchone() == (4, 8192)
    achieved = cast(dict[str, object], payload["achieved_shape"])
    assert achieved["candidate_count"] == 3
    assert achieved["expanded_candidate_count"] == 4
    assert achieved["authority_component_count"] == 2
    assert achieved["expanded_total_blob_bytes"] == 8192
    assert payload["prepared_only"] is True
