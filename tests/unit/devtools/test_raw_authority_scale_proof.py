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
from polylogue.storage.blob_publication import ArchiveBlobPublisher
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
    assert len(cast(list[object], payload["generation_samples"])) >= 2
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


def test_raw_authority_scale_proof_rechecks_pressure_during_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = iter(
        (
            ProcessSample(1, 1, 0, 0, 0, 0, 0.0, 0.0),
            ProcessSample(1, 1, 0, 0, 0, 0, 2.1, 0.0),
        )
    )
    monkeypatch.setattr("devtools.raw_authority_scale_proof._process_sample", lambda: next(samples))

    with pytest.raises(RuntimeError, match="I/O pressure gate"):
        run_raw_authority_scale_proof(
            tmp_path,
            components=1,
            raws=1,
            prepare_only=True,
            max_io_full_avg10=2.0,
        )


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
    assert profile["component_cohort_distribution"] == [
        {"component_raw_count": 1, "direct_candidate_count": 1, "component_count": 1}
    ]
    assert profile["component_byte_cohort_distribution"] == [
        {
            "component_raw_count": 1,
            "direct_candidate_count": 1,
            "upper_bound_blob_bytes": 64,
            "component_count": 1,
        }
    ]
    serialized = str(profile)
    assert "private-native-id" not in serialized
    assert "/private/source/session.jsonl" not in serialized


def test_raw_authority_scale_profile_selects_candidates_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        for index in range(3):
            archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=f'{{"type":"session_meta","payload":{{"id":"profile-{index}"}}}}\n'.encode(),
                source_path=f"/private/source/profile-{index}.jsonl",
                acquired_at_ms=index,
            )

    original = repair._raw_materialization_candidate_ids
    calls = 0

    def counted(config: Config) -> repair.RawMaterializationCandidates:
        nonlocal calls
        calls += 1
        return original(config)

    monkeypatch.setattr(repair, "_raw_materialization_candidate_ids", counted)
    profile = repair.raw_materialization_scale_profile(
        Config(archive_root=tmp_path, render_root=tmp_path, sources=[], db_path=tmp_path / "index.db")
    )

    assert profile["candidate_count"] == 3
    assert calls == 1


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


def test_raw_authority_scale_proof_keeps_prefix_chain_across_blob_flushes(tmp_path: Path) -> None:
    """A component larger than the publisher batch must continue from its blob."""
    scenario = RawAuthorityScaleScenario(
        components=1,
        direct_candidates=129,
        expanded_candidates=129,
        total_payload_bytes=2_200_000,
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
        assert conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone() == (129,)
    achieved = cast(dict[str, object], payload["achieved_shape"])
    assert achieved["authority_component_count"] == 1
    assert achieved["expanded_candidate_count"] == 129


def test_raw_authority_scale_proof_preserves_exact_private_free_component_cohorts(tmp_path: Path) -> None:
    scenario = RawAuthorityScaleScenario(
        components=2,
        direct_candidates=3,
        expanded_candidates=4,
        total_payload_bytes=4096,
        component_cohorts=((1, 1, 1), (3, 2, 1)),
    )

    payload = run_raw_authority_scale_proof(
        tmp_path,
        scenario=scenario,
        keep=True,
        pass_limit=2,
        max_io_full_avg10=None,
        max_memory_full_avg10=None,
    )

    achieved = cast(dict[str, object], payload["achieved_shape"])
    assert achieved["component_cohort_distribution"] == [
        {"component_raw_count": 1, "direct_candidate_count": 1, "component_count": 1},
        {"component_raw_count": 3, "direct_candidate_count": 2, "component_count": 1},
    ]
    passes = cast(list[dict[str, object]], payload["passes"])
    assert sum(cast(dict[str, int], item["plan_status_counts"])["terminal"] for item in passes) == 1


def test_raw_authority_scale_proof_converges_with_explicit_deferred_cohort(tmp_path: Path) -> None:
    scenario = RawAuthorityScaleScenario(
        components=1,
        direct_candidates=1,
        expanded_candidates=2,
        total_payload_bytes=2048,
        component_cohorts=((2, 1, 1),),
        terminal_sibling_outcome="deferred",
    )

    payload = run_raw_authority_scale_proof(
        tmp_path,
        scenario=scenario,
        pass_limit=1,
        keep=True,
        max_io_full_avg10=None,
        max_memory_full_avg10=None,
    )

    passes = cast(list[dict[str, object]], payload["passes"])
    assert sum(cast(dict[str, int], item["plan_status_counts"])["deferred"] for item in passes) == 1
    digests = cast(list[str], payload["fixed_point_digests"])
    assert digests[0] == digests[1]


def test_raw_authority_scale_proof_reaches_fixed_point_with_resource_deferred_residual(tmp_path: Path) -> None:
    """A bounded envelope may leave durable debt without preventing quiescence."""
    scenario = RawAuthorityScaleScenario(
        components=2,
        direct_candidates=3,
        expanded_candidates=4,
        total_payload_bytes=10_000,
        component_cohorts=((1, 1, 1), (3, 2, 1)),
        component_byte_cohorts=((1, 1, 2_048, 1), (3, 2, 8_192, 1)),
    )

    payload = run_raw_authority_scale_proof(
        tmp_path,
        scenario=scenario,
        pass_limit=2,
        keep=True,
        max_payload_bytes=4_096,
        max_io_full_avg10=None,
        max_memory_full_avg10=None,
    )

    passes = cast(list[dict[str, object]], payload["passes"])
    assert passes[-1]["candidate_count"] == 2
    assert passes[-1]["executable_candidate_count"] == 0
    assert any(item["executable_candidate_count"] == 1 for item in passes)
    assert all(item["executable_candidate_count"] == 0 for item in passes[-3:])
    digests = cast(list[str], payload["fixed_point_digests"])
    assert digests[0] == digests[1]


def test_raw_authority_scale_proof_preserves_private_free_joint_byte_cohorts(tmp_path: Path) -> None:
    expected = [
        {
            "component_raw_count": 1,
            "direct_candidate_count": 1,
            "upper_bound_blob_bytes": 1024,
            "component_count": 1,
        },
        {
            "component_raw_count": 3,
            "direct_candidate_count": 2,
            "upper_bound_blob_bytes": 8192,
            "component_count": 1,
        },
    ]
    scenario = RawAuthorityScaleScenario.from_profile(
        {
            "format": "raw-authority-scale-profile-v1",
            "authority_component_count": 2,
            "candidate_count": 3,
            "expanded_candidate_count": 4,
            "expanded_total_blob_bytes": 6_000,
            "component_cohort_distribution": [
                {"component_raw_count": 1, "direct_candidate_count": 1, "component_count": 1},
                {"component_raw_count": 3, "direct_candidate_count": 2, "component_count": 1},
            ],
            "component_byte_cohort_distribution": expected,
        }
    )

    payload = run_raw_authority_scale_proof(
        tmp_path,
        scenario=scenario,
        keep=True,
        pass_limit=2,
        max_io_full_avg10=None,
        max_memory_full_avg10=None,
    )

    requested = cast(dict[str, object], payload["requested_shape"])
    achieved = cast(dict[str, object], payload["achieved_shape"])
    assert requested["component_byte_cohort_distribution"] == expected
    assert achieved["component_byte_cohort_distribution"] == expected
    assert achieved["expanded_total_blob_bytes"] == 6_000
    root = Path(cast(str, payload["archive_root"]))
    with sqlite3.connect(root / "source.db") as conn:
        largest_blob = conn.execute("SELECT MAX(blob_size) FROM raw_sessions").fetchone()
    assert largest_blob is not None
    assert int(largest_blob[0]) < 4_096


def test_explicit_cohorts_publish_bounded_payloads_without_disk_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scenario = RawAuthorityScaleScenario(
        components=1,
        direct_candidates=2,
        expanded_candidates=2,
        total_payload_bytes=2_048,
        component_cohorts=((2, 2, 1),),
        component_byte_cohorts=((2, 2, 2048, 1),),
    )

    def reject_staged_publication(*_args: object, **_kwargs: object) -> tuple[str, int]:
        raise AssertionError("explicit cohorts must publish bytes without a staged payload path")

    monkeypatch.setattr(ArchiveBlobPublisher, "write_from_path", reject_staged_publication)
    payload = run_raw_authority_scale_proof(
        tmp_path,
        scenario=scenario,
        prepare_only=True,
        max_io_full_avg10=None,
        max_memory_full_avg10=None,
    )

    achieved = cast(dict[str, object], payload["achieved_shape"])
    assert achieved["expanded_total_blob_bytes"] == 2_048
