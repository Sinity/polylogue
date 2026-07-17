from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import pytest

from devtools.raw_authority_scale_proof import ProcessSample, run_raw_authority_scale_proof


def test_raw_authority_scale_proof_reaches_two_matching_quiescent_censuses(tmp_path: Path) -> None:
    payload = run_raw_authority_scale_proof(
        tmp_path,
        components=3,
        raws=5,
        pass_limit=2,
        max_io_full_avg10=None,
        max_memory_full_avg10=None,
    )

    assert payload["requested_shape"] == {"components": 3, "raws": 5, "pass_limit": 2}
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
