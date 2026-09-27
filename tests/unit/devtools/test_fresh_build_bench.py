"""The fresh-build benchmark's receipt arithmetic and corpus sealing.

The end-to-end run is exercised by running it; these pin the pure parts a
wrong receipt would come from: the event-log reduction, the stage rollup, the
projection, budgets, the terminal predicate and the corpus seal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from devtools.fresh_build_bench.corpus import (
    SampleSource,
    corpus_from_files,
    load_manifest,
    sample_real,
    seal,
    verify_manifest,
)
from devtools.fresh_build_bench.profile_report import thread_group_label
from devtools.fresh_build_bench.report import (
    _derive_dependents,
    _ts,
    analyse_batches,
    analyse_events,
    compare,
    evaluate_budgets,
    projection,
)
from devtools.fresh_build_bench.run import REQUIRED_READINESS_DOMAINS, Observation, RunConfig, _daemon_env
from devtools.fresh_build_bench.sampler import thread_group


def _event(ts: str, event: str, **fields: object) -> str:
    return json.dumps({"ts": f"2026-09-27T10:00:{ts}Z", "event": event, **fields})


def test_event_reduction_scopes_intake_and_writer_share_to_promotion(tmp_path: Path) -> None:
    """Anti-vacuity: counting the post-promotion chunk moves the intake end to
    50 s, and counting the post-promotion hold raises the busy share above 0.4."""
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            [
                _event("00.000", "daemon.run.start"),
                _event("02.000", "daemon.cold_build.preparation"),
                _event("10.000", "live.ingest.chunk", files=2, bytes=100, duration_ms=5000),
                _event("10.000", "daemon.writer.released", actor="watcher.live_ingest.full", hold_ms=4000, queued=1),
                _event("12.000", "live.ingest.source_group", source_name="codex", files=2, duration_ms=5000),
                _event("20.000", "live.ingest.chunk", files=1, bytes=50, duration_ms=2000),
                _event("20.000", "daemon.cold_build.generation_promoted"),
                _event("50.000", "live.ingest.chunk", files=1, bytes=10, duration_ms=1000),
                _event("50.000", "daemon.writer.released", actor="derivation.session_profile", hold_ms=20000),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = analyse_events(events)
    milestones = summary["milestones_s"]
    assert milestones["promoted_s"] == 20.0
    assert milestones["last_chunk_done_s"] == 20.0
    assert milestones["post_promotion_chunks"] == 1
    assert summary["writer"]["busy_s_to_promotion"] == 4.0
    assert summary["writer"]["busy_share_to_promotion"] == 0.2
    assert summary["writer"]["queue_depth_max"] == 1
    assert summary["by_source"]["codex"] == {"groups": 1, "files": 2, "seconds": 5.0}


def test_stage_rollup_counts_leaf_timers_once(tmp_path: Path) -> None:
    """Anti-vacuity: rolling up the parent ``full.index.full_replace`` as well
    as its children doubles the ``index`` bucket."""
    import sqlite3

    ops = tmp_path / "ops.db"
    with sqlite3.connect(ops) as conn:
        conn.execute("CREATE TABLE daemon_events (kind TEXT, payload_json TEXT)")
        payload = {
            "total_time_s": 10.0,
            "stage_timings_s": {
                "full.index.full_replace": 3.0,
                "full.index.full_replace.messages": 2.0,
                "full.index.full_replace.blocks": 1.0,
                "full.index.prepare": 4.0,
            },
        }
        conn.execute("INSERT INTO daemon_events VALUES ('ingestion_batch', ?)", (json.dumps(payload),))
    batches = analyse_batches(ops)
    buckets = dict(batches["stage_buckets_seconds"])
    assert buckets["index"] == 3.0
    assert buckets["materialize"] == 4.0
    assert batches["totals"]["total_time_s"] == 10.0


def test_projection_scales_origin_rates_to_the_population() -> None:
    manifest = {
        "by_origin": {"codex": {"files": 2, "bytes": 2 << 20}, "claude-code": {"files": 1, "bytes": 1 << 20}},
        "parameters": {"population": {"codex": {"bytes": 200 << 20}, "claude-code": {"bytes": 10 << 20}}},
    }
    by_source = {"codex": {"seconds": 4.0}, "claude-code": {"seconds": 1.0}}
    result = projection(manifest, by_source, intake_wall_s=10.0)
    assert result["by_origin"]["codex"]["projected_s"] == 400.0
    assert result["by_origin"]["claude-code"]["projected_s"] == 10.0
    assert result["wall_scale"] == 2.0
    assert result["projected_intake_hours"] == round(820.0 / 3600, 2)


def test_budgets_fail_closed() -> None:
    rows = evaluate_budgets({"rss_peak_mib": 3072.0, "terminal_s": 60.0}, {"rss_peak_mib": 2000.0})
    assert rows["rss_peak_mib"]["pass"] is True
    assert rows["terminal_s"]["pass"] is False
    with pytest.raises(ValueError, match="unknown budget"):
        evaluate_budgets({"rss": 1.0}, {})


def test_terminal_requires_every_readiness_domain() -> None:
    """Anti-vacuity: accepting a readiness map that lacks a required domain
    makes the first observation terminal."""
    promoted = "/archive/.index-generations/gen/index.db"
    ready = dict.fromkeys(REQUIRED_READINESS_DOMAINS, True)
    partial = Observation(
        1.0, cursor_rows=3, cursor_complete=3, promoted_index=promoted, readiness={"archive_sessions": True}
    )
    complete = Observation(1.0, cursor_rows=3, cursor_complete=3, promoted_index=promoted, readiness=ready)
    unsettled = Observation(
        1.0, cursor_rows=3, cursor_complete=3, memberships_pending=1, promoted_index=promoted, readiness=ready
    )
    assert not partial.terminal
    assert complete.terminal
    assert not unsettled.terminal


def test_seal_detects_a_changed_file(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    transcript = corpus / "home" / ".codex" / "sessions" / "2026" / "rollout-a.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text('{"type":"session_meta"}\n', encoding="utf-8")
    manifest = seal(corpus, kind="sample", parameters={})
    assert manifest["by_origin"] == {"codex": {"files": 1, "bytes": transcript.stat().st_size}}
    verify_manifest(corpus, load_manifest(corpus))
    transcript.write_text('{"type":"session_meta","extra":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="changed size"):
        verify_manifest(corpus, load_manifest(corpus))


def test_seal_detects_a_same_size_edit_and_an_added_file(tmp_path: Path) -> None:
    """Anti-vacuity: checking sizes only, or only the sealed file list,
    accepts both corpora."""
    corpus = tmp_path / "corpus"
    transcript = corpus / "home" / ".codex" / "sessions" / "rollout-a.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text('{"type":"session_meta"}\n', encoding="utf-8")
    seal(corpus, kind="sample", parameters={})
    transcript.write_text('{"type":"session_mete"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="changed content"):
        verify_manifest(corpus, load_manifest(corpus))
    transcript.write_text('{"type":"session_meta"}\n', encoding="utf-8")
    (transcript.parent / "rollout-b.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="added since sealing"):
        verify_manifest(corpus, load_manifest(corpus))


def test_explicit_corpus_admits_only_watched_transcripts(tmp_path: Path) -> None:
    home = tmp_path / "home"
    sessions = home / ".codex" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "rollout.jsonl").write_text("{}\n", encoding="utf-8")
    (sessions / "large.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="not a transcript"):
        corpus_from_files(tmp_path / "bad", [sessions / "large.bin"], home=home)
    manifest = corpus_from_files(tmp_path / "good", [sessions / "rollout.jsonl"], home=home)
    assert manifest["kind"] == "files"


def test_a_sample_that_draws_nothing_still_seals(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    (root / "one.jsonl").write_bytes(b"x" * 1000)
    sources = (SampleSource("codex", root, "home/.codex/sessions", (".jsonl",)),)
    for seed in range(20):
        manifest = sample_real(tmp_path / f"s{seed}", seed=seed, fraction=0.01, sources=sources)
        assert manifest["file_count"] in {0, 1}


def test_event_timestamps_are_utc() -> None:
    assert _ts("1970-01-01T00:00:10.000000Z") == 10.0


def _receipt(**overrides: object) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "label": "r",
        "outcome": "terminal",
        "candidate": {"git_sha": "0" * 40},
        "corpus": {"digest": "c", "total_bytes": 10 << 20, "file_count": 10},
        "config": {"digest": "k"},
        "environment": {"python": "3.14.4", "gil_enabled": False},
        "timing_s": {"promotion": 10.0, "terminal": 20.0},
        "process_tree": {"rss_peak_bytes": 1 << 30},
        "checks": {"promoted": True},
        "budgets": {"promotion_s": {"limit": 15.0}},
        "stages": {},
        "throughput": {},
        "output_fingerprint": {"digest": "same", "tables": {}},
    }
    receipt.update(overrides)
    return receipt


def test_dependent_fields_follow_the_milestones() -> None:
    """Anti-vacuity: keeping throughput, budgets or qualification from an
    earlier promotion time leaves them at 1.0 MiB/s and qualified."""
    receipt = _receipt()
    _derive_dependents(receipt)
    assert receipt["throughput"]["mib_per_s_to_promotion"] == 1.0
    assert receipt["qualified"] is True
    receipt["timing_s"]["promotion"] = 20.0
    _derive_dependents(receipt)
    assert receipt["throughput"]["mib_per_s_to_promotion"] == 0.5
    assert receipt["timing_s"]["derived_after_promotion"] == 0.0
    assert receipt["budgets"]["promotion_s"]["pass"] is False
    assert receipt["qualified"] is False


def test_compare_refuses_different_configs_and_unqualified_runs() -> None:
    """Anti-vacuity: checking only the corpus digest reports IDENTICAL for
    runs made under different configurations."""
    before = _receipt(qualified=True)
    ok, text = compare(before, _receipt(qualified=True))
    assert ok and "IDENTICAL" in text
    ok, text = compare(before, _receipt(qualified=True, config={"digest": "profiled"}))
    assert not ok and "IDENTICAL" not in text
    ok, _text = compare(before, _receipt(qualified=True, environment={"python": "3.14.4", "gil_enabled": True}))
    assert not ok
    unqualified = _receipt(qualified=False, outcome="settle_timeout")
    ok, _text = compare(before, unqualified)
    assert not ok
    ok, text = compare(before, unqualified, allow_unqualified=True)
    assert ok and "WARNING" in text and "IDENTICAL" in text


def test_overrides_cannot_redirect_the_isolated_archive(tmp_path: Path) -> None:
    config = RunConfig(
        corpus=tmp_path,
        work=tmp_path,
        candidate=tmp_path,
        python="python",
        label="override",
        extra_env=(("POLYLOGUE_ARCHIVE_ROOT", "/elsewhere"),),
    )
    paths = {name: tmp_path / name for name in ("home", "xdg", "tmp", "archive", "config", "events", "stacks")}
    with pytest.raises(ValueError, match="driver-owned"):
        _daemon_env(config, paths)


def test_sample_is_seeded_and_keeps_session_units_together(tmp_path: Path) -> None:
    """Anti-vacuity: sampling files instead of session units separates a
    subagent transcript from its parent in some seed."""
    home = tmp_path / "home"
    projects = home / ".claude" / "projects" / "proj"
    for index in range(20):
        session = projects / f"s{index:02d}.jsonl"
        session.parent.mkdir(parents=True, exist_ok=True)
        session.write_bytes(b"x" * (1000 + index))
        child = projects / f"s{index:02d}" / "subagents" / "agent-a.jsonl"
        child.parent.mkdir(parents=True)
        child.write_bytes(b"y" * 500)
    sources = (SampleSource("claude-code", home / ".claude" / "projects", "home/.claude/projects", (".jsonl",), True),)
    first = sample_real(tmp_path / "a", seed=5, fraction=0.3, sources=sources)
    again = sample_real(tmp_path / "b", seed=5, fraction=0.3, sources=sources)
    assert first["digest"] == again["digest"]
    paths = {row[0] for row in first["files"]}
    parents = {path for path in paths if "subagents" not in path}
    assert parents
    for parent in parents:
        assert parent.removesuffix(".jsonl") + "/subagents/agent-a.jsonl" in paths
    assert 0.1 < first["total_bytes"] / first["parameters"]["population"]["claude-code"]["bytes"] < 0.6


def test_pool_workers_collapse_into_one_thread_group() -> None:
    assert thread_group("ThreadPoolExecutor-3_17") == "ThreadPoolExecutor"
    assert thread_group("polylogue-writer:watcher.live_ingest.full") == "polylogue-writer:watcher.live_ingest.full"
    assert thread_group_label("polylogue-writer:watcher.live_ingest.full") == "polylogue-writer"


def test_thread_cpu_separates_writer_actors_from_other_threads() -> None:
    """Anti-vacuity: grouping by the collapsed label merges every writer
    actor into one row and loses the per-actor split."""
    from devtools.fresh_build_bench.report import thread_cpu_summary

    summary = thread_cpu_summary(
        {
            "clock_ticks_per_s": 100,
            "thread_cpu_ticks": {
                "polylogue-writer:watcher.live_ingest.full": 300,
                "polylogue-writer:derivation.session_profile": 100,
                "MainThread": 50,
            },
        }
    )
    assert summary["writer_total"] == 4.0
    assert summary["writer_by_actor"] == [["watcher.live_ingest.full", 3.0], ["derivation.session_profile", 1.0]]
    assert summary["other_threads"] == [["MainThread", 0.5]]


def test_event_times_share_the_driver_clock_and_stop_at_promotion(tmp_path: Path) -> None:
    """Anti-vacuity: measuring from ``daemon.run.start`` shifts promotion to
    18 s; charging the post-promotion group doubles codex's seconds."""
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            [
                _event("02.000", "daemon.run.start"),
                _event("10.000", "live.ingest.source_group", source_name="codex", files=1, duration_ms=3000),
                _event("20.000", "daemon.cold_build.generation_promoted"),
                _event("30.000", "live.ingest.source_group", source_name="codex", files=1, duration_ms=3000),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    origin = _ts("2026-09-27T10:00:00.000000Z")
    summary = analyse_events(events, origin_unix=origin)
    assert summary["milestones_s"]["promoted_s"] == 20.0
    assert summary["by_source"]["codex"] == {"groups": 1, "files": 1, "seconds": 3.0}
    assert analyse_events(tmp_path / "absent.jsonl") == {"event_count": 0}


def test_stratum_boundary_is_one_draw(tmp_path: Path) -> None:
    """Anti-vacuity: redrawing the boundary on every later unit selects a
    file in nearly every seed instead of about one in ten."""
    root = tmp_path / "src"
    root.mkdir()
    for index in range(50):
        (root / f"f{index:02d}.jsonl").write_bytes(b"x" * 1000)
    sources = (SampleSource("codex", root, "home/.codex/sessions", (".jsonl",)),)
    seeds = 60
    selected = sum(
        sample_real(tmp_path / f"s{seed}", seed=seed, fraction=0.002, sources=sources)["file_count"]
        for seed in range(seeds)
    )
    # Expected 0.1 per seed; allow generous noise, far below "every seed".
    assert selected <= seeds * 0.35
